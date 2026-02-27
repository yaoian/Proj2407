from __future__ import annotations

import base64
import csv
import io
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .time_interp import guess_traj_time_interp

_MODEL_CACHE: Dict[Tuple[str, Optional[str]], Tuple[object, object]] = {}
_ROOMS_MAPPING_CACHE: Dict[str, List[Tuple[str, int, int, int, int]]] = {}
_PORTS_MAPPING_CACHE: Dict[str, Dict[Tuple[int, int], str]] = {}
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.S | re.I)
_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
_THINK_END_RE = re.compile(r"</\s*think\s*>", flags=re.I)
_ROLE_PREFIX_RE = re.compile(r"^(?:system|user|assistant)\s*[:：]?\s*", flags=re.I)


def load_norm_stats(norm_stats_path: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    path = Path(norm_stats_path)
    with path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    mean = data.get("mean")
    std = data.get("std")
    if not (isinstance(mean, list) and isinstance(std, list) and len(mean) >= 3 and len(std) >= 3):
        raise ValueError(f"Invalid norm stats JSON: {path}")
    mean_tuple = (float(mean[0]), float(mean[1]), float(mean[2]))
    std_tuple = (float(std[0]), float(std[1]), float(std[2]))
    if std_tuple[0] <= 0 or std_tuple[1] <= 0:
        raise ValueError(f"Invalid std in norm stats JSON: {path}")
    return mean_tuple, std_tuple


def denorm_xy(traj_0: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_x, mean_y = float(mean[0]), float(mean[1])
    std_x, std_y = float(std[0]), float(std[1])
    xy = traj_0[:2]
    xy_denorm = torch.empty_like(xy)
    xy_denorm[0] = xy[0] * std_x + mean_x
    xy_denorm[1] = xy[1] * std_y + mean_y
    return xy_denorm


def normalize_xy(xy: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    mean_x, mean_y = float(mean[0]), float(mean[1])
    std_x, std_y = float(std[0]), float(std[1])
    xy_norm = torch.empty_like(xy)
    xy_norm[0] = (xy[0] - mean_x) / std_x
    xy_norm[1] = (xy[1] - mean_y) / std_y
    return xy_norm


def _xy_to_pixel(
    x_coord: float,
    y_coord: float,
    map_extent: Sequence[float],
    image_size: Tuple[int, int],
    pixel_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[float, float]:
    x_min, x_max, y_min, y_max = map_extent
    width, height = image_size
    left, top, right, bottom = 0.0, 0.0, float(width - 1), float(height - 1)
    if pixel_bbox is not None:
        left, top, right, bottom = (float(pixel_bbox[0]), float(pixel_bbox[1]), float(pixel_bbox[2]), float(pixel_bbox[3]))
    if x_max == x_min or y_max == y_min:
        raise ValueError("Invalid map_extent with zero range.")
    cols = float(y_max - y_min)
    rows = float(x_max - x_min)
    span_w = right - left
    span_h = bottom - top
    if span_w <= 0 or span_h <= 0:
        raise ValueError("Invalid pixel_bbox with non-positive span.")
    cell_w = span_w / cols
    cell_h = span_h / rows
    pixel_x = left + (y_coord - y_min + 0.5) * cell_w
    pixel_y = top + (x_coord - x_min + 0.5) * cell_h
    pixel_x = max(left, min(right, float(pixel_x)))
    pixel_y = max(top, min(bottom, float(pixel_y)))
    return pixel_x, pixel_y


def _detect_grid_bbox(image, *, border_tol: int = 10, ratio_threshold: float = 0.9) -> Optional[Tuple[int, int, int, int]]:
    width, height = image.size
    pixels = image.load()
    border_color = pixels[0, 0]

    def is_border_color(color) -> bool:
        return max(abs(int(color[i]) - int(border_color[i])) for i in range(3)) <= border_tol

    top = 0
    for y in range(height):
        count = 0
        for x in range(width):
            if is_border_color(pixels[x, y]):
                count += 1
        if count / float(width) < ratio_threshold:
            top = y
            break

    bottom = height - 1
    for y in range(height - 1, -1, -1):
        count = 0
        for x in range(width):
            if is_border_color(pixels[x, y]):
                count += 1
        if count / float(width) < ratio_threshold:
            bottom = y
            break

    left = 0
    for x in range(width):
        count = 0
        for y in range(height):
            if is_border_color(pixels[x, y]):
                count += 1
        if count / float(height) < ratio_threshold:
            left = x
            break

    right = width - 1
    for x in range(width - 1, -1, -1):
        count = 0
        for y in range(height):
            if is_border_color(pixels[x, y]):
                count += 1
        if count / float(height) < ratio_threshold:
            right = x
            break

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _load_rooms_mapping(rooms_mapping_path: str) -> List[Tuple[str, int, int, int, int]]:
    path = str(Path(rooms_mapping_path))
    cached = _ROOMS_MAPPING_CACHE.get(path)
    if cached is not None:
        return cached

    rooms: List[Tuple[str, int, int, int, int]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            try:
                name = str(row.get("name", "")).strip()
                x1 = int(float(row.get("x1", "0")))
                y1 = int(float(row.get("y1", "0")))
                x2 = int(float(row.get("x2", "0")))
                y2 = int(float(row.get("y2", "0")))
            except Exception:
                continue
            if not name:
                continue
            x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
            y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
            rooms.append((name, x_lo, y_lo, x_hi, y_hi))

    _ROOMS_MAPPING_CACHE[path] = rooms
    return rooms


def _load_ports_mapping(ports_mapping_path: str) -> Dict[Tuple[int, int], str]:
    path = str(Path(ports_mapping_path))
    cached = _PORTS_MAPPING_CACHE.get(path)
    if cached is not None:
        return cached

    ports: Dict[Tuple[int, int], str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            try:
                name = str(row.get("name", "")).strip()
                kind = str(row.get("kind", "")).strip()
                x = int(float(row.get("x", "0")))
                y = int(float(row.get("y", "0")))
            except Exception:
                continue
            if not name and not kind:
                continue
            label = f"{kind}:{name}" if kind and name else (name or kind)
            ports[(x, y)] = label

    _PORTS_MAPPING_CACHE[path] = ports
    return ports


def _room_for_cell(x: int, y: int, rooms: Sequence[Tuple[str, int, int, int, int]]) -> str:
    for name, x1, y1, x2, y2 in rooms:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return "Corridor"


def _parse_hex_color(value: str) -> Optional[Tuple[int, int, int]]:
    value = value.strip()
    if not value or value.lower() in {"none", "transparent"}:
        return None
    if value.startswith("#"):
        value = value[1:]
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        return None
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
    except Exception:
        return None
    return r, g, b


def _float_attr(attrs: dict, key: str, default: float = 0.0) -> float:
    raw = attrs.get(key)
    if raw is None:
        return default
    try:
        return float(str(raw))
    except Exception:
        return default


def _render_floor1_svg(svg_path: str):
    from PIL import Image, ImageDraw

    root = ET.parse(svg_path).getroot()
    width = int(round(_float_attr(root.attrib, "width", 0.0)))
    height = int(round(_float_attr(root.attrib, "height", 0.0)))
    if width <= 0 or height <= 0:
        # fall back to common size in this repo
        width, height = 832, 496

    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def merge_style(parent: dict, elem) -> dict:
        style = dict(parent)
        for key in ("stroke", "stroke-width", "stroke-dasharray", "fill"):
            if key in elem.attrib:
                style[key] = elem.attrib[key]
        return style

    def draw_line(x1: float, y1: float, x2: float, y2: float, style: dict) -> None:
        stroke = _parse_hex_color(str(style.get("stroke", "")))
        if stroke is None:
            return
        width_px = max(1, int(round(float(style.get("stroke-width", 1.0)))))
        dash = str(style.get("stroke-dasharray", "")).strip()
        start = (float(x1), float(y1))
        end = (float(x2), float(y2))
        if dash:
            parts = [p.strip() for p in dash.split(",") if p.strip()]
            dash_len = float(parts[0]) if parts else 6.0
            gap_len = float(parts[1]) if len(parts) > 1 else dash_len
            _draw_dashed_line(draw, start, end, fill=stroke, width=width_px, dash_length=dash_len, gap_length=gap_len)
        else:
            draw.line([start, end], fill=stroke, width=width_px)

    def draw_rect(x: float, y: float, w: float, h: float, style: dict) -> None:
        fill = _parse_hex_color(str(style.get("fill", "")))
        stroke = _parse_hex_color(str(style.get("stroke", "")))
        width_px = max(1, int(round(float(style.get("stroke-width", 1.0))))) if stroke else 1
        left = float(x)
        top = float(y)
        right = float(x + w)
        bottom = float(y + h)
        bbox = [left, top, right, bottom]
        if stroke is None:
            draw.rectangle(bbox, fill=fill)
        else:
            draw.rectangle(bbox, fill=fill, outline=stroke, width=width_px)

    def walk(elem, style: dict) -> None:
        style2 = merge_style(style, elem)
        tag = local_name(elem.tag)
        if tag == "line":
            x1 = _float_attr(elem.attrib, "x1")
            y1 = _float_attr(elem.attrib, "y1")
            x2 = _float_attr(elem.attrib, "x2")
            y2 = _float_attr(elem.attrib, "y2")
            draw_line(x1, y1, x2, y2, style2)
        elif tag == "rect":
            x = _float_attr(elem.attrib, "x")
            y = _float_attr(elem.attrib, "y")
            w = _float_attr(elem.attrib, "width")
            h = _float_attr(elem.attrib, "height")
            if w > 0 and h > 0:
                draw_rect(x, y, w, h, style2)
        for child in list(elem):
            walk(child, style2)

    walk(root, {})
    return image


def _load_map_image(map_image_path: str):
    from PIL import Image

    path = Path(map_image_path)
    suffix = path.suffix.lower()
    if suffix == ".svg":
        # Purpose-built renderer for our floor1.svg to avoid extra deps.
        return _render_floor1_svg(str(path))
    return Image.open(map_image_path).convert("RGB")


def _svg_grid_bbox(svg_path: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract grid bbox from our generated floor1.svg.

    floor1.svg contains an explicit rectangle for the grid border, e.g.:
      <rect x="56" y="56" width="720" height="384" fill="none" stroke="#111" ... />

    We return (left, top, right, bottom) in the SVG coordinate system, which matches
    the rasterized PNG pixels when rendered at the SVG width/height.
    """
    try:
        root = ET.parse(svg_path).getroot()
    except Exception:
        return None

    def local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    for elem in root.iter():
        if local_name(elem.tag) != "rect":
            continue
        stroke = (elem.attrib.get("stroke") or "").strip().lower()
        fill = (elem.attrib.get("fill") or "").strip().lower()
        if stroke not in {"#111", "#111111"}:
            continue
        if fill not in {"none", ""}:
            continue
        try:
            x = float(elem.attrib.get("x", "0"))
            y = float(elem.attrib.get("y", "0"))
            w = float(elem.attrib.get("width", "0"))
            h = float(elem.attrib.get("height", "0"))
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        left = int(round(x))
        top = int(round(y))
        right = int(round(x + w))
        bottom = int(round(y + h))
        return left, top, right, bottom
    return None


def _draw_dashed_line(
    draw,
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    *,
    fill,
    width: int,
    dash_length: float = 6.0,
    gap_length: float = 4.0,
) -> None:
    delta_x = end_point[0] - start_point[0]
    delta_y = end_point[1] - start_point[1]
    segment_length = math.hypot(delta_x, delta_y)
    if segment_length <= 1e-6:
        return
    unit_x = delta_x / segment_length
    unit_y = delta_y / segment_length
    cursor = 0.0
    while cursor < segment_length:
        dash_end = min(cursor + dash_length, segment_length)
        seg_start = (start_point[0] + unit_x * cursor, start_point[1] + unit_y * cursor)
        seg_end = (start_point[0] + unit_x * dash_end, start_point[1] + unit_y * dash_end)
        draw.line([seg_start, seg_end], fill=fill, width=width)
        cursor += dash_length + gap_length


def render_traj_on_map(
    xy_denorm: torch.Tensor,
    erase_mask: torch.Tensor,
    map_image_path: str,
    *,
    map_extent: Sequence[float] = (0.0, 16.0, 0.0, 30.0),
    line_width: int = 2,
    render_out_path: Optional[str] = None,
):
    from PIL import ImageDraw

    image = _load_map_image(map_image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    pixel_bbox = None
    if str(map_image_path).lower().endswith(".svg"):
        pixel_bbox = _svg_grid_bbox(map_image_path)
    if pixel_bbox is None:
        pixel_bbox = _detect_grid_bbox(image)

    valid_mask = erase_mask >= 0
    missing_mask = erase_mask > 0.1

    pixel_points: List[Optional[Tuple[float, float]]] = []
    for idx in range(int(xy_denorm.shape[1])):
        if not bool(valid_mask[idx].item()):
            pixel_points.append(None)
            continue
        x_coord = float(xy_denorm[0, idx].item())
        y_coord = float(xy_denorm[1, idx].item())
        if not (math.isfinite(x_coord) and math.isfinite(y_coord)):
            pixel_points.append(None)
            continue
        x_cell = float(int(round(x_coord)))
        y_cell = float(int(round(y_coord)))
        pixel_points.append(_xy_to_pixel(x_cell, y_cell, map_extent, (width, height), pixel_bbox))

    observed_point_color = (220, 20, 60)

    # draw observed points bigger for visibility
    for idx, point in enumerate(pixel_points):
        if point is None:
            continue
        if bool(missing_mask[idx].item()):
            continue
        radius = 5
        draw.ellipse(
            [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
            outline=observed_point_color,
            fill=observed_point_color,
        )

    if render_out_path:
        image.save(render_out_path)
    return image


def _format_traj_for_prompt(
    xy_denorm: torch.Tensor,
    time_denorm: Optional[torch.Tensor],
    erase_mask: torch.Tensor,
    *,
    rooms_mapping_path: Optional[str] = None,
    ports_mapping_path: Optional[str] = None,
    digits: int = 3,
) -> str:
    rooms = None
    ports = None
    if rooms_mapping_path:
        try:
            rooms = _load_rooms_mapping(rooms_mapping_path)
        except Exception:
            rooms = None
    if ports_mapping_path:
        try:
            ports = _load_ports_mapping(ports_mapping_path)
        except Exception:
            ports = None

    if rooms is not None or ports is not None:
        lines = ["idx,time,x,y,missing,room,port"]
    else:
        lines = ["idx,time,x,y,missing"]
    missing_mask = erase_mask > 0.1
    valid_mask = erase_mask >= 0
    total_len = int(xy_denorm.shape[1])
    for idx in range(total_len):
        if not bool(valid_mask[idx].item()):
            continue
        missing = int(bool(missing_mask[idx].item()))
        x_val = float(xy_denorm[0, idx].item())
        y_val = float(xy_denorm[1, idx].item())
        time_val = None
        if time_denorm is not None:
            time_val = float(time_denorm[idx].item())
        if missing:
            x_text = "NA"
            y_text = "NA"
            room_text = "NA"
            port_text = "NA"
        else:
            x_int = int(round(x_val))
            y_int = int(round(y_val))
            x_text = str(x_int)
            y_text = str(y_int)
            if rooms is not None:
                room_text = _room_for_cell(x_int, y_int, rooms)
            else:
                room_text = "NA"
            if ports is not None:
                port_text = ports.get((x_int, y_int), "NA")
            else:
                port_text = "NA"
        if time_val is None:
            time_text = "NA"
        else:
            time_text = f"{time_val:.{digits}f}"
        if rooms is not None or ports is not None:
            lines.append(f"{idx},{time_text},{x_text},{y_text},{missing},{room_text},{port_text}")
        else:
            lines.append(f"{idx},{time_text},{x_text},{y_text},{missing}")
    return "\n".join(lines)


def build_qwen_prompt(
    xy_denorm: torch.Tensor,
    erase_mask: torch.Tensor,
    *,
    time_denorm: Optional[torch.Tensor] = None,
    map_extent: Sequence[float] = (0.0, 16.0, 0.0, 30.0),
    use_map: bool = True,
    rooms_mapping_path: Optional[str] = None,
    ports_mapping_path: Optional[str] = None,
) -> str:
    x_min, x_max, y_min, y_max = map_extent
    traj_text = _format_traj_for_prompt(
        xy_denorm,
        time_denorm,
        erase_mask,
        rooms_mapping_path=rooms_mapping_path,
        ports_mapping_path=ports_mapping_path,
    )
    lines: List[str] = ["你是一名轨迹修复助手。"]
    if use_map:
        lines.append("已给出室内地图与轨迹示意图（来自 Dataset/Indoor Sensor/floor1.svg）：")
        lines.append("- 红色点：观测到的轨迹点（只显示可视点）")
        lines.append("- 红色实线：墙壁，无法通行")
        lines.append("- 红色虚线：障碍物，禁止直接通行")
    else:
        lines.append("未提供地图图像，仅提供轨迹表与通行规则。")
    lines.append("坐标系：x 轴向下递增，y 轴向右递增。")
    lines.append(f"地图范围：x in [{x_min}, {x_max}], y in [{y_min}, {y_max}]。")
    lines.append("轨迹表中的 (x,y) 为方格编号，对齐 floor1.svg 的刻度。")
    if rooms_mapping_path or ports_mapping_path:
        lines.append("轨迹表中的 room/port 列由 mapping CSV 解析得到，用于辅助判断（不需要输出）。")
    lines.append("通行规则：")
    lines.append("1) 不得穿越红色实线墙壁。")
    lines.append("2) 红色虚线障碍物处，禁止直接相邻通行（仅改变 x 或仅改变 y 的移动）。")
    lines.append("3) 允许斜向绕行（x 与 y 同时变化的移动）。")
    if use_map:
        lines.append("请根据图像与下方轨迹表，补全所有 missing==1 的点，输出 JSON：")
    else:
        lines.append("请根据轨迹表，补全所有 missing==1 的点，输出 JSON：")
    lines.append("若包含思考过程，请将其放在内部，最终答案仅保留 JSON。")
    lines.append("{\"points\": [[idx, x0, y0], [idx, x1, y1], ...]} 只输出缺失点。")
    lines.append("不要输出除 JSON 外的任何文字。")
    return "\n".join(lines) + f"\n\n轨迹表：\n{traj_text}\n"


def _run_qwen_vl(
    image,
    prompt: str,
    *,
    model_id: str,
    max_new_tokens: int,
    device: Optional[str],
    force_download: bool,
    return_model_info: bool = False,
) -> str | tuple[str, dict]:
    from transformers import AutoModelForVision2Seq, AutoProcessor

    cache_key = (model_id, device)
    if cache_key in _MODEL_CACHE and not force_download:
        processor, model = _MODEL_CACHE[cache_key]
    else:
        processor = AutoProcessor.from_pretrained(model_id, force_download=force_download)
        device_map = "auto" if device is None else None
        max_memory = None
        offload_buffers = None
        if device_map == "auto":
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                try:
                    total_memory = int(torch.xpu.get_device_properties(0).total_memory)
                except Exception:
                    total_memory = 0
                if total_memory > 0:
                    # Some IPEX/PyTorch builds report incorrect free memory via mem_get_info(),
                    # which can cause accelerate to assign 0 modules to XPU. Use a conservative
                    # fraction of total memory instead.
                    usable = int(total_memory * 0.85)
                    gib = max(1, usable // (1024**3))
                    # Opt-in XPU placement. When this env var is unset/<=0 we avoid forcing any weights onto XPU,
                    # which keeps the default transformers/accelerate device_map behavior (often CPU+disk on low-RAM
                    # systems). Set e.g. `QWEN_VL_XPU_MAX_GIB=2` to try placing a small subset of modules on XPU.
                    cap_text = os.getenv("QWEN_VL_XPU_MAX_GIB")
                    if cap_text:
                        try:
                            cap_gib = int(cap_text)
                        except Exception:
                            cap_gib = 0
                        if cap_gib > 0:
                            gib = max(1, min(gib, cap_gib))
                            max_memory = {0: f"{gib}GiB"}
                            offload_buffers = True
            if max_memory is not None and "cpu" not in max_memory:
                try:
                    import psutil  # accelerate already depends on this

                    max_memory["cpu"] = int(psutil.virtual_memory().available * 0.90)
                except Exception:
                    # Fallback to a reasonable default; accelerate will validate the key.
                    max_memory["cpu"] = "32GiB"

        model_kwargs = {"device_map": device_map, "force_download": force_download}
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
        if offload_buffers is not None:
            model_kwargs["offload_buffers"] = offload_buffers

        desired_dtype = "auto"
        xpu_available = bool(getattr(torch, "xpu", None) is not None and torch.xpu.is_available())
        if xpu_available and device_map is not None and isinstance(max_memory, dict) and 0 in max_memory:
            desired_dtype = torch.bfloat16

        def _load_model():
            try:
                return AutoModelForVision2Seq.from_pretrained(model_id, dtype=desired_dtype, **model_kwargs)
            except TypeError:
                return AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=desired_dtype, **model_kwargs)

        xpu_gib = None
        if isinstance(max_memory, dict) and 0 in max_memory and isinstance(max_memory[0], str) and max_memory[0].endswith("GiB"):
            try:
                xpu_gib = int(max_memory[0].rstrip("GiB"))
            except Exception:
                xpu_gib = None

        import transformers.modeling_utils as modeling_utils

        original_warmup = getattr(modeling_utils, "caching_allocator_warmup", None)

        def _safe_warmup(*args, **kwargs):
            if original_warmup is None:
                return None
            try:
                return original_warmup(*args, **kwargs)
            except RuntimeError as exc:
                if "doesn't support querying the available free memory" in str(exc):
                    return None
                raise

        if xpu_available and device_map is not None and original_warmup is not None:
            modeling_utils.caching_allocator_warmup = _safe_warmup

        try:
            candidates = None
            if xpu_available and device_map == "auto" and xpu_gib is not None and isinstance(max_memory, dict):
                # Keep candidates small by default (see cap above), then back off if needed.
                candidates = sorted(set([xpu_gib, 2, 1]), reverse=True)

            if not candidates:
                model = _load_model()
            else:
                import gc

                last_exc: Optional[RuntimeError] = None
                for candidate_gib in candidates:
                    max_memory[0] = f"{int(candidate_gib)}GiB"
                    model_kwargs["max_memory"] = max_memory
                    try:
                        model = _load_model()
                        break
                    except RuntimeError as exc:
                        msg = str(exc)
                        if "XPU out of memory" not in msg and " out of memory" not in msg:
                            raise
                        last_exc = exc
                        try:
                            torch.xpu.empty_cache()
                        except Exception:
                            pass
                        gc.collect()
                else:
                    raise last_exc if last_exc is not None else RuntimeError("XPU OOM while loading model")
        finally:
            if original_warmup is not None:
                modeling_utils.caching_allocator_warmup = original_warmup
        if device is not None:
            model = model.to(device)
        _MODEL_CACHE[cache_key] = (processor, model)

    model_info = None
    if return_model_info:
        model_info = {
            "device": str(getattr(model, "device", "unknown")),
            "hf_device_map": getattr(model, "hf_device_map", None),
            "xpu_available": bool(getattr(torch, "xpu", None) is not None and torch.xpu.is_available()),
        }

    if image is None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if hasattr(processor, "apply_chat_template"):
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[prompt_text], return_tensors="pt")
        else:
            inputs = processor(text=[prompt], return_tensors="pt")
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if hasattr(processor, "apply_chat_template"):
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
        else:
            inputs = processor(text=[prompt], images=[image], return_tensors="pt")

    def _move_tensors_to_device(batch, target_device: str):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(target_device) if isinstance(value, torch.Tensor) else value
        return moved

    def _normalize_device_name(value) -> Optional[str]:
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, int):
            # accelerate may use int GPU ids in device_map
            if torch.cuda.is_available():
                return f"cuda:{value}"
            return None
        if isinstance(value, str):
            text = value.strip().lower()
            if not text or text in {"cpu", "disk"}:
                return None
            if text.isdigit() and torch.cuda.is_available():
                return f"cuda:{text}"
            return text
        return None

    def _pick_runtime_input_device() -> Optional[str]:
        if device is not None:
            return str(device)

        model_device = _normalize_device_name(getattr(model, "device", None))
        if model_device is not None:
            return model_device

        hf_device_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_device_map, dict) and hf_device_map:
            for raw_dev in hf_device_map.values():
                normalized = _normalize_device_name(raw_dev)
                if normalized is not None:
                    return normalized
        return None

    runtime_input_device = _pick_runtime_input_device()
    if runtime_input_device is not None:
        inputs = _move_tensors_to_device(inputs, runtime_input_device)
    generation_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
    }
    generated_ids = model.generate(**inputs, **generation_kwargs)

    input_ids = inputs.get("input_ids")
    input_len = None
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 2:
        input_len = int(input_ids.shape[1])

    ids_for_decode = generated_ids
    if (
        input_len is not None
        and isinstance(generated_ids, torch.Tensor)
        and generated_ids.ndim == 2
        and generated_ids.shape[1] > input_len
    ):
        # Decoder-only chat models return prompt+completion; keep only new tokens.
        ids_for_decode = generated_ids[:, input_len:]

    output = processor.batch_decode(ids_for_decode, skip_special_tokens=True)[0]
    if not output.strip() and ids_for_decode is not generated_ids:
        # Fallback for model variants that return completion-only ids already.
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output = output.strip()
    if return_model_info:
        return output, model_info
    return output


def _encode_image_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _extract_json_objects(text: str) -> List[dict]:
    objects: List[dict] = []
    depth = 0
    start = None
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                blob = text[start : idx + 1]
                try:
                    obj = json.loads(blob)
                    if isinstance(obj, dict):
                        objects.append(obj)
                except Exception:
                    pass
                start = None
    return objects


def _candidate_answer_texts(text: str) -> List[str]:
    raw = text.strip()
    if not raw:
        return []

    candidates: List[str] = []
    seen: set[str] = set()

    def push(value: str) -> None:
        item = value.strip()
        if not item or item in seen:
            return
        seen.add(item)
        candidates.append(item)

    push(raw)

    # Thinking models often return <think>...</think> + final answer.
    for match in _THINK_END_RE.finditer(raw):
        push(raw[match.end() :])

    # Keep the final assistant segment when role prefixes are echoed in output.
    role_markers = [m.end() for m in re.finditer(r"(?im)^\s*assistant\s*[:：]?\s*", raw)]
    if role_markers:
        push(raw[role_markers[-1] :])

    # Some providers wrap final text in <answer>...</answer>.
    for match in re.finditer(r"(?is)<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", raw):
        push(match.group(1))

    # Normalize role prefix noise.
    normalized = list(candidates)
    for item in normalized:
        push(_ROLE_PREFIX_RE.sub("", item).strip())

    return candidates


def _parse_openai_content(message_content) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        chunks = []
        for item in message_content:
            if isinstance(item, dict):
                if "text" in item:
                    chunks.append(str(item["text"]))
                elif item.get("type") == "text" and "content" in item:
                    chunks.append(str(item["content"]))
        return "".join(chunks)
    return str(message_content)


def _run_qwen_vl_siliconflow(
    image,
    prompt: str,
    *,
    model_id: str,
    max_new_tokens: int,
    api_base: Optional[str],
    api_key: Optional[str],
    api_path: str,
    timeout: float,
) -> str:
    import urllib.request

    api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
    api_base = api_base or os.getenv("SILICONFLOW_API_BASE")
    if not api_key:
        raise RuntimeError("Missing SILICONFLOW_API_KEY (or pass --api-key).")
    if not api_base:
        raise RuntimeError("Missing SILICONFLOW_API_BASE (or pass --api-base).")

    api_base = api_base.rstrip("/")
    api_path = api_path if api_path.startswith("/") else f"/{api_path}"
    url = f"{api_base}{api_path}"

    if image is None:
        content = [
            {"type": "text", "text": prompt},
        ]
    else:
        image_b64 = _encode_image_base64(image)
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": int(max_new_tokens),
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        resp_text = response.read().decode("utf-8")
    data = json.loads(resp_text)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Empty response: {resp_text[:500]}")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    return _parse_openai_content(content).strip()


def _parse_points_from_text(text: str, expected_length: int) -> Optional[List[Tuple[float, float]]]:
    for candidate in _candidate_answer_texts(text):
        points = None

        fenced = _JSON_FENCE_RE.findall(candidate)
        for block in reversed(fenced):
            try:
                data = json.loads(block)
                if isinstance(data, dict):
                    data = data.get("points")
                if isinstance(data, list):
                    points = data
                    break
            except Exception:
                continue

        if points is None:
            objects = _extract_json_objects(candidate)
            for obj in reversed(objects):
                if isinstance(obj, dict) and isinstance(obj.get("points"), list):
                    points = obj.get("points")
                    break

        if not isinstance(points, list):
            continue

        parsed: List[Tuple[float, float]] = []
        ok = True
        for item in points[:expected_length]:
            if isinstance(item, dict):
                raw_x = item.get("x")
                raw_y = item.get("y")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # Support both [x, y] and [idx, x, y] layouts.
                raw_x = item[-2]
                raw_y = item[-1]
            else:
                ok = False
                break
            try:
                x_val = float(raw_x)
                y_val = float(raw_y)
            except Exception:
                ok = False
                break
            parsed.append((x_val, y_val))

        if ok and len(parsed) == expected_length:
            return parsed

    return None


def _parse_missing_points_from_text(text: str) -> Optional[List[Tuple[int, float, float]]]:
    for candidate in _candidate_answer_texts(text):
        points = None

        fenced = _JSON_FENCE_RE.findall(candidate)
        for block in reversed(fenced):
            try:
                data = json.loads(block)
                if isinstance(data, dict):
                    data = data.get("points")
                if isinstance(data, list):
                    points = data
                    break
            except Exception:
                continue

        if points is None:
            objects = _extract_json_objects(candidate)
            for obj in reversed(objects):
                if isinstance(obj, dict) and isinstance(obj.get("points"), list):
                    points = obj.get("points")
                    break

        if not isinstance(points, list):
            continue

        parsed: List[Tuple[int, float, float]] = []
        ok = True
        for item in points:
            if isinstance(item, dict):
                raw_idx = item.get("idx", item.get("index"))
                raw_x = item.get("x")
                raw_y = item.get("y")
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                raw_idx = item[0]
                raw_x = item[1]
                raw_y = item[2]
            else:
                ok = False
                break
            try:
                idx_val = int(round(float(raw_idx)))
                x_val = float(raw_x)
                y_val = float(raw_y)
            except Exception:
                ok = False
                break
            parsed.append((idx_val, x_val, y_val))

        # Reject empty points: for this task, empty output means "no usable prediction".
        if ok and len(parsed) > 0:
            return parsed

    return None


def guess_traj_qwen_vl(
    traj_0: torch.Tensor,
    erase_mask: torch.Tensor,
    map_image_path: Optional[str],
    *,
    norm_stats_path: str = "Dataset/Indoor_norm_stats.json",
    map_extent: Sequence[float] = (0.0, 16.0, 0.0, 30.0),
    prompt: Optional[str] = None,
    model_id: str = "Qwen/Qwen3-VL-8B-Thinking",
    max_new_tokens: int = 4096,
    device: Optional[str] = None,
    force_download: bool = False,
    provider: str = "local",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    api_path: str = "/chat/completions",
    api_timeout: float = 60.0,
    render_out_path: Optional[str] = None,
    return_debug: bool = False,
):
    """
    Qwen3-VL prior for trajectory guessing.
    Input/Output are aligned with TaxiDataset.guessTraj: traj_0 (3,L), erase_mask (L,) -> loc_guess (2,L)
    """
    try:
        model_info = None
        mean, std = load_norm_stats(norm_stats_path)
        xy_denorm = denorm_xy(traj_0, mean, std)
        time_denorm = traj_0[2] * std[2] + mean[2]
        use_map = map_image_path is not None
        rendered = None
        rooms_mapping_path = None
        ports_mapping_path = None
        mapping_dir = None
        if map_image_path is not None:
            mapping_dir = Path(map_image_path).resolve().parent
        else:
            default_dir = Path("Dataset/Indoor Sensor")
            if default_dir.exists():
                mapping_dir = default_dir
        if mapping_dir is not None:
            candidate_rooms = mapping_dir / "floor1_rooms_mapping.csv"
            candidate_ports = mapping_dir / "floor1_ports_mapping.csv"
            if candidate_rooms.exists():
                rooms_mapping_path = str(candidate_rooms)
            if candidate_ports.exists():
                ports_mapping_path = str(candidate_ports)
        if use_map:
            rendered = render_traj_on_map(
                xy_denorm,
                erase_mask,
                map_image_path,
                map_extent=map_extent,
                render_out_path=render_out_path,
            )
        if prompt is None:
            prompt = build_qwen_prompt(
                xy_denorm,
                erase_mask,
                time_denorm=time_denorm,
                map_extent=map_extent,
                use_map=use_map,
                rooms_mapping_path=rooms_mapping_path,
                ports_mapping_path=ports_mapping_path,
            )
        if provider == "local":
            output_text = _run_qwen_vl(
                rendered,
                prompt,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                device=device,
                force_download=force_download,
                return_model_info=return_debug,
            )
            if return_debug and isinstance(output_text, tuple):
                output_text, model_info = output_text
        elif provider == "siliconflow":
            output_text = _run_qwen_vl_siliconflow(
                rendered,
                prompt,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                api_base=api_base,
                api_key=api_key,
                api_path=api_path,
                timeout=api_timeout,
            )
        else:
            raise RuntimeError(f"Unknown provider: {provider}")
        expected_length = int(traj_0.shape[1])
        missing_mask = erase_mask > 0.1

        def _parse_loc_guess_from_text(text: str) -> Optional[torch.Tensor]:
            missing_points = _parse_missing_points_from_text(text)
            if missing_points is None:
                points = _parse_points_from_text(text, expected_length)
                if points is None:
                    return None
                points_tensor = torch.tensor(points, dtype=traj_0.dtype, device=traj_0.device).T  # (2,L)
                parsed_guess = normalize_xy(points_tensor, mean, std)
                parsed_guess[:, ~missing_mask] = traj_0[:2, ~missing_mask]
                return parsed_guess

            parsed_guess = traj_0[:2].clone()
            mean_x, mean_y = float(mean[0]), float(mean[1])
            std_x, std_y = float(std[0]), float(std[1])
            for idx_val, x_val, y_val in missing_points:
                if 0 <= idx_val < expected_length and bool(missing_mask[idx_val].item()):
                    parsed_guess[0, idx_val] = (x_val - mean_x) / std_x
                    parsed_guess[1, idx_val] = (y_val - mean_y) / std_y
            return parsed_guess

        loc_guess = _parse_loc_guess_from_text(output_text)

        if loc_guess is None:
            if return_debug:
                return None, {
                    "rendered": rendered,
                    "prompt": prompt,
                    "raw_text": output_text,
                    "model_info": model_info if provider == "local" else None,
                    "fallback": "parse_failed",
                }
            raise RuntimeError("Qwen output parse failed (missing points).")

        invalid_mask = erase_mask < 0
        if torch.any(invalid_mask):
            loc_guess[:, invalid_mask] = 0.0
        loc_guess = torch.nan_to_num(loc_guess, nan=0.0)

        if return_debug:
            return loc_guess, {
                "rendered": rendered,
                "prompt": prompt,
                "raw_text": output_text,
                "model_info": model_info if provider == "local" else None,
                "fallback": None,
            }
        return loc_guess
    except Exception as exc:
        if return_debug:
            return None, {"error": str(exc), "fallback": "exception"}
        raise
