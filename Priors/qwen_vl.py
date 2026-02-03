from __future__ import annotations

import base64
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .time_interp import guess_traj_time_interp

_MODEL_CACHE: Dict[Tuple[str, Optional[str]], Tuple[object, object]] = {}


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
    left, top, right, bottom = 0, 0, width - 1, height - 1
    if pixel_bbox is not None:
        left, top, right, bottom = pixel_bbox
        width = right - left + 1
        height = bottom - top + 1
    if x_max == x_min or y_max == y_min:
        raise ValueError("Invalid map_extent with zero range.")
    pixel_x = left + (y_coord - y_min) / (y_max - y_min) * (width - 1)
    pixel_y = top + (x_coord - x_min) / (x_max - x_min) * (height - 1)
    pixel_x = max(float(left), min(float(right), float(pixel_x)))
    pixel_y = max(float(top), min(float(bottom), float(pixel_y)))
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


def _draw_dashed_line(draw, start_point: Tuple[float, float], end_point: Tuple[float, float], *, fill, width: int) -> None:
    dash_length = 6.0
    gap_length = 4.0
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
    from PIL import Image, ImageDraw

    image = Image.open(map_image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
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
        pixel_points.append(_xy_to_pixel(x_coord, y_coord, map_extent, (width, height), pixel_bbox))

    observed_color = (31, 119, 180)
    missing_point_color = (214, 39, 40)

    for idx in range(len(pixel_points) - 1):
        start_point = pixel_points[idx]
        end_point = pixel_points[idx + 1]
        if start_point is None or end_point is None:
            continue
        if not bool(missing_mask[idx].item()) and not bool(missing_mask[idx + 1].item()):
            draw.line([start_point, end_point], fill=observed_color, width=line_width)

    for idx, point in enumerate(pixel_points):
        if point is None:
            continue
        if bool(missing_mask[idx].item()):
            radius = 5
            draw.ellipse(
                [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
                outline=missing_point_color,
                fill=missing_point_color,
            )

    if render_out_path:
        image.save(render_out_path)
    return image


def _format_traj_for_prompt(
    xy_denorm: torch.Tensor,
    time_denorm: Optional[torch.Tensor],
    erase_mask: torch.Tensor,
    *,
    digits: int = 3,
) -> str:
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
        else:
            x_text = f"{x_val:.{digits}f}"
            y_text = f"{y_val:.{digits}f}"
        if time_val is None:
            time_text = "NA"
        else:
            time_text = f"{time_val:.{digits}f}"
        lines.append(f"{idx},{time_text},{x_text},{y_text},{missing}")
    return "\n".join(lines)


def build_qwen_prompt(
    xy_denorm: torch.Tensor,
    erase_mask: torch.Tensor,
    *,
    time_denorm: Optional[torch.Tensor] = None,
    map_extent: Sequence[float] = (0.0, 16.0, 0.0, 30.0),
) -> str:
    x_min, x_max, y_min, y_max = map_extent
    traj_text = _format_traj_for_prompt(xy_denorm, time_denorm, erase_mask)
    return (
        "你是一名轨迹修复助手。已给出室内地图与轨迹示意图：\n"
        "- 蓝色实线：观测到的轨迹\n"
        "- 橙色虚线：缺失区段\n"
        "- 红点：缺失点\n"
        "坐标系：x 轴向下递增，y 轴向右递增。\n"
        f"地图范围：x in [{x_min}, {x_max}], y in [{y_min}, {y_max}]。\n"
        "请根据图像与下方轨迹表，补全所有缺失点，输出 JSON：\n"
        "{\"points\": [[x0, y0], [x1, y1], ...]}，长度必须等于轨迹长度。\n"
        "不要输出除 JSON 外的任何文字。\n\n"
        "轨迹表：\n"
        f"{traj_text}\n"
    )


def _run_qwen_vl(
    image,
    prompt: str,
    *,
    model_id: str,
    max_new_tokens: int,
    device: Optional[str],
    force_download: bool,
) -> str:
    from transformers import AutoModelForVision2Seq, AutoProcessor

    cache_key = (model_id, device)
    if cache_key in _MODEL_CACHE and not force_download:
        processor, model = _MODEL_CACHE[cache_key]
    else:
        processor = AutoProcessor.from_pretrained(model_id, force_download=force_download)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto" if device is None else None,
            force_download=force_download,
        )
        if device is not None:
            model = model.to(device)
        _MODEL_CACHE[cache_key] = (processor, model)

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

    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output.strip()


def _encode_image_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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

    image_b64 = _encode_image_base64(image)
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
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
    text = text.strip()
    json_text = None
    if text.startswith("{") and text.endswith("}"):
        json_text = text
    elif text.startswith("[") and text.endswith("]"):
        json_text = text
    else:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx >= 0 and end_idx > start_idx:
            json_text = text[start_idx : end_idx + 1]
    points = None
    if json_text is not None:
        try:
            data = json.loads(json_text)
            if isinstance(data, dict):
                data = data.get("points")
            if isinstance(data, list):
                points = data
        except Exception:
            points = None
    if points is None:
        number_list = re.findall(r"[-+]?(?:\\d*\\.\\d+|\\d+)(?:[eE][-+]?\\d+)?", text)
        if len(number_list) >= expected_length * 2:
            values = [float(item) for item in number_list[: expected_length * 2]]
            points = [[values[idx], values[idx + 1]] for idx in range(0, len(values), 2)]
    if not isinstance(points, list):
        return None
    parsed: List[Tuple[float, float]] = []
    for item in points[:expected_length]:
        if not (isinstance(item, list) or isinstance(item, tuple)) or len(item) < 2:
            return None
        try:
            x_val = float(item[0])
            y_val = float(item[1])
        except Exception:
            return None
        parsed.append((x_val, y_val))
    if len(parsed) != expected_length:
        return None
    return parsed


def guess_traj_qwen_vl(
    traj_0: torch.Tensor,
    erase_mask: torch.Tensor,
    map_image_path: str,
    *,
    norm_stats_path: str = "Dataset/Indoor_norm_stats.json",
    map_extent: Sequence[float] = (0.0, 16.0, 0.0, 30.0),
    prompt: Optional[str] = None,
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    max_new_tokens: int = 1024,
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
    baseline_guess = guess_traj_time_interp(traj_0, erase_mask)
    try:
        mean, std = load_norm_stats(norm_stats_path)
        xy_denorm = denorm_xy(traj_0, mean, std)
        time_denorm = traj_0[2] * std[2] + mean[2]
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
            )
        if provider == "local":
            output_text = _run_qwen_vl(
                rendered,
                prompt,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
                device=device,
                force_download=force_download,
            )
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
        points = _parse_points_from_text(output_text, expected_length)
        if points is None:
            if return_debug:
                return baseline_guess, {
                    "rendered": rendered,
                    "prompt": prompt,
                    "raw_text": output_text,
                    "fallback": "parse_failed",
                }
            return baseline_guess
        points_tensor = torch.tensor(points, dtype=traj_0.dtype, device=traj_0.device).T  # (2,L)
        loc_guess = normalize_xy(points_tensor, mean, std)
        missing_mask = erase_mask > 0.1
        loc_guess[:, ~missing_mask] = traj_0[:2, ~missing_mask]

        invalid_mask = erase_mask < 0
        if torch.any(invalid_mask):
            loc_guess[:, invalid_mask] = 0.0
        loc_guess = torch.nan_to_num(loc_guess, nan=0.0)

        if return_debug:
            return loc_guess, {"rendered": rendered, "prompt": prompt, "raw_text": output_text, "fallback": None}
        return loc_guess
    except Exception as exc:
        if return_debug:
            return baseline_guess, {"error": str(exc), "fallback": "exception"}
        return baseline_guess
