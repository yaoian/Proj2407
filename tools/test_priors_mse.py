from __future__ import annotations

import argparse
import csv
import json
import random
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Priors import guess_traj_time_interp, guess_traj_qwen_vl
from Priors.qwen_vl import load_norm_stats

WINDOW_LEN = 25
MISSING_START = 10
MISSING_END = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare priors MSE on missing points (CSV by user).")
    parser.add_argument("--csv", type=str, required=True, help="Path to merged_user_trajectories.csv")
    parser.add_argument("--map", type=str, required=False, help="Path to indoor map image")
    parser.add_argument("--norm-stats", type=str, default="Dataset/Indoor_norm_stats.json")
    parser.add_argument("--map-extent", type=str, default="0,16,0,30", help="x_min,x_max,y_min,y_max")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-8B-Thinking")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--render-dir", type=str, default="")
    parser.add_argument("--no-map", action="store_true", help="Do not send map image to Qwen (text-only).")
    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument("--sample-user", action="store_true", help="Randomly sample a single user_id to evaluate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-errors", action="store_true", help="Print Qwen errors / raw outputs when available.")
    parser.add_argument("--print-device-map", action="store_true", help="Print HF device_map summary once.")
    parser.add_argument("--save-raw", type=str, default="", help="Save Qwen raw outputs to directory.")
    parser.add_argument("--save-qwen-traj", type=str, default="", help="Save Qwen 25-point trajectories per window.")
    parser.add_argument("--force-download", action="store_true", help="Force redownload model files.")
    parser.add_argument("--provider", type=str, default="local", choices=["local", "siliconflow"])
    parser.add_argument("--api-base", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--api-path", type=str, default="/chat/completions")
    parser.add_argument("--api-timeout", type=float, default=60.0)
    return parser.parse_args()


def parse_extent(text: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("map-extent must be x_min,x_max,y_min,y_max")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def iter_users(csv_path: Path) -> Iterable[Tuple[int, List[Tuple[float, float, float]]]]:
    required = {"user_id", "x", "y", "time"}
    cur_user: Optional[int] = None
    cur_points: List[Tuple[float, float, float]] = []
    rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if not required.issubset(set(reader.fieldnames or [])):
            raise SystemExit(f"CSV header must contain {sorted(required)}, got {reader.fieldnames}")
        for row in reader:
            rows += 1
            uid = int(row["user_id"])
            x_val = float(row["x"])
            y_val = float(row["y"])
            t_val = float(row["time"])
            if cur_user is None:
                cur_user = uid
            if uid < cur_user:
                raise SystemExit(
                    "CSV must be grouped/sorted by user_id (non-decreasing). "
                    f"Found uid={uid} after uid={cur_user} at row={rows}."
                )
            if uid != cur_user:
                yield cur_user, cur_points
                cur_user = uid
                cur_points = []
            cur_points.append((t_val, x_val, y_val))
    if cur_user is not None:
        yield cur_user, cur_points


def sample_user_id(csv_path: Path, seed: int) -> Optional[int]:
    rng = random.Random(int(seed))
    chosen: Optional[int] = None
    count = 0
    for user_id, _points in iter_users(csv_path):
        count += 1
        if rng.randint(1, count) == 1:
            chosen = user_id
    return chosen


def build_traj(points: Sequence[Tuple[float, float, float]], mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    length = len(points)
    traj = torch.empty((3, length), dtype=torch.float32)
    mean_x, mean_y, mean_t = float(mean[0]), float(mean[1]), float(mean[2])
    std_x, std_y, std_t = float(std[0]), float(std[1]), float(std[2])
    for idx, (t_val, x_val, y_val) in enumerate(points):
        traj[0, idx] = (x_val - mean_x) / std_x
        traj[1, idx] = (y_val - mean_y) / std_y
        traj[2, idx] = (t_val - mean_t) / std_t
    return traj


def build_fixed_gap_mask(length: int) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.float32)
    mask[MISSING_START:MISSING_END] = 1.0
    return mask


def mse_on_missing(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    missing_mask = mask > 0.1
    if not torch.any(missing_mask):
        return None
    pred_m = pred[:, missing_mask]
    gt_m = gt[:, missing_mask]
    diff = pred_m - gt_m
    return float((diff * diff).mean().item())


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


def _extract_points_list_strict(raw_text: str) -> Optional[list]:
    text = raw_text.strip()
    if not text:
        return None

    # Prefer JSON code blocks, then plain JSON objects. Do not parse free-form numbers from CoT text.
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    for block in reversed(fenced):
        try:
            data = json.loads(block)
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("points"), list):
            return data["points"]

    for obj in reversed(_extract_json_objects(text)):
        if isinstance(obj.get("points"), list):
            return obj["points"]
    return None


def _build_final_answer_json(raw_text: str, mask: torch.Tensor) -> str:
    points = _extract_points_list_strict(raw_text)
    if not isinstance(points, list):
        return json.dumps({"points": []}, ensure_ascii=False, indent=2)

    expected_len = int(mask.shape[0])
    missing_idx = [int(i) for i in torch.argwhere(mask > 0.1).flatten().tolist()]

    # Case 1: missing-only output -> [idx, x, y] or {"idx"/"index","x","y"}
    missing_out: List[List[float]] = []
    ok_missing = True
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
            ok_missing = False
            break
        try:
            idx_val = int(round(float(raw_idx)))
            x_val = float(raw_x)
            y_val = float(raw_y)
        except Exception:
            ok_missing = False
            break
        missing_out.append([idx_val, x_val, y_val])
    if ok_missing and missing_out:
        return json.dumps({"points": missing_out}, ensure_ascii=False, indent=2)

    # Case 2: full trajectory output -> [x, y] or {"x","y"} with length==window_len
    full_xy: List[Tuple[float, float]] = []
    ok_full = True
    for item in points:
        if isinstance(item, dict):
            raw_x = item.get("x")
            raw_y = item.get("y")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            raw_x = item[0]
            raw_y = item[1]
        else:
            ok_full = False
            break
        try:
            x_val = float(raw_x)
            y_val = float(raw_y)
        except Exception:
            ok_full = False
            break
        full_xy.append((x_val, y_val))
    if ok_full and len(full_xy) == expected_len:
        converted = [[idx, float(full_xy[idx][0]), float(full_xy[idx][1])] for idx in missing_idx]
        return json.dumps({"points": converted}, ensure_ascii=False, indent=2)

    return json.dumps({"points": []}, ensure_ascii=False, indent=2)


def _compose_prompt_and_answer(prompt: str, answer_json: str) -> str:
    return "\n".join(
        [
            "user",
            str(prompt).strip(),
            "",
            "assistant",
            "```json",
            answer_json.strip(),
            "```",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    map_extent = parse_extent(args.map_extent)
    if not args.no_map and not args.map:
        print("Missing --map (or set --no-map for text-only testing).")
        return 1
    if args.provider == "siliconflow" and args.model_id == "Qwen/Qwen3-VL-4B-Instruct":
        args.model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    mean, std = load_norm_stats(args.norm_stats)
    map_path = None if args.no_map else args.map
    render_dir = "" if args.no_map else args.render_dir.strip()
    if render_dir:
        os.makedirs(render_dir, exist_ok=True)
    raw_dir = args.save_raw.strip()
    if raw_dir:
        os.makedirs(raw_dir, exist_ok=True)
    traj_dir = args.save_qwen_traj.strip()
    if traj_dir:
        os.makedirs(traj_dir, exist_ok=True)

    if not args.no_qwen and args.provider == "local":
        try:
            import torchvision  # noqa: F401
        except Exception as exc:
            print(f"warn: torchvision not available, Qwen may fail: {exc}")

    mask = build_fixed_gap_mask(WINDOW_LEN)
    force_download_once = bool(args.force_download)

    target_user = None
    if args.sample_user:
        target_user = sample_user_id(csv_path, args.seed)
        if target_user is None:
            print("No users found in CSV.")
            return 1
        print(f"Sampled user_id: {target_user}")

    for user_id, points in iter_users(csv_path):
        if target_user is not None and user_id != target_user:
            continue
        if len(points) < WINDOW_LEN:
            continue

        traj = build_traj(points, mean, std)
        total_windows = traj.shape[1] - WINDOW_LEN + 1
        time_sum = 0.0
        time_count = 0
        qwen_sum = 0.0
        qwen_count = 0
        qwen_error = 0
        printed_device_map = False

        for start in range(total_windows):
            window = traj[:, start : start + WINDOW_LEN]
            mse_time = mse_on_missing(guess_traj_time_interp(window, mask), window[:2], mask)
            if mse_time is not None:
                time_sum += mse_time
                time_count += 1
            time_mse_str = f"{mse_time:.6f}" if mse_time is not None else "N/A"

            if not args.no_qwen:
                qwen_mse_str = "N/A"
                render_out = ""
                if render_dir:
                    render_out = os.path.join(render_dir, f"user_{user_id}_start_{start}.png")
                qwen_out = guess_traj_qwen_vl(
                    window,
                    mask,
                    map_path,
                    norm_stats_path=args.norm_stats,
                    map_extent=map_extent,
                    model_id=args.model_id,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                    force_download=force_download_once,
                    provider=args.provider,
                    api_base=args.api_base or None,
                    api_key=args.api_key or None,
                    api_path=args.api_path,
                    api_timeout=float(args.api_timeout),
                    render_out_path=render_out if render_out else None,
                    return_debug=True,
                )
                if force_download_once:
                    force_download_once = False
                if isinstance(qwen_out, tuple):
                    loc_guess_qwen, debug = qwen_out
                    had_error = False
                    if args.print_device_map and not printed_device_map and isinstance(debug, dict):
                        model_info = debug.get("model_info") if isinstance(debug, dict) else None
                        hf_device_map = model_info.get("hf_device_map") if isinstance(model_info, dict) else None
                        if isinstance(hf_device_map, dict) and hf_device_map:
                            counts = {}
                            for dev in hf_device_map.values():
                                key = str(dev)
                                counts[key] = counts.get(key, 0) + 1
                            summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
                            print(f"[device_map] {summary}")
                        else:
                            print("[device_map] unavailable (no hf_device_map on model)")
                        printed_device_map = True
                    if isinstance(debug, dict) and "error" in debug:
                        qwen_error += 1
                        had_error = True
                        if args.print_errors:
                            print(f"[user {user_id} start {start}] Qwen error: {debug['error']}")
                    elif isinstance(debug, dict) and "raw_text" in debug and args.print_errors:
                        raw_text = debug.get("raw_text", "")
                        raw_text = raw_text if len(raw_text) <= 400 else raw_text[:400] + "..."
                        print(f"[user {user_id} start {start}] Qwen raw_text: {raw_text}")
                    if isinstance(debug, dict) and "raw_text" in debug and raw_dir:
                        raw_text = debug.get("raw_text", "")
                        prompt_text = debug.get("prompt", "")
                        final_answer_json = _build_final_answer_json(raw_text, mask)
                        save_text = _compose_prompt_and_answer(prompt_text, final_answer_json)
                        raw_path = os.path.join(raw_dir, f"user_{user_id}_start_{start}.txt")
                        with open(raw_path, "w", encoding="utf-8") as file_handle:
                            file_handle.write(save_text)
                    if loc_guess_qwen is None and not had_error:
                        qwen_error += 1
                else:
                    loc_guess_qwen = qwen_out

                if loc_guess_qwen is not None:
                    mse_qwen = mse_on_missing(loc_guess_qwen, window[:2], mask)
                    if mse_qwen is not None:
                        qwen_sum += mse_qwen
                        qwen_count += 1
                        qwen_mse_str = f"{mse_qwen:.6f}"
                    if traj_dir:
                        mean_x, mean_y, mean_t = float(mean[0]), float(mean[1]), float(mean[2])
                        std_x, std_y, std_t = float(std[0]), float(std[1]), float(std[2])
                        denorm_x = loc_guess_qwen[0] * std_x + mean_x
                        denorm_y = loc_guess_qwen[1] * std_y + mean_y
                        denorm_t = window[2] * std_t + mean_t
                        traj_path = os.path.join(traj_dir, f"user_{user_id}_start_{start}.csv")
                        with open(traj_path, "w", encoding="utf-8", newline="") as file_handle:
                            writer = csv.writer(file_handle)
                            writer.writerow(["idx", "time", "x", "y"])
                            for i in range(WINDOW_LEN):
                                writer.writerow(
                                    [
                                        i,
                                        float(denorm_t[i].item()),
                                        float(denorm_x[i].item()),
                                        float(denorm_y[i].item()),
                                    ]
                                )
                print(
                    f"user {user_id} window {start + 1}/{total_windows} "
                    f"time_mse={time_mse_str} qwen_mse={qwen_mse_str}"
                )
            else:
                print(f"user {user_id} window {start + 1}/{total_windows} time_mse={time_mse_str}")

        if time_count == 0:
            continue

        time_mse = time_sum / time_count
        if args.no_qwen:
            print(f"user {user_id} windows={time_count} time_mse={time_mse:.6f}")
        else:
            if qwen_count > 0:
                qwen_mse = qwen_sum / qwen_count
                print(
                    f"user {user_id} windows={time_count} time_mse={time_mse:.6f} "
                    f"qwen_mse={qwen_mse:.6f} qwen_errors={qwen_error}"
                )
            else:
                print(
                    f"user {user_id} windows={time_count} time_mse={time_mse:.6f} "
                    f"qwen_mse=N/A qwen_errors={qwen_error}"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
