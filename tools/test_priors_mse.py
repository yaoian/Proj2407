from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Priors import guess_traj_time_interp, guess_traj_qwen_vl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare priors MSE on missing points.")
    parser.add_argument("--cache", type=str, required=True, help="Path to taxi cache .pth")
    parser.add_argument("--map", type=str, required=True, help="Path to indoor map image")
    parser.add_argument("--norm-stats", type=str, default="Dataset/Indoor_norm_stats.json")
    parser.add_argument("--num", type=int, default=8, help="Number of trajectories to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--erase-rate", type=float, default=0.5)
    parser.add_argument("--map-extent", type=str, default="0,16,0,30", help="x_min,x_max,y_min,y_max")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--render-dir", type=str, default="")
    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument("--print-errors", action="store_true", help="Print Qwen errors / raw outputs when available.")
    parser.add_argument("--save-raw", type=str, default="", help="Save Qwen raw outputs to directory.")
    parser.add_argument("--force-download", action="store_true", help="Force redownload model files.")
    parser.add_argument("--provider", type=str, default="local", choices=["local", "siliconflow"])
    parser.add_argument("--api-base", type=str, default="")
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--api-path", type=str, default="/chat/completions")
    parser.add_argument("--api-timeout", type=float, default=60.0)
    parser.add_argument("--print-last-points", type=int, default=0, help="Print last N missing points for Qwen and TimeInterp.")
    parser.add_argument("--print-all-missing", action="store_true", help="Print all missing points for Qwen and TimeInterp (last traj).")
    parser.add_argument("--print-fallback", action="store_true", help="Print if Qwen falls back to TimeInterp.")
    parser.add_argument("--test-len", type=int, default=0, help="Use only first N points (0 means full length).")
    parser.add_argument("--test-len-mode", type=str, default="head", choices=["head", "random"])
    return parser.parse_args()


def parse_extent(text: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("map-extent must be x_min,x_max,y_min,y_max")
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def load_trajs(cache_path: str) -> List[torch.Tensor]:
    dataset_part = torch.load(cache_path, map_location="cpu")
    trajs: List[torch.Tensor] = []
    for sample in dataset_part:
        if isinstance(sample, (list, tuple)) and len(sample) > 0:
            traj = sample[0]
        else:
            traj = sample
        if not isinstance(traj, torch.Tensor):
            continue
        trajs.append(traj.to(torch.float32))
    return trajs


def build_erase_mask(length: int, erase_rate: float, generator: torch.Generator) -> Optional[torch.Tensor]:
    if length < 2:
        return None
    n_remain = length - int(length * erase_rate)
    n_remain = max(2, min(length, n_remain))
    if length <= 2:
        remain_indices = torch.tensor([0, length - 1], dtype=torch.long)
    else:
        n_middle = max(0, n_remain - 2)
        perm = torch.randperm(length - 2, generator=generator)[:n_middle] + 1
        perm = torch.sort(perm)[0]
        remain_indices = torch.cat([torch.tensor([0]), perm, torch.tensor([length - 1])])
    mask = torch.ones(length, dtype=torch.float32)
    mask[remain_indices] = 0.0
    return mask


def mse_on_missing(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    missing_mask = mask > 0.1
    if not torch.any(missing_mask):
        return None
    pred_m = pred[:, missing_mask]
    gt_m = gt[:, missing_mask]
    diff = pred_m - gt_m
    return float((diff * diff).mean().item())


def main() -> int:
    args = parse_args()
    map_extent = parse_extent(args.map_extent)
    if args.provider == "siliconflow" and args.model_id == "Qwen/Qwen3-VL-4B-Instruct":
        args.model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    trajs = load_trajs(args.cache)
    if not trajs:
        print("No trajectories loaded.")
        return 1

    rng = random.Random(int(args.seed))
    indices = list(range(len(trajs)))
    if args.num > 0 and args.num < len(indices):
        indices = rng.sample(indices, args.num)

    torch_gen = torch.Generator().manual_seed(int(args.seed))
    render_dir = args.render_dir.strip()
    if render_dir:
        os.makedirs(render_dir, exist_ok=True)
    raw_dir = args.save_raw.strip()
    if raw_dir:
        os.makedirs(raw_dir, exist_ok=True)

    if not args.no_qwen and args.provider == "local":
        try:
            import torchvision  # noqa: F401
        except Exception as exc:
            print(f"warn: torchvision not available, Qwen may fail: {exc}")

    mse_time_total = 0.0
    mse_qwen_total = 0.0
    count = 0
    qwen_error = 0
    force_download_once = bool(args.force_download)
    last_pair = None

    for idx in indices:
        traj = trajs[idx]
        if args.test_len and int(args.test_len) > 0:
            target_len = int(args.test_len)
            if traj.shape[1] > target_len:
                if args.test_len_mode == "random":
                    start = rng.randint(0, int(traj.shape[1]) - target_len)
                else:
                    start = 0
                traj = traj[:, start : start + target_len]
        length = int(traj.shape[1])
        mask = build_erase_mask(length, float(args.erase_rate), torch_gen)
        if mask is None:
            continue
        mse_time = mse_on_missing(guess_traj_time_interp(traj, mask), traj[:2], mask)
        if mse_time is None:
            continue

        mse_time_total += mse_time
        count += 1

        last_pair = {
            "traj_idx": idx,
            "mask": mask,
            "gt": traj[:2],
            "time_guess": guess_traj_time_interp(traj, mask),
        }

        if not args.no_qwen:
            render_out = ""
            if render_dir:
                render_out = os.path.join(render_dir, f"traj_{idx}.png")
            qwen_out = guess_traj_qwen_vl(
                traj,
                mask,
                args.map,
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
                if isinstance(debug, dict) and "error" in debug:
                    qwen_error += 1
                    if args.print_errors:
                        print(f"[traj {idx}] Qwen error: {debug['error']}")
                    if args.print_fallback:
                        print(f"[traj {idx}] Qwen fallback: exception")
                elif isinstance(debug, dict) and "raw_text" in debug and args.print_errors:
                    raw_text = debug.get("raw_text", "")
                    raw_text = raw_text if len(raw_text) <= 400 else raw_text[:400] + "..."
                    print(f"[traj {idx}] Qwen raw_text: {raw_text}")
                if isinstance(debug, dict) and "raw_text" in debug and raw_dir:
                    raw_text = debug.get("raw_text", "")
                    raw_path = os.path.join(raw_dir, f"traj_{idx}.txt")
                    with open(raw_path, "w", encoding="utf-8") as f:
                        f.write(raw_text)
                if args.print_fallback and isinstance(debug, dict) and debug.get("fallback") == "parse_failed":
                    print(f"[traj {idx}] Qwen fallback: parse_failed")
                if loc_guess_qwen is None:
                    qwen_error += 1
                    if args.print_fallback:
                        print(f"[traj {idx}] Qwen output missing; skipped.")
            else:
                loc_guess_qwen = qwen_out
            if loc_guess_qwen is not None:
                mse_qwen = mse_on_missing(loc_guess_qwen, traj[:2], mask)
                if mse_qwen is not None:
                    mse_qwen_total += mse_qwen
            if last_pair is not None:
                last_pair["qwen_guess"] = loc_guess_qwen

    if count == 0:
        print("No valid trajectories with missing points.")
        return 1

    print(f"Samples: {count}")
    print(f"TimeInterp MSE (missing): {mse_time_total / count:.6f}")
    if not args.no_qwen:
        if qwen_error < count:
            print(f"QwenVL   MSE (missing): {mse_qwen_total / max(1, count - qwen_error):.6f}")
        else:
            print("QwenVL   MSE (missing): N/A (all failed)")
        if qwen_error:
            print(f"Qwen errors: {qwen_error}")
    if (args.print_last_points > 0 or args.print_all_missing) and last_pair is not None:
        mask = last_pair["mask"]
        missing_idx = (mask > 0.1).nonzero(as_tuple=False).view(-1).tolist()
        if not args.print_all_missing:
            n = int(args.print_last_points)
            missing_idx = missing_idx[-n:] if n > 0 else missing_idx
        print(f"Last traj idx: {last_pair['traj_idx']}")
        print("Missing idx:", missing_idx)
        time_guess = last_pair["time_guess"]
        gt = last_pair["gt"]
        qwen_guess = last_pair.get("qwen_guess")
        for i in missing_idx:
            gt_xy = (float(gt[0, i].item()), float(gt[1, i].item()))
            ti_xy = (float(time_guess[0, i].item()), float(time_guess[1, i].item()))
            if qwen_guess is not None:
                qw_xy = (float(qwen_guess[0, i].item()), float(qwen_guess[1, i].item()))
            else:
                qw_xy = None
            print(f"idx {i}: gt={gt_xy} time={ti_xy} qwen={qw_xy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
