"""
Rewrite an existing eval.py-style taxi test batch (.pth tuple of len=8) to match
the `indoor_step_5.py` point-set style:

- Only keep a local window of length (2*context + missing) as valid points.
- Within that window, the middle `missing` points are marked as erased/query.
- Everything outside the window is treated as padding (mask=-1) and zeroed out,
  so the model cannot "peek" at extra context.

Input tuple format (len=8):
0 loc_0:     (B, 2, L)
1 loc_T:     (B, 2, L)
2 loc_guess: (B, 2, L)
3 time:      (B, 1, L)
4 mask:      (B, 1, L)  # 1=erase, 0=observe, -1=padding
5 bool_mask: (B, 2, L)
6 query_len: (B,)
7 observe_len:(B,)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rewrite test batch to a mid-segment missing-5 setup.")
    p.add_argument("--in", dest="in_path", type=str, required=True, help="Input test batch .pth (len=8 tuple).")
    p.add_argument("--out", dest="out_path", type=str, required=True, help="Output test batch .pth.")
    p.add_argument("--context", type=int, default=10, help="Context points on each side (default: 10).")
    p.add_argument("--missing", type=int, default=5, help="Consecutive missing points (default: 5).")
    p.add_argument(
        "--center",
        type=str,
        default="middle",
        choices=("middle",),
        help="Where to place the window. Currently only supports fixed middle.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed used for loc_T noise (default: 0).")
    return p.parse_args()


def _interp_guess(
    loc_0: torch.Tensor,
    time: torch.Tensor,
    window_start: int,
    context: int,
    missing: int,
) -> torch.Tensor:
    """
    Build loc_guess by time interpolation for the missing segment (only within the window).
    loc_0: (B, 2, L), time: (B, 1, L)
    """
    B, _, L = loc_0.shape
    loc_guess = loc_0.clone()

    miss_start = window_start + context
    miss_end = miss_start + missing  # exclusive
    left_idx = miss_start - 1
    right_idx = miss_end
    eps = 1e-8

    t_left = time[:, 0, left_idx]
    t_right = time[:, 0, right_idx]
    denom = (t_right - t_left).abs()
    safe = denom > eps

    left = loc_0[:, :, left_idx]
    right = loc_0[:, :, right_idx]

    for k in range(missing):
        idx = miss_start + k
        tk = time[:, 0, idx]
        ratio = torch.zeros(B, dtype=loc_0.dtype, device=loc_0.device)
        ratio[safe] = (tk[safe] - t_left[safe]) / (t_right[safe] - t_left[safe])
        ratio = ratio.clamp(0.0, 1.0).unsqueeze(1)  # (B, 1)
        loc_guess[:, :, idx] = left * (1.0 - ratio) + right * ratio

    return loc_guess


def main() -> None:
    args = parse_args()

    torch.manual_seed(int(args.seed))

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    batch = torch.load(in_path.as_posix(), map_location="cpu")
    if not isinstance(batch, tuple) or len(batch) != 8:
        raise TypeError(f"Expected a len=8 tuple test batch, got: {type(batch)} len={getattr(batch, '__len__', None)}")

    loc_0, _, _, time, _, _, _, _ = batch
    if not (isinstance(loc_0, torch.Tensor) and isinstance(time, torch.Tensor)):
        raise TypeError("Unexpected batch contents: loc_0/time must be tensors")

    B, _, L = loc_0.shape
    context = int(args.context)
    missing = int(args.missing)
    if context <= 0 or missing <= 0:
        raise ValueError(f"context/missing must be > 0, got context={context} missing={missing}")

    window_len = 2 * context + missing
    if window_len + 2 > L:
        raise ValueError(f"Window too long for L={L}: window_len={window_len} (need room for boundaries)")

    if args.center == "middle":
        mid = L // 2
        window_start = mid - (context + missing // 2)
    else:
        raise ValueError(f"Unsupported center mode: {args.center}")

    window_start = max(0, min(int(window_start), L - window_len))
    miss_start = window_start + context
    miss_end = miss_start + missing
    if not (0 <= window_start < L and 0 <= miss_start < miss_end <= L and window_start + window_len <= L):
        raise AssertionError("Internal window indices out of range")
    if miss_end >= L:
        raise ValueError("Missing segment hits the right boundary; adjust parameters.")
    if miss_start <= 0:
        raise ValueError("Missing segment hits the left boundary; adjust parameters.")

    # Build mask: default padding (-1), within window observed=0, missing=1.
    mask = torch.full((B, 1, L), -1.0, dtype=torch.float32)
    mask[:, 0, window_start : window_start + window_len] = 0.0
    mask[:, 0, miss_start:miss_end] = 1.0

    valid_1d = mask[:, 0, :] >= 0
    erased_1d = mask[:, 0, :] > 0.1
    bool_mask = erased_1d.unsqueeze(1).repeat(1, 2, 1)

    # Zero out everything outside the window, so extra information is not leaked.
    loc_0_new = loc_0.clone()
    time_new = time.clone()
    invalid = ~valid_1d
    loc_0_new = loc_0_new.masked_fill(invalid.unsqueeze(1), 0.0)
    time_new = time_new.masked_fill(invalid.unsqueeze(1), 0.0)

    loc_guess = _interp_guess(loc_0_new, time_new, window_start, context, missing)
    loc_guess = loc_guess.masked_fill(invalid.unsqueeze(1), 0.0)

    loc_T = loc_0_new.clone()
    loc_T[bool_mask] = torch.randn_like(loc_T[bool_mask])
    loc_T = loc_T.masked_fill(invalid.unsqueeze(1), 0.0)

    query_len = torch.ones(B, dtype=torch.long) * int(missing)
    observe_len = torch.ones(B, dtype=torch.long) * int(2 * context)

    out_batch = (loc_0_new, loc_T, loc_guess, time_new, mask, bool_mask, query_len, observe_len)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_batch, out_path.as_posix())
    print(f"saved: {out_path.as_posix()}")
    print(
        f"B={B} L={L} window_start={window_start} window_len={window_len} "
        f"context={context} missing={missing} points_valid={int(valid_1d.sum().item())} points_erased={int(erased_1d.sum().item())}"
    )


if __name__ == "__main__":
    main()
