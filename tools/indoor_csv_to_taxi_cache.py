"""
把室内轨迹 CSV 转换成本项目 TaxiDataset 可直接读取的 `.pth` 缓存文件（Xian/Chengdu 同类“taxi cache”格式）。

输入：
- CSV 必须包含表头：user_id,x,y,time
  - x/y：二维位置（可为网格坐标或平面坐标），会作为轨迹的前两通道
  - time：时间戳（数值，建议单位为秒）；脚本会对每个 segment/window 转成相对时间（从 0 开始）

输出（与 `Dataset/Xian_nov_cache.pth` 兼容的“Xian cache 结构”）：
- 顶层是 list
- 每个元素是一个 tuple(len=2)，其中：
  - tuple[0] 为轨迹张量 traj: (3, N)
  - tuple[1] 为占位张量（空张量），不承载数据
- 3 通道依次为 (x, y, rel_time)
- 三个通道做全局标准化（z-score）：(x - mean) / std，使其近似 N(0,1)

使用示例：
  # 1) 先生成 test，再生成 cache（自动排除 test 中的轨迹窗口）
  .venv/bin/python tools/indoor_csv_to_taxi_cache.py \\
    --csv Dataset/merged_user_trajectories.csv \\
    --test-out Dataset/test_Indoor_B100_l512_E0.5.pth \\
    --out Dataset/Indoor_nov_cache.pth \\
    --gap-seconds 86400 \\
    --window-len 512 \\
    --window-stride 512 \\
    --test-batch 100 \\
    --erase-rate 0.5 \\
    --min-points 20

  # 2) 如只生成 cache（不做 test），不传 --test-out 即可
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class Point:
    t: float
    x: float
    y: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert indoor CSV to TaxiDataset cache (.pth).")
    p.add_argument("--csv", type=str, required=True, help="Input CSV path, containing columns user_id,x,y,time")
    p.add_argument(
        "--test-out",
        type=str,
        default="",
        help="Optional: output test batch tuple (.pth). If set, selected test windows will be excluded from --out cache.",
    )
    p.add_argument("--test-batch", type=int, default=100, help="Test batch size B (only used when --test-out is set).")
    p.add_argument("--erase-rate", type=float, default=0.5, help="Erase rate for test batch (only used when --test-out).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for selecting test windows and masks.")
    p.add_argument("--out", type=str, required=True, help="Output cache .pth path, e.g. Dataset/Indoor_nov_cache.pth")
    p.add_argument(
        "--gap-seconds",
        type=float,
        default=86400.0,
        help="Split segments when time gap > this (seconds). Set <=0 to disable splitting (one per user).",
    )
    p.add_argument("--min-points", type=int, default=20, help="Drop segments/windows shorter than this.")
    p.add_argument(
        "--window-len",
        type=int,
        default=512,
        help="If >0, cut each segment into fixed-length windows of this many points.",
    )
    p.add_argument(
        "--window-stride",
        type=int,
        default=512,
        help="Stride for windowing (points). Default: window-len (non-overlap).",
    )
    p.add_argument("--max-trajs", type=int, default=0, help="Limit number of trajectories saved (0 means no limit).")
    return p.parse_args()


def _split_by_gap(points: Sequence[Point], gap_seconds: float) -> List[List[Point]]:
    if not points:
        return []
    if gap_seconds <= 0:
        return [list(points)]
    segments: List[List[Point]] = []
    cur: List[Point] = []
    prev_t: Optional[float] = None
    for p in points:
        if prev_t is not None and p.t - prev_t > gap_seconds and cur:
            segments.append(cur)
            cur = []
        cur.append(p)
        prev_t = p.t
    if cur:
        segments.append(cur)
    return segments


def _to_traj(points: Sequence[Point]) -> torch.Tensor:
    t0 = float(points[0].t)
    xs = [float(p.x) for p in points]
    ys = [float(p.y) for p in points]
    ts = [float(p.t) - t0 for p in points]
    return torch.tensor([xs, ys, ts], dtype=torch.float32)


def _window(points: Sequence[Point], window_len: int, stride: int) -> Iterable[Sequence[Point]]:
    if window_len <= 0:
        yield points
        return
    if stride <= 0:
        stride = window_len
    n = len(points)
    for start in range(0, n - window_len + 1, stride):
        yield points[start : start + window_len]


def _accumulate_sums(x: torch.Tensor, s: torch.Tensor, ss: torch.Tensor) -> int:
    # x: (3, N)
    s += x.to(torch.float64).sum(dim=1)
    ss += (x.to(torch.float64) ** 2).sum(dim=1)
    return int(x.shape[1])


def _compute_mean_std_from_sums(s: torch.Tensor, ss: torch.Tensor, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if n <= 1:
        raise ValueError("Not enough points to compute mean/std.")
    mean = s / n
    var = (ss / n) - mean * mean
    var = torch.clamp(var, min=1e-12)
    std = torch.sqrt(var)
    return mean.to(torch.float32), std.to(torch.float32)


def _flush_user(
    points: List[Point],
    gap_seconds: float,
    min_points: int,
    window_len: int,
    window_stride: int,
    on_traj,
    max_trajs: int,
    start_traj_id: int,
) -> int:
    if not points:
        return start_traj_id

    # Ensure time is strictly increasing (drop duplicates / non-increasing).
    points_sorted = points  # CSV 已按 user_id,time 排序时可直接用
    filtered: List[Point] = []
    last_t: Optional[float] = None
    for p in points_sorted:
        if last_t is not None and p.t <= last_t:
            continue
        filtered.append(p)
        last_t = p.t

    traj_id = start_traj_id
    for seg in _split_by_gap(filtered, gap_seconds=gap_seconds):
        if len(seg) < max(2, min_points):
            continue
        for win in _window(seg, window_len=window_len, stride=window_stride):
            if len(win) < max(2, min_points):
                continue
            x = _to_traj(win)
            on_traj(traj_id, x)
            traj_id += 1
            if max_trajs and traj_id >= max_trajs:
                return traj_id
    return traj_id


def _reservoir_update(
    traj_id: int,
    x: torch.Tensor,
    reservoir_ids: List[int],
    reservoir_trajs: List[torch.Tensor],
    seen: int,
    k: int,
    rng: torch.Generator,
) -> None:
    if k <= 0:
        return
    if seen < k:
        reservoir_ids.append(traj_id)
        reservoir_trajs.append(x)
        return
    j = int(torch.randint(low=0, high=seen + 1, size=(1,), generator=rng).item())
    if j < k:
        reservoir_ids[j] = traj_id
        reservoir_trajs[j] = x


def _make_test_batch(
    trajs: List[torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
    erase_rate: float,
    seed: int,
) -> Tuple[torch.Tensor, ...]:
    # Format matches eval.py taxi branch: (len=8)
    # 0 loc_0: (B,2,L)
    # 1 loc_T: (B,2,L)
    # 2 loc_guess: (B,2,L)
    # 3 time: (B,1,L)
    # 4 mask: (B,1,L) float, 1=erase, 0=observe
    # 5 bool_mask: (B,2,L) bool
    # 6 query_len: (B,)
    # 7 observe_len: (B,)
    from Dataset.DatasetTaxi import TaxiDataset

    if not trajs:
        raise ValueError("Empty test trajectories list.")

    B = len(trajs)
    L = int(trajs[0].shape[1])
    for x in trajs:
        if int(x.shape[1]) != L:
            raise ValueError("All test trajectories must have the same length for a fixed-L test batch.")

    g = torch.Generator().manual_seed(int(seed))

    query_len = int(L * float(erase_rate))
    observe_len = L - query_len
    batch_query_len = torch.ones(B, dtype=torch.long) * query_len
    batch_observe_len = torch.ones(B, dtype=torch.long) * observe_len

    loc_0 = torch.zeros(B, 2, L, dtype=torch.float32)
    loc_guess = torch.zeros(B, 2, L, dtype=torch.float32)
    time = torch.zeros(B, 1, L, dtype=torch.float32)
    mask = torch.zeros(B, 1, L, dtype=torch.float32)

    for i, raw in enumerate(trajs):
        x = (raw - mean[:, None]) / std[:, None]
        loc_0[i] = x[:2]
        time[i] = x[2:].reshape(1, -1)

        # Build erase mask: 1=erase, 0=observe (keep first/last always observed)
        n_remain = L - int(L * float(erase_rate))
        if n_remain < 2:
            raise ValueError(f"Invalid erase_rate={erase_rate}: remaining points < 2")
        mid = torch.randperm(L - 2, generator=g)[: max(0, n_remain - 2)] + 1
        mid = torch.sort(mid)[0]
        remain_indices = torch.cat([torch.tensor([0]), mid, torch.tensor([L - 1])], dim=0)
        erase = torch.ones(L, dtype=torch.float32)
        erase[remain_indices] = 0.0
        mask[i, 0] = erase

        # loc_guess uses the same logic as training
        loc_guess[i] = TaxiDataset.guessTraj(x, erase)

    bool_mask = (mask > 0.1).repeat(1, 2, 1)
    loc_T = loc_0.clone()
    # use the same generator for reproducibility
    noise = torch.randn(loc_T.shape, dtype=loc_T.dtype, generator=g)
    loc_T[bool_mask] = noise[bool_mask]

    return (loc_0, loc_T, loc_guess, time, mask, bool_mask, batch_query_len, batch_observe_len)


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    test_out = Path(args.test_out) if str(args.test_out).strip() else None
    test_batch = int(args.test_batch)
    erase_rate = float(args.erase_rate)
    seed = int(args.seed)
    gap_seconds = float(args.gap_seconds)
    min_points = int(args.min_points)
    window_len = int(args.window_len)
    window_stride = int(args.window_stride) if int(args.window_stride) > 0 else (window_len if window_len > 0 else 0)
    max_trajs = int(args.max_trajs)

    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    required = {"user_id", "x", "y", "time"}

    def iter_users() -> Iterable[Tuple[int, List[Point], int]]:
        cur_user: Optional[int] = None
        cur_points: List[Point] = []
        rows = 0
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not required.issubset(set(reader.fieldnames or [])):
                raise SystemExit(f"CSV header must contain {sorted(required)}, got {reader.fieldnames}")
            for row in reader:
                rows += 1
                uid = int(row["user_id"])
                x = float(row["x"])
                y = float(row["y"])
                t = float(row["time"])
                if cur_user is None:
                    cur_user = uid
                if uid < cur_user:
                    raise SystemExit(
                        "CSV must be grouped/sorted by user_id (non-decreasing). "
                        f"Found uid={uid} after uid={cur_user} at row={rows}."
                    )
                if uid != cur_user:
                    yield cur_user, cur_points, rows
                    cur_user = uid
                    cur_points = []
                cur_points.append(Point(t=t, x=x, y=y))
        if cur_user is not None:
            yield cur_user, cur_points, rows

    # Pass 1: select test windows (reservoir) and accumulate sums.
    rng = torch.Generator().manual_seed(seed)
    total_s = torch.zeros(3, dtype=torch.float64)
    total_ss = torch.zeros(3, dtype=torch.float64)
    total_points = 0

    test_ids_reservoir: List[int] = []
    test_trajs_reservoir: List[torch.Tensor] = []
    seen = 0

    traj_id = 0
    last_uid: Optional[int] = None
    last_rows = 0

    def on_traj_pass1(tid: int, x: torch.Tensor) -> None:
        nonlocal total_points, seen
        _accumulate_sums(x, s=total_s, ss=total_ss)
        total_points += int(x.shape[1])
        _reservoir_update(
            tid,
            x,
            reservoir_ids=test_ids_reservoir,
            reservoir_trajs=test_trajs_reservoir,
            seen=seen,
            k=(test_batch if test_out else 0),
            rng=rng,
        )
        seen += 1

    for uid, points, rows in iter_users():
        if last_uid is not None and uid < last_uid:
            raise SystemExit("Internal error: uid order violated.")
        last_uid = uid
        last_rows = rows
        traj_id = _flush_user(
            points=points,
            gap_seconds=gap_seconds,
            min_points=min_points,
            window_len=window_len,
            window_stride=window_stride,
            on_traj=on_traj_pass1,
            max_trajs=max_trajs,
            start_traj_id=traj_id,
        )
        if max_trajs and traj_id >= max_trajs:
            break

    if seen <= 0:
        raise SystemExit("No trajectories/windows generated. Try lowering --min-points or adjusting --gap-seconds.")

    test_ids = set(test_ids_reservoir) if test_out else set()

    # Compute mean/std from training set only (exclude test windows).
    train_s = total_s.clone()
    train_ss = total_ss.clone()
    train_points = total_points
    if test_out:
        test_s = torch.zeros(3, dtype=torch.float64)
        test_ss = torch.zeros(3, dtype=torch.float64)
        test_points = 0
        for x in test_trajs_reservoir:
            test_points += _accumulate_sums(x, s=test_s, ss=test_ss)
        train_s -= test_s
        train_ss -= test_ss
        train_points -= test_points
        if len(test_trajs_reservoir) < test_batch:
            raise SystemExit(
                f"Not enough windows for test batch: want {test_batch}, got {len(test_trajs_reservoir)}. "
                "Consider reducing --test-batch or adjusting --gap-seconds / --min-points."
            )
    mean, std = _compute_mean_std_from_sums(s=train_s, ss=train_ss, n=train_points)

    # Build test batch (normalized with training mean/std) and save first.
    if test_out:
        # keep reservoir order stable for reproducibility across runs with same seed
        test_tuple = _make_test_batch(
            test_trajs_reservoir, mean=mean, std=std, erase_rate=erase_rate, seed=seed
        )
        test_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(test_tuple, str(test_out))
        print(
            f"saved_test: {test_out} "
            f"(B={len(test_trajs_reservoir)} L={int(test_trajs_reservoir[0].shape[1])} erase_rate={erase_rate})"
        )

    # Pass 2: write cache excluding test windows.
    pad = torch.empty(0, dtype=torch.float32)
    cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
    traj_id2 = 0

    def on_traj_pass2(tid: int, x: torch.Tensor) -> None:
        if tid in test_ids:
            return
        xn = (x - mean[:, None]) / std[:, None]
        cache.append((xn, pad))

    for uid, points, rows in iter_users():
        traj_id2 = _flush_user(
            points=points,
            gap_seconds=gap_seconds,
            min_points=min_points,
            window_len=window_len,
            window_stride=window_stride,
            on_traj=on_traj_pass2,
            max_trajs=max_trajs,
            start_traj_id=traj_id2,
        )
        if max_trajs and traj_id2 >= max_trajs:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, str(out_path))

    print(f"csv={csv_path}")
    print(f"rows_read={last_rows}")
    print(f"gap_seconds={gap_seconds} min_points={min_points} window_len={window_len} window_stride={window_stride}")
    print(f"windows_total={seen} windows_test={len(test_ids)} windows_cache={len(cache)}")
    print(f"train_points={train_points} mean={mean.tolist()} std={std.tolist()}")
    print(f"saved_cache: {out_path}")


if __name__ == "__main__":
    main()
