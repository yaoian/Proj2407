"""
把 T-Drive 原始轨迹转换成本项目 TaxiDataset 可直接读取的 `.pth` 缓存文件。

输出文件格式（与 `Dataset/Xian_nov_cache.pth` 一致的“列表式缓存”）：
- 顶层是 list
- 每个元素是一个 tuple，tuple[0] 是轨迹张量 `traj: (3, N)`
- 3 个通道依次为 (lon, lat, time)，其中 time 为“该段轨迹内相对时间（秒）”
- 三个通道会做全局标准化：减均值/除标准差，使其近似 N(0,1)

常见 T-Drive 原始行格式（微软 T-Drive 2008 数据集）类似：
`taxi_id,YYYY-mm-dd HH:MM:SS,lon,lat`

示例：
python tools/tdrive_preprocess.py --input /path/to/T-Drive --out Dataset/TDrive_nov_cache.pth
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class Segment:
    lon: List[float]
    lat: List[float]
    t: List[float]  # seconds since segment start

    def __len__(self) -> int:
        return len(self.t)


def _iter_files(root: str, include_ext: Sequence[str]) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if include_ext and not any(name.endswith(ext) for ext in include_ext):
                continue
            yield os.path.join(dirpath, name)


def _parse_tdrive_line(line: str) -> Optional[Tuple[datetime, float, float]]:
    line = line.strip()
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None

    ts_str = parts[1]
    try:
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        lon = float(parts[2])
        lat = float(parts[3])
    except (ValueError, IndexError):
        return None
    return ts, lon, lat


def _abs_seconds(ts: datetime) -> int:
    sec_of_day = ts.hour * 3600 + ts.minute * 60 + ts.second
    # 使用 toordinal 避免时区/epoch 依赖：保证跨天单调递增
    return ts.toordinal() * 86400 + sec_of_day


def _segment_points(
    points: List[Tuple[int, float, float]], gap_seconds: int, min_points: int
) -> List[Segment]:
    if not points:
        return []
    points.sort(key=lambda x: x[0])

    segments: List[Segment] = []
    lon: List[float] = []
    lat: List[float] = []
    t: List[float] = []

    seg_start_abs_t: Optional[int] = None
    last_abs_t: Optional[int] = None

    def flush() -> None:
        nonlocal lon, lat, t
        if len(t) >= min_points:
            segments.append(Segment(lon=lon, lat=lat, t=t))
        lon, lat, t = [], [], []

    for abs_t, xlon, xlat in points:
        if seg_start_abs_t is None:
            seg_start_abs_t = abs_t
            last_abs_t = abs_t
        else:
            assert last_abs_t is not None
            if abs_t - last_abs_t > gap_seconds:
                flush()
                seg_start_abs_t = abs_t
        assert seg_start_abs_t is not None
        rel_t = float(abs_t - seg_start_abs_t)

        # 去重/保持 time 严格递增，避免插值出现 0 除
        if t and rel_t <= t[-1]:
            continue

        lon.append(float(xlon))
        lat.append(float(xlat))
        t.append(rel_t)
        last_abs_t = abs_t

    flush()
    return segments


def _compute_mean_std(segments: Sequence[Segment]) -> Tuple[torch.Tensor, torch.Tensor]:
    # 用 sum / sumsq 计算均值方差，保持实现简单；必要时用 float64 降低数值误差
    s = torch.zeros(3, dtype=torch.float64)
    ss = torch.zeros(3, dtype=torch.float64)
    n = 0
    for seg in segments:
        n_seg = len(seg)
        if n_seg <= 0:
            continue
        x = torch.tensor([seg.lon, seg.lat, seg.t], dtype=torch.float64)  # (3, N)
        s += x.sum(dim=1)
        ss += (x * x).sum(dim=1)
        n += n_seg

    if n <= 1:
        raise ValueError("Not enough points to compute mean/std. Check your input and filters.")

    mean = s / n
    var = (ss / n) - mean * mean
    var = torch.clamp(var, min=1e-12)
    std = torch.sqrt(var)
    return mean.to(torch.float32), std.to(torch.float32)


def _to_cache(
    segments: Sequence[Segment], mean: torch.Tensor, std: torch.Tensor, max_trajs: int
) -> List[Tuple[torch.Tensor]]:
    cache: List[Tuple[torch.Tensor]] = []
    for seg in segments[:max_trajs]:
        x = torch.tensor([seg.lon, seg.lat, seg.t], dtype=torch.float32)  # (3, N)
        x = (x - mean[:, None]) / std[:, None]
        cache.append((x,))
    return cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert raw T-Drive data to this project's taxi cache format.")
    p.add_argument("--input", type=str, required=True, help="T-Drive root folder containing raw trajectory files.")
    p.add_argument("--out", type=str, required=True, help="Output .pth path, e.g. Dataset/TDrive_nov_cache.pth")
    p.add_argument("--gap-minutes", type=int, default=30, help="Split trajectory when time gap exceeds this.")
    p.add_argument("--min-points", type=int, default=10, help="Drop segments shorter than this.")
    p.add_argument(
        "--include-ext",
        type=str,
        default=".txt",
        help="Comma-separated file extensions to include; empty means include all files.",
    )
    p.add_argument("--max-files", type=int, default=0, help="Limit scanned files (0 means no limit).")
    p.add_argument("--max-trajs", type=int, default=200000, help="Limit output trajectories to avoid OOM.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    include_ext = [x.strip() for x in str(args.include_ext).split(",") if x.strip()]
    gap_seconds = int(args.gap_minutes) * 60

    raw_segments: List[Segment] = []
    scanned_files = 0

    for path in _iter_files(args.input, include_ext):
        scanned_files += 1
        if args.max_files and scanned_files > int(args.max_files):
            break

        points: List[Tuple[int, float, float]] = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parsed = _parse_tdrive_line(line)
                    if parsed is None:
                        continue
                    ts, lon, lat = parsed
                    points.append((_abs_seconds(ts), lon, lat))
        except OSError:
            continue

        raw_segments.extend(_segment_points(points, gap_seconds=gap_seconds, min_points=int(args.min_points)))

    if not raw_segments:
        raise SystemExit("No valid segments found. Check --input/--include-ext/--min-points.")

    mean, std = _compute_mean_std(raw_segments)
    cache = _to_cache(raw_segments, mean=mean, std=std, max_trajs=int(args.max_trajs))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(cache, args.out)

    print(f"scanned_files={scanned_files}")
    print(f"segments_total={len(raw_segments)} segments_saved={len(cache)}")
    print(f"mean={mean.tolist()} std={std.tolist()}")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()

