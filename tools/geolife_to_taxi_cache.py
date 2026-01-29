"""
把 GeoLife Trajectories 1.3 的 .plt 转为本项目 TaxiDataset 可读取的缓存 `.pth`。

输出格式：
- 顶层是 list
- 每个元素是 tuple，tuple[0] 为轨迹张量 `traj: (3, N)`，通道为 (lon, lat, rel_time_seconds)
- 轨迹会做全局标准化：减均值/除标准差，使其近似 N(0,1)

默认只保留“新增日期范围”（可通过 --start/--end 调整）。

示例：
python tools/geolife_to_taxi_cache.py \
  --data "Dataset/Geolife Trajectories 1.3/Data" \
  --out Dataset/Geolife_newperiod_cache.pth \
  --start "2011-11-01 00:00:00" \
  --end "2012-08-31 23:59:59" \
  --min-points 10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class RawPoint:
    abs_t: int
    lon: float
    lat: float


def _iter_user_dirs(data_dir: Path) -> Iterable[Path]:
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and p.name.isdigit():
            yield p


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _parse_geolife_ts(date_str: str, time_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _abs_seconds(ts: datetime) -> int:
    sec_of_day = ts.hour * 3600 + ts.minute * 60 + ts.second
    return ts.toordinal() * 86400 + sec_of_day


def _scan_plt_points_in_range(path: Path, start: datetime, end: Optional[datetime]) -> List[RawPoint]:
    points: List[RawPoint] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i < 6:
                    continue
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                except ValueError:
                    continue
                ts = _parse_geolife_ts(parts[5], parts[6])
                if ts is None:
                    continue
                if ts < start:
                    continue
                if end is not None and ts > end:
                    continue
                points.append(RawPoint(abs_t=_abs_seconds(ts), lon=lon, lat=lat))
    except OSError:
        return []
    points.sort(key=lambda p: p.abs_t)
    return points


def _split_by_gap(points: Sequence[RawPoint], gap_seconds: int, min_points: int) -> List[List[RawPoint]]:
    if not points:
        return []
    if gap_seconds <= 0:
        return [list(points)] if len(points) >= min_points else []

    segments: List[List[RawPoint]] = []
    cur: List[RawPoint] = [points[0]]
    for p in points[1:]:
        if p.abs_t - cur[-1].abs_t > gap_seconds:
            if len(cur) >= min_points:
                segments.append(cur)
            cur = [p]
        else:
            cur.append(p)
    if len(cur) >= min_points:
        segments.append(cur)
    return segments


def _segment_to_traj(seg: Sequence[RawPoint]) -> torch.Tensor:
    start_abs_t = seg[0].abs_t
    lon: List[float] = []
    lat: List[float] = []
    rel_t: List[float] = []
    last_t: Optional[float] = None
    for p in seg:
        t = float(p.abs_t - start_abs_t)
        if last_t is not None and t <= last_t:
            continue
        lon.append(float(p.lon))
        lat.append(float(p.lat))
        rel_t.append(t)
        last_t = t
    return torch.tensor([lon, lat, rel_t], dtype=torch.float32)


def _compute_mean_std(trajs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    s = torch.zeros(3, dtype=torch.float64)
    ss = torch.zeros(3, dtype=torch.float64)
    n = 0
    for x in trajs:
        if x.numel() == 0:
            continue
        s += x.to(torch.float64).sum(dim=1)
        ss += (x.to(torch.float64) ** 2).sum(dim=1)
        n += int(x.shape[1])
    if n <= 1:
        raise ValueError("Not enough points to compute mean/std.")
    mean = s / n
    var = (ss / n) - mean * mean
    var = torch.clamp(var, min=1e-12)
    std = torch.sqrt(var)
    return mean.to(torch.float32), std.to(torch.float32), n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert GeoLife .plt data to TaxiDataset cache (.pth).")
    p.add_argument("--data", type=str, required=True, help="GeoLife Data/ directory, e.g. .../Data")
    p.add_argument("--out", type=str, required=True, help="Output cache .pth path")
    p.add_argument("--start", type=str, default="2011-11-01 00:00:00", help="Inclusive start datetime")
    p.add_argument("--end", type=str, default="2012-08-31 23:59:59", help="Inclusive end datetime (empty means none)")
    p.add_argument("--min-points", type=int, default=10, help="Drop segments shorter than this")
    p.add_argument(
        "--gap-minutes",
        type=int,
        default=0,
        help="Optional: split a .plt into segments if time gap > this (0 disables, keep one per .plt).",
    )
    p.add_argument("--max-files", type=int, default=0, help="Limit scanned .plt files (0 means no limit)")
    p.add_argument("--max-trajs", type=int, default=0, help="Limit output trajectories (0 means no limit)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data)
    start = _parse_dt(args.start)
    end = _parse_dt(args.end) if str(args.end).strip() else None
    min_points = int(args.min_points)
    gap_seconds = int(args.gap_minutes) * 60
    max_files = int(args.max_files)
    max_trajs = int(args.max_trajs)

    trajs: List[torch.Tensor] = []
    scanned_files = 0
    matched_files = 0

    for user_dir in _iter_user_dirs(data_dir):
        traj_dir = user_dir / "Trajectory"
        if not traj_dir.is_dir():
            continue
        for plt_path in sorted(traj_dir.glob("*.plt")):
            scanned_files += 1
            if max_files and scanned_files > max_files:
                break
            points = _scan_plt_points_in_range(plt_path, start=start, end=end)
            if not points:
                continue
            matched_files += 1
            segments = _split_by_gap(points, gap_seconds=gap_seconds, min_points=min_points)
            for seg in segments:
                x = _segment_to_traj(seg)
                if x.shape[1] < min_points:
                    continue
                trajs.append(x)
                if max_trajs and len(trajs) >= max_trajs:
                    break
            if max_trajs and len(trajs) >= max_trajs:
                break
        if max_files and scanned_files > max_files:
            break
        if max_trajs and len(trajs) >= max_trajs:
            break

    if not trajs:
        raise SystemExit("No trajectories found after filtering. Check --start/--end/--min-points.")

    mean, std, total_points = _compute_mean_std(trajs)
    cache = [((x - mean[:, None]) / std[:, None],) for x in trajs]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, str(out_path))

    print(f"data_dir={data_dir}")
    print(f"range_start={start} range_end={end or ''}")
    print(f"gap_minutes={args.gap_minutes} min_points={min_points}")
    print(f"scanned_files={scanned_files} matched_files={matched_files}")
    print(f"trajs_saved={len(cache)} total_points={total_points}")
    print(f"mean={mean.tolist()} std={std.tolist()}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

