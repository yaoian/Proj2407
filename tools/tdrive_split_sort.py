"""
按 `Dataset/T-Drive_0.txt` 的行格式拆分并排序轨迹：

行格式：
    user_id,YYYY-mm-dd HH:MM:SS,lon,lat

处理逻辑：
1) 按 user_id 分组
2) 每个 user 内按时间排序
3) 时间间隔超过阈值（默认 2 小时）则切分为两条轨迹
4) 按“轨迹程度”排序（默认按点数 points；也可选 duration/distance）

输出：
- summary CSV：按 user 排序（用户得分为其轨迹 metric 之和），并在用户内部按轨迹 metric 排序
- 可选保存为本项目 TaxiDataset 兼容的 `.pth`（list[tuple]，tuple[0] 是 (3, N) 轨迹张量）
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class Point:
    abs_t: int
    lon: float
    lat: float
    ts: datetime


@dataclass
class Trajectory:
    user_id: int
    points: List[Point]

    def __len__(self) -> int:
        return len(self.points)

    @property
    def start_ts(self) -> datetime:
        return self.points[0].ts

    @property
    def end_ts(self) -> datetime:
        return self.points[-1].ts

    @property
    def duration_seconds(self) -> int:
        return max(0, self.points[-1].abs_t - self.points[0].abs_t)


def _abs_seconds(ts: datetime) -> int:
    sec_of_day = ts.hour * 3600 + ts.minute * 60 + ts.second
    return ts.toordinal() * 86400 + sec_of_day


def _parse_line(line: str) -> Optional[Tuple[int, datetime, float, float]]:
    line = line.strip()
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None
    try:
        user_id = int(parts[0])
        ts = datetime.strptime(parts[1], "%Y-%m-%d %H:%M:%S")
        lon = float(parts[2])
        lat = float(parts[3])
    except ValueError:
        return None
    return user_id, ts, lon, lat


def _haversine_meters(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371000.0
    lon1r, lat1r, lon2r, lat2r = map(radians, (lon1, lat1, lon2, lat2))
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = sin(dlat / 2) ** 2 + cos(lat1r) * cos(lat2r) * sin(dlon / 2) ** 2
    return 2 * r * asin(sqrt(a))


def _traj_distance_meters(points: List[Point]) -> float:
    if len(points) < 2:
        return 0.0
    dist = 0.0
    for i in range(1, len(points)):
        p0 = points[i - 1]
        p1 = points[i]
        dist += _haversine_meters(p0.lon, p0.lat, p1.lon, p1.lat)
    return dist


def _metric_value(traj: Trajectory, metric: str) -> float:
    if metric == "points":
        return float(len(traj))
    if metric == "duration":
        return float(traj.duration_seconds)
    if metric == "distance":
        return float(_traj_distance_meters(traj.points))
    raise ValueError(f"Unknown metric: {metric}")


def _split_by_gap(user_id: int, points: List[Point], gap_seconds: int, min_points: int) -> List[Trajectory]:
    if not points:
        return []
    points.sort(key=lambda p: p.abs_t)

    trajectories: List[Trajectory] = []
    cur: List[Point] = [points[0]]
    for p in points[1:]:
        if p.abs_t - cur[-1].abs_t > gap_seconds:
            if len(cur) >= min_points:
                trajectories.append(Trajectory(user_id=user_id, points=cur))
            cur = [p]
        else:
            cur.append(p)
    if len(cur) >= min_points:
        trajectories.append(Trajectory(user_id=user_id, points=cur))
    return trajectories


def _to_taxi_cache_entry(traj: Trajectory) -> Tuple[torch.Tensor, int, str, str]:
    # 生成 (3, N)：lon, lat, rel_time(s)，并保证 rel_time 严格递增（过滤掉重复/乱序点）
    lon: List[float] = []
    lat: List[float] = []
    rel_t: List[float] = []

    start_abs_t = traj.points[0].abs_t
    last_rel_t: Optional[float] = None
    for p in traj.points:
        t = float(p.abs_t - start_abs_t)
        if last_rel_t is not None and t <= last_rel_t:
            continue
        lon.append(float(p.lon))
        lat.append(float(p.lat))
        rel_t.append(t)
        last_rel_t = t

    x = torch.tensor([lon, lat, rel_t], dtype=torch.float32)
    return x, traj.user_id, traj.start_ts.strftime("%Y-%m-%d %H:%M:%S"), traj.end_ts.strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split and sort T-Drive trajectories by user id and time gap.")
    p.add_argument("--input", type=str, required=True, help="Input txt path, e.g. Dataset/T-Drive_0.txt")
    p.add_argument("--gap-hours", type=float, default=2.0, help="Split if time gap > this (hours). Default: 2")
    p.add_argument("--min-points", type=int, default=10, help="Drop trajectories shorter than this. Default: 10")
    p.add_argument(
        "--sort-metric",
        type=str,
        default="points",
        choices=["points", "duration", "distance"],
        help="How to rank trajectories/users. Default: points",
    )
    p.add_argument("--summary-csv", type=str, default="output/tdrive_summary.csv", help="Output summary csv path")
    p.add_argument(
        "--out-pth",
        type=str,
        default="",
        help="Optional: output .pth (list of tuples) compatible with TaxiDataset",
    )
    p.add_argument("--encoding", type=str, default="utf-8", help="Input file encoding. Default: utf-8")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    gap_seconds = int(float(args.gap_hours) * 3600)
    min_points = int(args.min_points)
    metric = str(args.sort_metric)

    by_user: DefaultDict[int, List[Point]] = DefaultDict(list)
    total_lines = 0
    valid_lines = 0

    with input_path.open("r", encoding=args.encoding, errors="ignore") as f:
        for line in f:
            total_lines += 1
            parsed = _parse_line(line)
            if parsed is None:
                continue
            user_id, ts, lon, lat = parsed
            by_user[user_id].append(Point(abs_t=_abs_seconds(ts), lon=lon, lat=lat, ts=ts))
            valid_lines += 1

    user_trajs: Dict[int, List[Trajectory]] = {}
    for user_id, points in by_user.items():
        user_trajs[user_id] = _split_by_gap(user_id, points, gap_seconds=gap_seconds, min_points=min_points)

    user_score = {
        user_id: sum(_metric_value(t, metric) for t in trajs) for user_id, trajs in user_trajs.items()
    }
    ordered_users = sorted(user_trajs.keys(), key=lambda uid: (user_score[uid], uid), reverse=True)

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "user_id",
                "traj_rank_in_user",
                "points",
                "duration_seconds",
                "distance_meters",
                "start_time",
                "end_time",
            ]
        )

        for user_id in ordered_users:
            trajs = sorted(user_trajs[user_id], key=lambda t: _metric_value(t, metric), reverse=True)
            for rank, t in enumerate(trajs, start=1):
                writer.writerow(
                    [
                        user_id,
                        rank,
                        len(t),
                        t.duration_seconds,
                        round(_traj_distance_meters(t.points), 3),
                        t.start_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        t.end_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    ]
                )

    if args.out_pth:
        entries: List[Tuple[torch.Tensor, int, str, str]] = []
        for user_id in ordered_users:
            trajs = sorted(user_trajs[user_id], key=lambda t: _metric_value(t, metric), reverse=True)
            for t in trajs:
                entries.append(_to_taxi_cache_entry(t))
        out_path = Path(args.out_pth)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(entries, str(out_path))

    print(f"input={input_path} total_lines={total_lines} valid_lines={valid_lines} users={len(by_user)}")
    print(f"gap_hours={args.gap_hours} min_points={min_points} metric={metric}")
    print(f"summary_csv={summary_path}")
    if args.out_pth:
        print(f"out_pth={args.out_pth}")


if __name__ == "__main__":
    main()

