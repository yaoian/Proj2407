"""
从 GeoLife Trajectories 1.3 中提取“新增日期范围”的轨迹信息（按时间过滤，而非对比 1.2/1.3 目录差集）。

背景：
- GeoLife 1.2 覆盖约 2007/04 - 2011/10
- GeoLife 1.3 覆盖约 2007/04 - 2012/08
因此，1.3 的“新增数据”可按时间范围过滤：start >= 2011-11-01（默认，可改）

GeoLife 常见结构：
<root>/Data/<user_id>/Trajectory/*.plt

输出：
- per-file CSV：每个 .plt 在给定时间范围内的点数/时间跨度/bbox（只统计落在范围内的点）
- per-user CSV：按 user_id 聚合上述指标（可选）

示例：
python tools/geolife_extract_new_period.py \
  --root "/path/to/Geolife Trajectories 1.3" \
  --start "2011-11-01 00:00:00" \
  --end   "2012-08-31 23:59:59" \
  --min-points 10 \
  --out-file output/geolife_v13_new_period_files.csv \
  --out-user output/geolife_v13_new_period_users.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _find_data_dir(root: Path) -> Path:
    if root.is_dir() and (root / "Data").is_dir():
        return root / "Data"
    if root.is_dir() and root.name == "Data":
        return root
    raise FileNotFoundError(f"Cannot find Data/ under: {root}")


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


@dataclass
class Stats:
    points: int = 0
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # min_lon, min_lat, max_lon, max_lat

    def add_point(self, lon: float, lat: float, ts: datetime) -> None:
        if self.start_ts is None or ts < self.start_ts:
            self.start_ts = ts
        if self.end_ts is None or ts > self.end_ts:
            self.end_ts = ts
        if self.bbox is None:
            self.bbox = (lon, lat, lon, lat)
        else:
            min_lon, min_lat, max_lon, max_lat = self.bbox
            self.bbox = (
                min(min_lon, lon),
                min(min_lat, lat),
                max(max_lon, lon),
                max(max_lat, lat),
            )
        self.points += 1

    def merge(self, other: "Stats") -> None:
        if other.points <= 0:
            return
        self.points += other.points
        if other.start_ts is not None and (self.start_ts is None or other.start_ts < self.start_ts):
            self.start_ts = other.start_ts
        if other.end_ts is not None and (self.end_ts is None or other.end_ts > self.end_ts):
            self.end_ts = other.end_ts
        if self.bbox is None:
            self.bbox = other.bbox
        elif other.bbox is not None:
            a = self.bbox
            b = other.bbox
            self.bbox = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _scan_plt_in_range(path: Path, start: datetime, end: Optional[datetime]) -> Stats:
    stats = Stats()
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
                stats.add_point(lon=lon, lat=lat, ts=ts)
    except OSError:
        return Stats()
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract GeoLife v1.3 new-period trajectory info by time filtering.")
    p.add_argument("--root", type=str, required=True, help="GeoLife root path (or Data/ path).")
    p.add_argument(
        "--start",
        type=str,
        default="2011-11-01 00:00:00",
        help="Inclusive start datetime, default: 2011-11-01 00:00:00",
    )
    p.add_argument(
        "--end",
        type=str,
        default="",
        help="Inclusive end datetime (empty means no upper bound), e.g. 2012-08-31 23:59:59",
    )
    p.add_argument("--min-points", type=int, default=10, help="Skip plt files with < min points in range.")
    p.add_argument("--out-file", type=str, required=True, help="Per-file CSV output path.")
    p.add_argument("--out-user", type=str, default="", help="Optional per-user CSV output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = _find_data_dir(Path(args.root))
    start = _parse_dt(args.start)
    end = _parse_dt(args.end) if str(args.end).strip() else None
    min_points = int(args.min_points)

    file_rows: List[Tuple[str, str, int, str, str, Optional[Tuple[float, float, float, float]]]] = []
    user_stats: Dict[str, Stats] = {}
    matched_files = 0

    for user_dir in _iter_user_dirs(data_dir):
        user_id = user_dir.name
        traj_dir = user_dir / "Trajectory"
        if not traj_dir.is_dir():
            continue
        for plt_path in sorted(traj_dir.glob("*.plt")):
            s = _scan_plt_in_range(plt_path, start=start, end=end)
            if s.points < min_points:
                continue
            matched_files += 1
            user_stats.setdefault(user_id, Stats()).merge(s)
            file_rows.append(
                (
                    user_id,
                    str(plt_path.relative_to(data_dir)),
                    s.points,
                    s.start_ts.strftime("%Y-%m-%d %H:%M:%S") if s.start_ts else "",
                    s.end_ts.strftime("%Y-%m-%d %H:%M:%S") if s.end_ts else "",
                    s.bbox,
                )
            )

    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "user_id",
                "plt_relpath",
                "points_in_range",
                "start_time",
                "end_time",
                "bbox_min_lon",
                "bbox_min_lat",
                "bbox_max_lon",
                "bbox_max_lat",
            ]
        )
        for user_id, relpath, pts, st, et, bbox in file_rows:
            if bbox is None:
                w.writerow([user_id, relpath, pts, st, et, "", "", "", ""])
            else:
                w.writerow(
                    [
                        user_id,
                        relpath,
                        pts,
                        st,
                        et,
                        f"{bbox[0]:.6f}",
                        f"{bbox[1]:.6f}",
                        f"{bbox[2]:.6f}",
                        f"{bbox[3]:.6f}",
                    ]
                )

    if args.out_user:
        out_user = Path(args.out_user)
        out_user.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for uid, s in user_stats.items():
            bbox = s.bbox
            rows.append(
                (
                    uid,
                    s.points,
                    s.start_ts.strftime("%Y-%m-%d %H:%M:%S") if s.start_ts else "",
                    s.end_ts.strftime("%Y-%m-%d %H:%M:%S") if s.end_ts else "",
                    bbox,
                )
            )
        rows.sort(key=lambda x: (x[1], x[0]), reverse=True)  # points desc
        with out_user.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "user_id",
                    "total_points_in_range",
                    "start_time",
                    "end_time",
                    "bbox_min_lon",
                    "bbox_min_lat",
                    "bbox_max_lon",
                    "bbox_max_lat",
                ]
            )
            for uid, pts, st, et, bbox in rows:
                if bbox is None:
                    w.writerow([uid, pts, st, et, "", "", "", ""])
                else:
                    w.writerow(
                        [
                            uid,
                            pts,
                            st,
                            et,
                            f"{bbox[0]:.6f}",
                            f"{bbox[1]:.6f}",
                            f"{bbox[2]:.6f}",
                            f"{bbox[3]:.6f}",
                        ]
                    )

    total_users = len({r[0] for r in file_rows})
    total_points = sum(r[2] for r in file_rows)
    print(f"data_dir={data_dir}")
    print(f"range_start={start} range_end={end or ''} min_points={min_points}")
    print(f"matched_files={matched_files} users_with_points={total_users} total_points={total_points}")
    print(f"saved: {out_file}")
    if args.out_user:
        print(f"saved: {args.out_user}")


if __name__ == "__main__":
    main()

