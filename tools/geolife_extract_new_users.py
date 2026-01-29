"""
提取 GeoLife Trajectories 1.3 相对旧版本“新增用户”的轨迹信息（按目录差集判断）。

GeoLife 常见结构：
<root>/Data/<user_id>/Trajectory/*.plt
<root>/Data/<user_id>/labels.txt  (可选)

使用：
python tools/geolife_extract_new_users.py \
  --v13 /path/to/Geolife\ Trajectories\ 1.3 \
  --v12 /path/to/Geolife\ Trajectories\ 1.2 \
  --out output/geolife_new_users.csv

说明：
- “新增用户”定义为：v13 Data 目录下存在但 v12 Data 目录下不存在的 user_id 目录
- 统计信息：plt 文件数、总点数、时间跨度、经纬度 bbox、是否有 labels.txt
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass(frozen=True)
class UserStats:
    user_id: str
    plt_files: int
    total_points: int
    start_time: Optional[str]
    end_time: Optional[str]
    bbox_min_lon: Optional[float]
    bbox_min_lat: Optional[float]
    bbox_max_lon: Optional[float]
    bbox_max_lat: Optional[float]
    has_labels: bool


def _find_data_dir(root: Path) -> Path:
    if root.is_dir() and (root / "Data").is_dir():
        return root / "Data"
    if root.is_dir() and root.name == "Data":
        return root
    raise FileNotFoundError(f"Cannot find Data/ under: {root}")


def _iter_user_dirs(data_dir: Path) -> Iterable[Path]:
    for p in sorted(data_dir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if name.isdigit():
            yield p


def _parse_geolife_ts(date_str: str, time_str: str) -> Optional[datetime]:
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _scan_plt_file(path: Path) -> Tuple[int, Optional[datetime], Optional[datetime], Optional[Tuple[float, float, float, float]]]:
    # GeoLife .plt 前 6 行通常是 header；数据行格式通常是：
    # lat,lon,0,altitude,days,date,time
    points = 0
    first_ts: Optional[datetime] = None
    last_ts: Optional[datetime] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # min_lon, min_lat, max_lon, max_lat

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

                if first_ts is None:
                    first_ts = ts
                last_ts = ts
                points += 1

                if bbox is None:
                    bbox = (lon, lat, lon, lat)
                else:
                    min_lon, min_lat, max_lon, max_lat = bbox
                    bbox = (
                        min(min_lon, lon),
                        min(min_lat, lat),
                        max(max_lon, lon),
                        max(max_lat, lat),
                    )
    except OSError:
        return 0, None, None, None

    return points, first_ts, last_ts, bbox


def _merge_bbox(a: Optional[Tuple[float, float, float, float]], b: Optional[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if a is None:
        return b
    if b is None:
        return a
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _collect_user_stats(user_dir: Path) -> UserStats:
    user_id = user_dir.name
    traj_dir = user_dir / "Trajectory"
    plt_paths = sorted(traj_dir.glob("*.plt")) if traj_dir.is_dir() else []
    has_labels = (user_dir / "labels.txt").is_file()

    total_points = 0
    start_ts: Optional[datetime] = None
    end_ts: Optional[datetime] = None
    bbox: Optional[Tuple[float, float, float, float]] = None

    for p in plt_paths:
        pts, first, last, b = _scan_plt_file(p)
        total_points += pts
        if first is not None and (start_ts is None or first < start_ts):
            start_ts = first
        if last is not None and (end_ts is None or last > end_ts):
            end_ts = last
        bbox = _merge_bbox(bbox, b)

    if bbox is None:
        bbox_min_lon = bbox_min_lat = bbox_max_lon = bbox_max_lat = None
    else:
        bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat = bbox

    return UserStats(
        user_id=user_id,
        plt_files=len(plt_paths),
        total_points=total_points,
        start_time=start_ts.strftime("%Y-%m-%d %H:%M:%S") if start_ts else None,
        end_time=end_ts.strftime("%Y-%m-%d %H:%M:%S") if end_ts else None,
        bbox_min_lon=bbox_min_lon,
        bbox_min_lat=bbox_min_lat,
        bbox_max_lon=bbox_max_lon,
        bbox_max_lat=bbox_max_lat,
        has_labels=has_labels,
    )


def _user_ids(data_dir: Path) -> Set[str]:
    return {p.name for p in _iter_user_dirs(data_dir)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract new users in GeoLife 1.3 vs older version by directory diff.")
    p.add_argument("--v13", type=str, required=True, help="GeoLife 1.3 root path (or Data/ path).")
    p.add_argument("--v12", type=str, required=True, help="Older GeoLife root path (or Data/ path).")
    p.add_argument("--out", type=str, required=True, help="Output CSV path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    v13_data = _find_data_dir(Path(args.v13))
    v12_data = _find_data_dir(Path(args.v12))

    v13_users = _user_ids(v13_data)
    v12_users = _user_ids(v12_data)
    new_users = sorted(v13_users - v12_users)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats: List[UserStats] = []
    for uid in new_users:
        stats.append(_collect_user_stats(v13_data / uid))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "user_id",
                "plt_files",
                "total_points",
                "start_time",
                "end_time",
                "bbox_min_lon",
                "bbox_min_lat",
                "bbox_max_lon",
                "bbox_max_lat",
                "has_labels",
            ]
        )
        for s in stats:
            w.writerow(
                [
                    s.user_id,
                    s.plt_files,
                    s.total_points,
                    s.start_time or "",
                    s.end_time or "",
                    "" if s.bbox_min_lon is None else f"{s.bbox_min_lon:.6f}",
                    "" if s.bbox_min_lat is None else f"{s.bbox_min_lat:.6f}",
                    "" if s.bbox_max_lon is None else f"{s.bbox_max_lon:.6f}",
                    "" if s.bbox_max_lat is None else f"{s.bbox_max_lat:.6f}",
                    int(s.has_labels),
                ]
            )

    print(f"v13_data={v13_data}")
    print(f"v12_data={v12_data}")
    print(f"v13_users={len(v13_users)} v12_users={len(v12_users)} new_users={len(new_users)}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

