# -*- coding: utf-8 -*-
"""
GeoLife .plt 批处理 → Valhalla map-matching + 语义增强 → 输出 points.csv / edges.csv / stats.json

用法示例：
1) 批处理整个 GeoLife 根目录：
   python tools/geolife_valhalla_batch.py --geolife_root /path/to/Geolife/Data --out_dir ./out

2) 快速验证（只处理前 N 个文件）：
   python tools/geolife_valhalla_batch.py --geolife_root /path/to/Geolife/Data --out_dir ./out --limit_files 20

依赖（仅允许）：
- requests
- pandas（可选，不安装也能运行；脚本默认用 csv 流式写出）

Valhalla：
- 默认服务地址：http://localhost:8002（可用 --valhalla_url 覆盖）
- 主要接口：
  - POST {valhalla_url}/trace_attributes（优先）
  - POST {valhalla_url}/locate（trace_attributes 失败时 fallback）
"""

import argparse
import csv
import datetime as _dt
import json
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import requests

try:
    import pandas as pd  # noqa: F401
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False


POINTS_FIELDS = [
    "file_id",
    "seg_id",
    "point_idx",
    "lat",
    "lon",
    "timestamp",
    "way_id",
    "names",
    "road_class",
    "use",
    "surface",
    "lane_count",
    "speed_limit",
    "bridge",
    "tunnel",
    "dist_to_road",
    "dist_along_edge",
    "source",
]

EDGES_FIELDS = [
    "way_id",
    "names",
    "road_class",
    "use",
    "surface",
    "lane_count",
    "speed_limit",
    "bridge",
    "tunnel",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8  # WGS84 mean radius
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def scan_plt_files(geolife_root: str) -> list[str]:
    paths: list[str] = []
    for root, _, files in os.walk(geolife_root):
        for name in files:
            if name.lower().endswith(".plt"):
                paths.append(os.path.join(root, name))
    paths.sort()
    return paths


def parse_geolife_timestamp(date_str: str, time_str: str):
    try:
        return _dt.datetime.strptime(date_str.strip() + " " + time_str.strip(), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def read_single_plt(file_path: str) -> list[dict]:
    """
    GeoLife .plt：
    前 6 行 header
    之后每行 CSV：lat, lon, 0, altitude, days, date, time
    """
    points: list[dict] = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for _ in range(6):
            next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            lat_s, lon_s, _, _, _, date_s, time_s = parts[:7]
            try:
                lat = float(lat_s)
                lon = float(lon_s)
            except Exception:
                continue
            ts = parse_geolife_timestamp(date_s, time_s)
            if ts is None:
                continue
            points.append({"lat": lat, "lon": lon, "ts": ts})

    if not points:
        return []

    points.sort(key=lambda x: x["ts"])

    # 连续重复 (lat, lon) 去重：只去掉紧邻重复
    deduped = [points[0]]
    for p in points[1:]:
        prev = deduped[-1]
        if p["lat"] == prev["lat"] and p["lon"] == prev["lon"]:
            continue
        deduped.append(p)
    return deduped


def split_by_time_gap(points: list[dict], time_gap_sec: int) -> list[list[dict]]:
    if not points:
        return []

    segments: list[list[dict]] = []
    cur = [points[0]]
    for p in points[1:]:
        dt = (p["ts"] - cur[-1]["ts"]).total_seconds()
        if dt > time_gap_sec:
            segments.append(cur)
            cur = [p]
        else:
            cur.append(p)
    segments.append(cur)

    return [seg for seg in segments if len(seg) >= 2]


def downsample_segment(
    points: list[dict],
    mode: str,
    downsample_time_sec: int,
    downsample_distance_m: float,
) -> list[dict]:
    if mode == "none":
        return points

    kept = [points[0]]
    last = points[0]

    if mode == "time":
        for p in points[1:]:
            dt = (p["ts"] - last["ts"]).total_seconds()
            if dt >= downsample_time_sec:
                kept.append(p)
                last = p
        return kept if len(kept) >= 2 else []

    if mode == "distance":
        for p in points[1:]:
            d = haversine_m(last["lat"], last["lon"], p["lat"], p["lon"])
            if d >= downsample_distance_m:
                kept.append(p)
                last = p
        return kept if len(kept) >= 2 else []

    raise ValueError("Invalid downsample_mode: %r" % mode)


def _to_iso8601(dt: _dt.datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _as_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def _names_to_str(names_val) -> str:
    if names_val is None:
        return ""
    if isinstance(names_val, str):
        return names_val
    if isinstance(names_val, list):
        out: list[str] = []
        for it in names_val:
            if it is None:
                continue
            if isinstance(it, str):
                s = it.strip()
                if s:
                    out.append(s)
                continue
            if isinstance(it, dict):
                s = it.get("value") or it.get("name") or it.get("text")
                if isinstance(s, str) and s.strip():
                    out.append(s.strip())
                continue
        seen = set()
        uniq: list[str] = []
        for s in out:
            if s in seen:
                continue
            seen.add(s)
            uniq.append(s)
        return "|".join(uniq)
    return _as_str(names_val)


def normalize_edge_attrs(edge_obj: dict) -> dict:
    def g(key: str):
        if key in edge_obj:
            return edge_obj.get(key)
        pref = "edge." + key
        if pref in edge_obj:
            return edge_obj.get(pref)
        last = key.split(".")[-1]
        if last in edge_obj:
            return edge_obj.get(last)
        return None

    way_id = g("way_id")
    return {
        "way_id": _as_str(way_id),
        "names": _names_to_str(g("names")),
        "road_class": _as_str(g("road_class")),
        "use": _as_str(g("use")),
        "surface": _as_str(g("surface")),
        "lane_count": _as_str(g("lane_count")),
        "speed_limit": _as_str(g("speed_limit")),
        "bridge": _as_str(g("bridge")),
        "tunnel": _as_str(g("tunnel")),
    }


def empty_semantics(source: str, way_id: str = "") -> dict:
    return {
        "way_id": _as_str(way_id),
        "names": "",
        "road_class": "",
        "use": "",
        "surface": "",
        "lane_count": "",
        "speed_limit": "",
        "bridge": "",
        "tunnel": "",
        "dist_to_road": "",
        "dist_along_edge": "",
        "source": source,
    }


def post_json_with_retry(session: requests.Session, url: str, payload: dict, retry: int, timeout_sec: int) -> dict:
    last_err = None
    for attempt in range(retry + 1):
        try:
            resp = session.post(url, json=payload, timeout=timeout_sec)
            if resp.status_code != 200:
                last_err = RuntimeError("HTTP %s: %s" % (resp.status_code, resp.text[:500]))
                raise last_err
            return resp.json()
        except Exception as e:
            last_err = e
            if attempt >= retry:
                break
            time.sleep(min(5.0, 0.5 * (2**attempt)))
    raise last_err


def call_trace_attributes(session: requests.Session, args, shape_points: list[dict]):
    url = args.valhalla_url.rstrip("/") + "/trace_attributes"
    payload = {
        "costing": args.costing,
        "shape_match": args.shape_match,
        "shape": [{"lat": p["lat"], "lon": p["lon"]} for p in shape_points],
        "filters": {
            "action": "include",
            "attributes": [
                "edge.way_id",
                "edge.names",
                "edge.road_class",
                "edge.use",
                "edge.surface",
                "edge.lane_count",
                "edge.speed_limit",
                "edge.bridge",
                "edge.tunnel",
                "matched.edge_index",
                "matched.distance_from_trace_point",
                "matched.distance_along_edge",
            ],
        },
    }
    data = post_json_with_retry(session, url, payload, args.retry, args.timeout_sec)
    edges = data.get("edges")
    matched_points = data.get("matched_points")
    if not isinstance(edges, list) or not isinstance(matched_points, list):
        raise ValueError("trace_attributes missing edges/matched_points")
    if len(matched_points) < len(shape_points):
        raise ValueError("trace_attributes matched_points too short (%d < %d)" % (len(matched_points), len(shape_points)))
    return edges, matched_points


def call_locate(session: requests.Session, args, shape_points: list[dict]) -> list[str]:
    url = args.valhalla_url.rstrip("/") + "/locate"
    payload = {
        "costing": args.costing,
        "verbose": True,
        "locations": [{"lat": p["lat"], "lon": p["lon"]} for p in shape_points],
    }
    data = post_json_with_retry(session, url, payload, args.retry, args.timeout_sec)

    way_ids = ["" for _ in shape_points]
    locations = data.get("locations")
    if isinstance(locations, list) and len(locations) >= len(shape_points):
        for i in range(len(shape_points)):
            loc = locations[i]
            if not isinstance(loc, dict):
                continue
            edges = loc.get("edges")
            if not isinstance(edges, list) or not edges:
                continue
            cand = edges[0]
            if isinstance(cand, dict):
                wid = cand.get("way_id") or cand.get("edge.way_id") or cand.get("id")
                if wid is not None:
                    way_ids[i] = _as_str(wid)
        return way_ids

    edges = data.get("edges")
    if isinstance(edges, list) and edges:
        wid = None
        if isinstance(edges[0], dict):
            wid = edges[0].get("way_id") or edges[0].get("edge.way_id") or edges[0].get("id")
        if wid is not None:
            return [_as_str(wid) for _ in shape_points]
    return way_ids


def split_two_parts_min2(n: int) -> int:
    mid = n // 2
    if mid < 2:
        mid = 2
    if n - mid < 2:
        mid = n - 2
    return mid


def enrich_chunk_recursive(
    session: requests.Session,
    args,
    shape_points: list[dict],
    edge_cache: dict,
    edge_lock: threading.Lock,
    req_stats: dict,
) -> list[dict]:
    n = len(shape_points)
    if n < 2:
        req_stats["points_failed"] += n
        return [empty_semantics("failed") for _ in shape_points]

    try:
        req_stats["trace_attributes_requests"] += 1
        edges, matched_points = call_trace_attributes(session, args, shape_points)

        edges_updates: dict[str, dict] = {}
        for e in edges:
            if not isinstance(e, dict):
                continue
            norm = normalize_edge_attrs(e)
            wid = norm.get("way_id") or ""
            if wid:
                edges_updates[wid] = {
                    "way_id": wid,
                    "names": norm.get("names", ""),
                    "road_class": norm.get("road_class", ""),
                    "use": norm.get("use", ""),
                    "surface": norm.get("surface", ""),
                    "lane_count": norm.get("lane_count", ""),
                    "speed_limit": norm.get("speed_limit", ""),
                    "bridge": norm.get("bridge", ""),
                    "tunnel": norm.get("tunnel", ""),
                }
        if edges_updates:
            with edge_lock:
                for wid, attrs in edges_updates.items():
                    if wid not in edge_cache:
                        edge_cache[wid] = attrs

        semantics: list[dict] = []
        for i in range(n):
            mp = matched_points[i] if i < len(matched_points) else {}
            if not isinstance(mp, dict):
                mp = {}

            edge_index = mp.get("edge_index")
            dist_to_road = mp.get("distance_from_trace_point")
            dist_along_edge = mp.get("distance_along_edge")

            if isinstance(edge_index, int) and 0 <= edge_index < len(edges) and isinstance(edges[edge_index], dict):
                norm = normalize_edge_attrs(edges[edge_index])
                sem = empty_semantics("trace_attributes", way_id=norm.get("way_id", ""))
                sem.update(
                    {
                        "names": norm.get("names", ""),
                        "road_class": norm.get("road_class", ""),
                        "use": norm.get("use", ""),
                        "surface": norm.get("surface", ""),
                        "lane_count": norm.get("lane_count", ""),
                        "speed_limit": norm.get("speed_limit", ""),
                        "bridge": norm.get("bridge", ""),
                        "tunnel": norm.get("tunnel", ""),
                    }
                )
            else:
                sem = empty_semantics("trace_attributes")

            sem["dist_to_road"] = "" if dist_to_road is None else _as_str(dist_to_road)
            sem["dist_along_edge"] = "" if dist_along_edge is None else _as_str(dist_along_edge)
            semantics.append(sem)

        req_stats["points_trace_attributes"] += n
        req_stats["points_matched_with_way_id"] += sum(1 for s in semantics if s.get("way_id"))
        return semantics

    except Exception:
        req_stats["trace_attributes_failures"] += 1

    if n == 2:
        req_stats["fallback_chunks"] += 1
        try:
            req_stats["locate_requests"] += 1
            way_ids = call_locate(session, args, shape_points)
            out: list[dict] = []
            for wid in way_ids:
                out.append(empty_semantics("locate", way_id=wid))
            req_stats["points_locate"] += n
            req_stats["points_matched_with_way_id"] += sum(1 for s in out if s.get("way_id"))
            return out
        except Exception:
            req_stats["locate_failures"] += 1
            req_stats["points_failed"] += n
            return [empty_semantics("failed") for _ in shape_points]

    if n == 3:
        left = enrich_chunk_recursive(session, args, shape_points[:2], edge_cache, edge_lock, req_stats)
        right = enrich_chunk_recursive(session, args, shape_points[1:3], edge_cache, edge_lock, req_stats)
        return [left[0], left[1], right[1]]

    mid = split_two_parts_min2(n)
    left = enrich_chunk_recursive(session, args, shape_points[:mid], edge_cache, edge_lock, req_stats)
    right = enrich_chunk_recursive(session, args, shape_points[mid:], edge_cache, edge_lock, req_stats)
    return left + right


def iter_chunks_indices(n: int, max_points_per_request: int):
    if n < 2:
        return
    m = max_points_per_request
    if m < 2:
        raise ValueError("--max_points_per_request must be >= 2")

    if m == 2:
        i = 0
        while i + 1 < n:
            yield (i, i + 2, 0)
            i += 2
        if n % 2 == 1:
            yield (n - 2, n, 1)
        return

    i = 0
    while i < n:
        j = min(i + m, n)
        if j - i >= 2:
            yield (i, j, 0)
            i = j
            continue
        yield (n - 2, n, 1)
        return


def points_writer_thread(points_csv_path: str, queue: Queue, stop_sentinel) -> None:
    os.makedirs(os.path.dirname(points_csv_path), exist_ok=True)
    need_header = True
    if os.path.exists(points_csv_path):
        try:
            if os.path.getsize(points_csv_path) > 0:
                need_header = False
        except Exception:
            need_header = True

    with open(points_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=POINTS_FIELDS)
        if need_header:
            writer.writeheader()

        while True:
            item = queue.get()
            try:
                if item is stop_sentinel:
                    return
                rows = item
                for row in rows:
                    writer.writerow(row)
            finally:
                queue.task_done()


def process_one_file(
    file_path: str,
    file_id: str,
    args,
    out_queue: Queue,
    edge_cache: dict,
    edge_lock: threading.Lock,
) -> dict:
    session = requests.Session()

    local = {
        "files_ok": 0,
        "files_failed": 0,
        "points_raw": 0,
        "points_after_dedup": 0,
        "segments_kept": 0,
        "segments_dropped_short": 0,
        "points_output": 0,
        "trace_attributes_requests": 0,
        "trace_attributes_failures": 0,
        "locate_requests": 0,
        "locate_failures": 0,
        "fallback_chunks": 0,
        "points_trace_attributes": 0,
        "points_locate": 0,
        "points_failed": 0,
        "points_matched_with_way_id": 0,
    }

    try:
        pts = read_single_plt(file_path)
        local["points_raw"] = len(pts)
        local["points_after_dedup"] = len(pts)

        segs = split_by_time_gap(pts, args.time_gap_sec)
        if not segs:
            local["segments_dropped_short"] += 1
            local["files_ok"] = 1
            return local

        buf: list[dict] = []
        for seg_id, seg in enumerate(segs):
            ds = downsample_segment(seg, args.downsample_mode, args.downsample_time_sec, args.downsample_distance_m)
            if len(ds) < 2:
                local["segments_dropped_short"] += 1
                continue
            local["segments_kept"] += 1

            semantics_all: list[dict | None] = [None] * len(ds)
            for start, end, overlap_prefix in iter_chunks_indices(len(ds), args.max_points_per_request):
                chunk_points = ds[start:end]
                sem_chunk = enrich_chunk_recursive(session, args, chunk_points, edge_cache, edge_lock, local)
                for i in range(overlap_prefix, len(chunk_points)):
                    semantics_all[start + i] = sem_chunk[i]

            for i, s in enumerate(semantics_all):
                if s is None:
                    semantics_all[i] = empty_semantics("failed")
                    local["points_failed"] += 1

            for i, p in enumerate(ds):
                sem = semantics_all[i]
                row = {
                    "file_id": file_id,
                    "seg_id": str(seg_id),
                    "point_idx": str(i),
                    "lat": _as_str(p["lat"]),
                    "lon": _as_str(p["lon"]),
                    "timestamp": _to_iso8601(p["ts"]),
                    "way_id": sem.get("way_id", ""),
                    "names": sem.get("names", ""),
                    "road_class": sem.get("road_class", ""),
                    "use": sem.get("use", ""),
                    "surface": sem.get("surface", ""),
                    "lane_count": sem.get("lane_count", ""),
                    "speed_limit": sem.get("speed_limit", ""),
                    "bridge": sem.get("bridge", ""),
                    "tunnel": sem.get("tunnel", ""),
                    "dist_to_road": sem.get("dist_to_road", ""),
                    "dist_along_edge": sem.get("dist_along_edge", ""),
                    "source": sem.get("source", ""),
                }
                buf.append(row)
                local["points_output"] += 1

                if len(buf) >= 500:
                    out_queue.put(buf)
                    buf = []

        if buf:
            out_queue.put(buf)

        local["files_ok"] = 1
        return local

    except Exception as e:
        logging.exception("Failed file: %s (%s)", file_path, e)
        local["files_failed"] = 1
        return local

    finally:
        try:
            session.close()
        except Exception:
            pass


def write_edges_csv(edges_csv_path: str, edge_cache: dict) -> None:
    os.makedirs(os.path.dirname(edges_csv_path), exist_ok=True)

    def sort_key(wid: str):
        try:
            return (0, int(wid))
        except Exception:
            return (1, str(wid))

    way_ids = sorted(edge_cache.keys(), key=sort_key)
    with open(edges_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EDGES_FIELDS)
        writer.writeheader()
        for wid in way_ids:
            attrs = edge_cache.get(wid, {})
            row = {k: _as_str(attrs.get(k, "")) for k in EDGES_FIELDS}
            writer.writerow(row)


def write_stats_json(stats_path: str, stats: dict) -> None:
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Batch GeoLife .plt → Valhalla map-matching + semantic enrichment")
    p.add_argument("--geolife_root", required=True, help="GeoLife 根目录（包含 user_id/Trajectory/*.plt）")
    p.add_argument("--out_dir", required=True, help="输出目录（points.csv / edges.csv / stats.json）")

    p.add_argument("--valhalla_url", default="http://localhost:8002", help="Valhalla 服务地址")
    p.add_argument("--costing", default="auto", help="Valhalla costing（默认 auto）")
    p.add_argument("--shape_match", default="walk_or_snap", help="Valhalla shape_match（默认 walk_or_snap）")

    p.add_argument("--time_gap_sec", type=int, default=300, help="切段阈值：相邻点时间差 > N 秒（默认 300）")
    p.add_argument("--max_points_per_request", type=int, default=300, help="每次请求最多点数（默认 300）")

    p.add_argument("--downsample_mode", default="none", choices=["none", "time", "distance"], help="下采样模式")
    p.add_argument("--downsample_time_sec", type=int, default=2, help="time 下采样：相邻保留点间隔 >= N 秒")
    p.add_argument("--downsample_distance_m", type=float, default=15.0, help="distance 下采样：相邻保留点距离 >= N 米")

    p.add_argument("--concurrency", type=int, default=4, help="并发处理 plt 文件数（默认 4）")
    p.add_argument("--retry", type=int, default=2, help="HTTP 重试次数（默认 2）")
    p.add_argument("--timeout_sec", type=int, default=60, help="HTTP 超时秒数（默认 60）")
    p.add_argument("--limit_files", type=int, default=0, help="只处理前 N 个文件（默认 0=不限制）")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    geolife_root = os.path.abspath(args.geolife_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(geolife_root):
        logging.error("Invalid --geolife_root (not a directory): %s", geolife_root)
        logging.error("CWD: %s", os.getcwd())
        logging.error(
            "If you run from repo root, try: --geolife_root Dataset/Geolife/Data; "
            "if you run from tools/, try: --geolife_root ../Dataset/Geolife/Data"
        )
        sys.exit(2)

    points_csv_path = os.path.join(out_dir, "points.csv")
    edges_csv_path = os.path.join(out_dir, "edges.csv")
    stats_json_path = os.path.join(out_dir, "stats.json")

    plt_files = scan_plt_files(geolife_root)
    total_scanned = len(plt_files)
    if total_scanned == 0 and os.path.isdir(os.path.join(geolife_root, "Data")):
        logging.error("No .plt found under --geolife_root: %s", geolife_root)
        logging.error("Hint: your GeoLife root might be: %s", os.path.join(geolife_root, "Data"))
        sys.exit(2)
    if args.limit_files and args.limit_files > 0:
        plt_files = plt_files[: args.limit_files]

    logging.info("Scan done: total=%d, will_process=%d, geolife_root=%s", total_scanned, len(plt_files), geolife_root)
    logging.info("Valhalla: url=%s costing=%s shape_match=%s", args.valhalla_url, args.costing, args.shape_match)
    logging.info("Output: %s", out_dir)

    edge_cache: dict[str, dict] = {}
    edge_lock = threading.Lock()

    q: Queue = Queue(maxsize=max(100, args.concurrency * 10))
    stop_sentinel = object()
    wt = threading.Thread(
        target=points_writer_thread,
        args=(points_csv_path, q, stop_sentinel),
        name="points-writer",
        daemon=True,
    )
    wt.start()

    agg = {
        "total_files_scanned": total_scanned,
        "total_files_target": len(plt_files),
        "files_ok": 0,
        "files_failed": 0,
        "points_raw": 0,
        "points_after_dedup": 0,
        "segments_kept": 0,
        "segments_dropped_short": 0,
        "points_output": 0,
        "trace_attributes_requests": 0,
        "trace_attributes_failures": 0,
        "locate_requests": 0,
        "locate_failures": 0,
        "fallback_chunks": 0,
        "points_trace_attributes": 0,
        "points_locate": 0,
        "points_failed": 0,
        "points_matched_with_way_id": 0,
    }

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = []
        for fp in plt_files:
            rel = os.path.relpath(fp, geolife_root).replace("\\", "/")
            futures.append(ex.submit(process_one_file, fp, rel, args, q, edge_cache, edge_lock))

        done_count = 0
        for fut in as_completed(futures):
            res = fut.result()
            for k in agg:
                if k in res and isinstance(res[k], (int, float)):
                    agg[k] += res[k]
            done_count += 1
            if done_count % 10 == 0 or done_count == len(futures):
                logging.info("Progress: %d/%d files processed", done_count, len(futures))

    q.join()
    q.put(stop_sentinel)
    q.join()
    wt.join(timeout=5.0)

    with edge_lock:
        edges_count = len(edge_cache)
        write_edges_csv(edges_csv_path, edge_cache)

    elapsed = time.time() - started
    total_out = agg["points_output"]
    match_rate = (agg["points_matched_with_way_id"] / total_out) if total_out else 0.0

    stats = dict(agg)
    stats.update(
        {
            "edges_unique": edges_count,
            "match_rate": match_rate,
            "elapsed_sec": elapsed,
            "has_pandas": _HAS_PANDAS,
        }
    )
    write_stats_json(stats_json_path, stats)

    logging.info(
        "Done. files_ok=%d files_failed=%d points=%d match_rate=%.4f edges_unique=%d elapsed=%.1fs",
        agg["files_ok"],
        agg["files_failed"],
        total_out,
        match_rate,
        edges_count,
        elapsed,
    )
    logging.info("Wrote: %s", points_csv_path)
    logging.info("Wrote: %s", edges_csv_path)
    logging.info("Wrote: %s", stats_json_path)


if __name__ == "__main__":
    main()
