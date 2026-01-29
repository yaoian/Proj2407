"""
将 `.pth` 导出为便于查看的表格格式（CSV/JSONL）。

支持：
- test batch tuple（`Dataset/test*.pth`）：
- apartments（len=10）:
  loc_0, loc_T, loc_guess, loc_mean, meta, time, mask, bool_mask, query_len, observe_len
- taxi/geolife（len=8）:
  loc_0, loc_T, loc_guess, time, mask, bool_mask, query_len, observe_len

- dataset dict（如 `Dataset/apartment_dataset.pth`，包含 `trajs` 和可选 ` metadata`）：
  默认导出“轨迹级别”汇总（每条轨迹 1 行），避免输出过大。

- list cache（如 taxi cache，顶层 list，元素形如 `(traj, ...)` 或 `traj`）：
  默认导出“轨迹级别”汇总（每条轨迹 1 行）。

示例：
  python tools/export_test_pth_to_table.py
  python tools/export_test_pth_to_table.py --pattern "test_*.pth" --format csv --out-dir output/test_exports
  python tools/export_test_pth_to_table.py --max-samples 5 --only-valid
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


def _iter_test_files(dataset_dir: Path, pattern: str) -> Iterable[Path]:
    for p in sorted(dataset_dir.glob(pattern)):
        if p.is_file() and p.suffix == ".pth":
            yield p


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _schema_from_loaded(obj: Any) -> Tuple[str, Dict[str, torch.Tensor]]:
    if not isinstance(obj, tuple):
        raise TypeError(f"Unsupported .pth top-level type (expect tuple): {type(obj)}")

    if len(obj) == 10:
        (
            loc_0,
            loc_T,
            loc_guess,
            loc_mean,
            meta,
            time,
            mask,
            bool_mask,
            query_len,
            observe_len,
        ) = obj
        return (
            "apartments",
            {
                "loc_0": loc_0,
                "loc_T": loc_T,
                "loc_guess": loc_guess,
                "loc_mean": loc_mean,
                "meta": meta,
                "time": time,
                "mask": mask,
                "bool_mask": bool_mask,
                "query_len": query_len,
                "observe_len": observe_len,
            },
        )

    if len(obj) == 8:
        loc_0, loc_T, loc_guess, time, mask, bool_mask, query_len, observe_len = obj
        return (
            "taxi",
            {
                "loc_0": loc_0,
                "loc_T": loc_T,
                "loc_guess": loc_guess,
                "time": time,
                "mask": mask,
                "bool_mask": bool_mask,
                "query_len": query_len,
                "observe_len": observe_len,
            },
        )

    raise ValueError(f"Unsupported test batch tuple length: {len(obj)}")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _as_traj_tensor(sample: Any) -> Optional[torch.Tensor]:
    if isinstance(sample, torch.Tensor):
        return sample
    if isinstance(sample, (list, tuple)) and sample and isinstance(sample[0], torch.Tensor):
        return sample[0]
    return None


def _write_traj_summary_csv(
    out_path: Path,
    rows: Iterable[Tuple[Any, ...]],
    headers: List[str],
) -> int:
    _ensure_parent(out_path)
    n = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(list(r))
            n += 1
    return n


def _export_apartment_dataset_dict_to_csv(
    out_path: Path,
    data: Dict[str, Any],
    max_trajs: int,
) -> int:
    trajs = data.get("trajs")
    if not isinstance(trajs, list):
        raise ValueError("Dataset dict missing list key 'trajs'.")

    meta = None
    # Real key in this repo is " metadata" (leading space).
    if " metadata" in data:
        meta = data[" metadata"]
    elif "metadata" in data:
        meta = data["metadata"]

    headers = [
        "traj_idx",
        "points",
        "lon_min",
        "lon_max",
        "lat_min",
        "lat_max",
        "time_min",
        "time_max",
        "has_meta",
        "meta_shape",
    ]

    def gen() -> Iterable[Tuple[Any, ...]]:
        limit = len(trajs) if max_trajs <= 0 else min(len(trajs), max_trajs)
        for i in range(limit):
            x = trajs[i]
            if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] < 3:
                continue
            lon = x[0].to(torch.float32)
            lat = x[1].to(torch.float32)
            tim = x[2].to(torch.float32)
            lon_min = float(lon.min().item()) if lon.numel() else ""
            lon_max = float(lon.max().item()) if lon.numel() else ""
            lat_min = float(lat.min().item()) if lat.numel() else ""
            lat_max = float(lat.max().item()) if lat.numel() else ""
            time_min = float(tim.min().item()) if tim.numel() else ""
            time_max = float(tim.max().item()) if tim.numel() else ""

            has_meta = 0
            meta_shape = ""
            if isinstance(meta, list) and i < len(meta) and isinstance(meta[i], torch.Tensor):
                has_meta = 1
                meta_shape = "x".join(str(d) for d in meta[i].shape)
            yield (
                i,
                int(x.shape[1]),
                lon_min,
                lon_max,
                lat_min,
                lat_max,
                time_min,
                time_max,
                has_meta,
                meta_shape,
            )

    return _write_traj_summary_csv(out_path, gen(), headers)


def _export_list_cache_to_csv(
    out_path: Path,
    data: List[Any],
    max_trajs: int,
) -> int:
    headers = [
        "traj_idx",
        "points",
        "lon_min",
        "lon_max",
        "lat_min",
        "lat_max",
        "time_min",
        "time_max",
    ]

    def gen() -> Iterable[Tuple[Any, ...]]:
        limit = len(data) if max_trajs <= 0 else min(len(data), max_trajs)
        for i in range(limit):
            x = _as_traj_tensor(data[i])
            if x is None or x.ndim != 2 or x.shape[0] < 3:
                continue
            lon = x[0].to(torch.float32)
            lat = x[1].to(torch.float32)
            tim = x[2].to(torch.float32)
            yield (
                i,
                int(x.shape[1]),
                float(lon.min().item()) if lon.numel() else "",
                float(lon.max().item()) if lon.numel() else "",
                float(lat.min().item()) if lat.numel() else "",
                float(lat.max().item()) if lat.numel() else "",
                float(tim.min().item()) if tim.numel() else "",
                float(tim.max().item()) if tim.numel() else "",
            )

    return _write_traj_summary_csv(out_path, gen(), headers)


def _write_csv(
    out_path: Path,
    dataset_kind: str,
    tensors: Dict[str, torch.Tensor],
    max_samples: int,
    only_valid: bool,
) -> int:
    loc_0 = tensors["loc_0"].to("cpu")
    loc_T = tensors["loc_T"].to("cpu")
    loc_guess = tensors["loc_guess"].to("cpu")
    time = tensors["time"].to("cpu")
    mask = tensors["mask"].to("cpu")
    bool_mask = tensors["bool_mask"].to("cpu")
    query_len = tensors["query_len"].to("cpu")
    observe_len = tensors["observe_len"].to("cpu")

    B = int(loc_0.shape[0])
    L = int(loc_0.shape[-1])
    sample_n = B if max_samples <= 0 else min(B, max_samples)

    loc_mean = tensors.get("loc_mean")
    meta = tensors.get("meta")
    if isinstance(loc_mean, torch.Tensor):
        loc_mean = loc_mean.to("cpu")
    if isinstance(meta, torch.Tensor):
        meta = meta.to("cpu")

    # Use a unified (superset) schema for all test batch files, so that different datasets
    # can be compared easily in spreadsheet tools.
    headers = [
        "sample_idx",
        "step",
        "valid",
        "mask",
        "is_observed",
        "is_erased",
        "bool_mask",
        "time",
        "query_len",
        "observe_len",
        "loc_0_x",
        "loc_0_y",
        "loc_guess_x",
        "loc_guess_y",
        "loc_T_x",
        "loc_T_y",
        "loc_mean_x",
        "loc_mean_y",
        "meta_0",
        "meta_1",
        "meta_2",
        "meta_3",
    ]

    _ensure_parent(out_path)
    rows_written = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for b in range(sample_n):
            ql = int(query_len[b].item()) if query_len.ndim == 1 else int(query_len.item())
            ol = int(observe_len[b].item()) if observe_len.ndim == 1 else int(observe_len.item())

            mask_1d = mask[b, 0] if mask.ndim == 3 else mask[b]
            valid = mask_1d >= 0
            is_observed = (mask_1d <= 0.1) & valid
            is_erased = (mask_1d > 0.1) & valid
            bm = bool_mask[b, 0] if bool_mask.ndim == 3 else bool_mask[b, 0]

            for t in range(L):
                if only_valid and not bool(valid[t].item()):
                    continue
                w.writerow(
                    [
                        b,
                        t,
                        int(valid[t].item()),
                        _as_float(mask_1d[t].item()),
                        int(is_observed[t].item()),
                        int(is_erased[t].item()),
                        int(bm[t].item()),
                        _as_float(time[b, 0, t].item()),
                        ql,
                        ol,
                        _as_float(loc_0[b, 0, t].item()),
                        _as_float(loc_0[b, 1, t].item()),
                        _as_float(loc_guess[b, 0, t].item()),
                        _as_float(loc_guess[b, 1, t].item()),
                        _as_float(loc_T[b, 0, t].item()),
                        _as_float(loc_T[b, 1, t].item()),
                        _as_float(loc_mean[b, 0, 0].item()) if isinstance(loc_mean, torch.Tensor) else None,
                        _as_float(loc_mean[b, 1, 0].item()) if isinstance(loc_mean, torch.Tensor) else None,
                        int(meta[b, 0, t].item()) if isinstance(meta, torch.Tensor) else None,
                        int(meta[b, 1, t].item()) if isinstance(meta, torch.Tensor) else None,
                        int(meta[b, 2, t].item()) if isinstance(meta, torch.Tensor) else None,
                        int(meta[b, 3, t].item()) if isinstance(meta, torch.Tensor) else None,
                    ]
                )
                rows_written += 1

    return rows_written


def _write_jsonl(
    out_path: Path,
    dataset_kind: str,
    tensors: Dict[str, torch.Tensor],
    max_samples: int,
    only_valid: bool,
) -> int:
    loc_0 = tensors["loc_0"].to("cpu")
    loc_T = tensors["loc_T"].to("cpu")
    loc_guess = tensors["loc_guess"].to("cpu")
    time = tensors["time"].to("cpu")
    mask = tensors["mask"].to("cpu")
    bool_mask = tensors["bool_mask"].to("cpu")
    query_len = tensors["query_len"].to("cpu")
    observe_len = tensors["observe_len"].to("cpu")

    B = int(loc_0.shape[0])
    L = int(loc_0.shape[-1])
    sample_n = B if max_samples <= 0 else min(B, max_samples)

    loc_mean = tensors.get("loc_mean")
    meta = tensors.get("meta")
    if isinstance(loc_mean, torch.Tensor):
        loc_mean = loc_mean.to("cpu")
    if isinstance(meta, torch.Tensor):
        meta = meta.to("cpu")

    _ensure_parent(out_path)
    rows_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for b in range(sample_n):
            ql = int(query_len[b].item()) if query_len.ndim == 1 else int(query_len.item())
            ol = int(observe_len[b].item()) if observe_len.ndim == 1 else int(observe_len.item())

            mask_1d = mask[b, 0] if mask.ndim == 3 else mask[b]
            valid = mask_1d >= 0
            is_observed = (mask_1d <= 0.1) & valid
            is_erased = (mask_1d > 0.1) & valid
            bm = bool_mask[b, 0] if bool_mask.ndim == 3 else bool_mask[b, 0]

            for t in range(L):
                if only_valid and not bool(valid[t].item()):
                    continue
                row: Dict[str, Any] = {
                    "sample_idx": b,
                    "step": t,
                    "kind": dataset_kind,
                    "valid": int(valid[t].item()),
                    "mask": _as_float(mask_1d[t].item()),
                    "is_observed": int(is_observed[t].item()),
                    "is_erased": int(is_erased[t].item()),
                    "bool_mask": int(bm[t].item()),
                    "time": _as_float(time[b, 0, t].item()),
                    "query_len": ql,
                    "observe_len": ol,
                    "loc_0": [_as_float(loc_0[b, 0, t].item()), _as_float(loc_0[b, 1, t].item())],
                    "loc_guess": [
                        _as_float(loc_guess[b, 0, t].item()),
                        _as_float(loc_guess[b, 1, t].item()),
                    ],
                    "loc_T": [_as_float(loc_T[b, 0, t].item()), _as_float(loc_T[b, 1, t].item())],
                }
                row["loc_mean"] = (
                    [
                        _as_float(loc_mean[b, 0, 0].item()),
                        _as_float(loc_mean[b, 1, 0].item()),
                    ]
                    if isinstance(loc_mean, torch.Tensor)
                    else None
                )
                row["meta"] = (
                    [
                        int(meta[b, 0, t].item()),
                        int(meta[b, 1, t].item()),
                        int(meta[b, 2, t].item()),
                        int(meta[b, 3, t].item()),
                    ]
                    if isinstance(meta, torch.Tensor)
                    else None
                )
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1

    return rows_written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Dataset/test*.pth to CSV/JSONL for inspection.")
    p.add_argument("--dataset-dir", type=str, default="Dataset", help="Dataset directory to scan.")
    p.add_argument("--pattern", type=str, default="test*.pth", help='Glob pattern, default: "test*.pth"')
    p.add_argument("--out-dir", type=str, default="output/test_exports", help="Output directory.")
    p.add_argument("--format", type=str, default="csv", choices=["csv", "jsonl"], help="Output format.")
    p.add_argument("--max-samples", type=int, default=0, help="Limit number of samples B to export (0 means all).")
    p.add_argument("--only-valid", action="store_true", help="Only export positions with mask>=0.")
    p.add_argument(
        "--max-trajs",
        type=int,
        default=0,
        help="For dict/list datasets: limit number of trajectories to export (0 means all).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(_iter_test_files(dataset_dir, args.pattern))
    if not files:
        raise SystemExit(f"No files matched: {dataset_dir}/{args.pattern}")

    for pth in files:
        obj = torch.load(str(pth), map_location="cpu")
        out_name = pth.name.rsplit(".", 1)[0] + f".{args.format}"
        out_path = out_dir / out_name

        if isinstance(obj, tuple):
            kind, tensors = _schema_from_loaded(obj)
            if args.format == "csv":
                rows = _write_csv(
                    out_path,
                    kind,
                    tensors,
                    max_samples=int(args.max_samples),
                    only_valid=bool(args.only_valid),
                )
            else:
                rows = _write_jsonl(
                    out_path,
                    kind,
                    tensors,
                    max_samples=int(args.max_samples),
                    only_valid=bool(args.only_valid),
                )
            print(f"{pth} -> {out_path} | kind={kind} rows={rows}")
            continue

        if isinstance(obj, dict):
            if args.format != "csv":
                raise SystemExit("dict dataset export currently supports --format csv only.")
            rows = _export_apartment_dataset_dict_to_csv(out_path, obj, max_trajs=int(args.max_trajs))
            print(f"{pth} -> {out_path} | kind=dict rows={rows}")
            continue

        if isinstance(obj, list):
            if args.format != "csv":
                raise SystemExit("list cache export currently supports --format csv only.")
            rows = _export_list_cache_to_csv(out_path, obj, max_trajs=int(args.max_trajs))
            print(f"{pth} -> {out_path} | kind=list rows={rows}")
            continue

        raise TypeError(f"Unsupported .pth top-level type: {type(obj)}")

        # unreachable


if __name__ == "__main__":
    main()
