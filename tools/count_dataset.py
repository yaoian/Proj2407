"""
统计 `.pth` 轨迹数据集的轨迹条数，并可选扫描前 N 条轨迹的长度分布。

用于快速判断数据规模（例如 Xian/Chengdu 数据集有多大），避免训练前“拍脑袋”配置 batch_size。
"""

import argparse
from typing import Any, Optional, Sequence, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count trajectories in a .pth dataset file.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to .pth file, e.g. Dataset/Xian_nov_cache.pth",
    )
    parser.add_argument(
        "--scan",
        type=int,
        default=0,
        help="Scan first N trajectories and print length stats (0 disables).",
    )
    return parser.parse_args()


def _as_traj(sample: Any) -> Optional[torch.Tensor]:
    if isinstance(sample, torch.Tensor):
        return sample
    if isinstance(sample, (list, tuple)) and sample:
        first = sample[0]
        if isinstance(first, torch.Tensor):
            return first
    return None


def _maybe_test_batch_tuple(data: Tuple[Any, ...]) -> Optional[int]:
    if len(data) < 6 or len(data) > 32:
        return None
    if not all(isinstance(x, torch.Tensor) for x in data[:6]):
        return None
    first = data[0]
    if first.ndim >= 1 and first.shape[0] > 1:
        return int(first.shape[0])
    return None


def _count_from_loaded(data: Any) -> Tuple[str, int]:
    if isinstance(data, dict):
        if "trajs" in data and isinstance(data["trajs"], Sequence):
            return "dict[trajs]", len(data["trajs"])
        if "trajectories" in data and isinstance(data["trajectories"], Sequence):
            return "dict[trajectories]", len(data["trajectories"])
        raise ValueError(f"Unsupported dict keys: {list(data.keys())}")

    if isinstance(data, tuple):
        batch_size = _maybe_test_batch_tuple(data)
        if batch_size is not None:
            raise ValueError(
                "This .pth looks like a saved test batch tuple, not a dataset. "
                f"batch_size={batch_size}"
            )
        return "tuple", len(data)

    if isinstance(data, list):
        return "list", len(data)

    raise ValueError(f"Unsupported top-level type: {type(data)}")


def main() -> None:
    args = parse_args()
    data = torch.load(args.path, map_location="cpu")

    try:
        kind, count = _count_from_loaded(data)
    except ValueError as exc:
        print(f"path={args.path}")
        print(f"type={type(data)}")
        print(f"error={exc}")
        return
    print(f"path={args.path}")
    print(f"type={type(data)} kind={kind}")
    print(f"traj_count={count}")

    scan_n = int(args.scan)
    if scan_n <= 0:
        return

    if isinstance(data, dict):
        seq = data.get("trajs") or data.get("trajectories")
    else:
        seq = data

    scan_n = min(scan_n, len(seq))
    lengths = []
    first_traj = None
    for i in range(scan_n):
        traj = _as_traj(seq[i])
        if traj is None:
            continue
        if first_traj is None:
            first_traj = traj
        if traj.ndim >= 2:
            lengths.append(int(traj.shape[-1]))

    if first_traj is not None:
        print(f"first_traj_shape={tuple(first_traj.shape)} dtype={first_traj.dtype}")
    if lengths:
        lengths_sorted = sorted(lengths)
        print(f"scan_n={scan_n}")
        print(
            "len_min={} len_p50={} len_p90={} len_max={}".format(
                lengths_sorted[0],
                lengths_sorted[len(lengths_sorted) // 2],
                lengths_sorted[int(len(lengths_sorted) * 0.9)],
                lengths_sorted[-1],
            )
        )


if __name__ == "__main__":
    main()
