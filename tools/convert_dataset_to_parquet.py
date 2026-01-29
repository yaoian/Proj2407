"""
将本项目的 `.pth` 轨迹数据集导出为 Parquet（便于用 pandas/pyarrow 等工具查看与分析）。

- 支持 apartments（dict: {"trajs", " metadata"/"metadata"}）与 taxi（list/tuple of samples）两种格式。
"""

import argparse
from pathlib import Path
import sys

import torch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pyarrow'. Install with: uv pip install pyarrow"
    ) from exc


def _load_dataset(path: Path):
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict) and "trajs" in data:
        meta = None
        if " metadata" in data:
            meta = data[" metadata"]
        elif "metadata" in data:
            meta = data["metadata"]
        return "apartments", data["trajs"], meta
    if isinstance(data, (list, tuple)):
        return "taxi", data, None
    raise ValueError(f"Unsupported dataset format: {type(data)}")


def _traj_from_sample(sample):
    if isinstance(sample, (list, tuple)):
        return sample[0]
    return sample


def _build_table(traj_id, traj, meta):
    if traj.ndim != 2 or traj.shape[0] < 3:
        raise ValueError(f"Unexpected traj shape: {tuple(traj.shape)}")
    traj = traj.detach().cpu()

    length = traj.shape[1]
    traj_id_col = torch.full((length,), traj_id, dtype=torch.int64).numpy()
    step_col = torch.arange(length, dtype=torch.int32).numpy()

    arrays = [
        pa.array(traj_id_col, type=pa.int64()),
        pa.array(step_col, type=pa.int32()),
        pa.array(traj[0].numpy(), type=pa.float32()),
        pa.array(traj[1].numpy(), type=pa.float32()),
        pa.array(traj[2].numpy(), type=pa.float32()),
    ]
    names = ["traj_id", "step", "lon", "lat", "time"]

    if meta is not None:
        meta = meta.detach().cpu()
        if meta.shape[0] < 4:
            raise ValueError(f"Unexpected meta shape: {tuple(meta.shape)}")
        arrays.extend(
            [
                pa.array(meta[0].numpy(), type=pa.int32()),
                pa.array(meta[1].numpy(), type=pa.int32()),
                pa.array(meta[2].numpy(), type=pa.int32()),
                pa.array(meta[3].numpy(), type=pa.int32()),
            ]
        )
        names.extend(["m0", "m1", "m2", "m3"])

    return pa.Table.from_arrays(arrays, names=names)


def main():
    parser = argparse.ArgumentParser(description="Convert .pth dataset to Parquet.")
    parser.add_argument(
        "--input",
        default="Dataset/apartment_dataset.pth",
        help="Path to .pth dataset file",
    )
    parser.add_argument(
        "--output",
        default="Dataset/apartment_dataset.parquet",
        help="Output Parquet path",
    )
    parser.add_argument(
        "--max-traj",
        type=int,
        default=None,
        help="Limit number of trajectories to export",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    kind, trajs, metas = _load_dataset(input_path)

    max_traj = args.max_traj if args.max_traj is not None else len(trajs)

    writer = None
    written = 0
    for traj_id in range(min(max_traj, len(trajs))):
        sample = trajs[traj_id]
        traj = _traj_from_sample(sample)
        meta = metas[traj_id] if metas is not None else None

        table = _build_table(traj_id, traj, meta)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        written += table.num_rows

    if writer is None:
        raise RuntimeError("No rows written. Dataset may be empty.")

    writer.close()
    print(
        f"Exported {min(max_traj, len(trajs))} trajectories "
        f"({written} rows) from {kind} dataset to {output_path}"
    )


if __name__ == "__main__":
    main()
