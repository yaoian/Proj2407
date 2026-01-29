"""
查看 `.pth` 文件内容并输出一张示意图。

- 若是 apartments 数据集 dict：读取 `trajs[idx]` 并绘制轨迹
- 若是 test batch tuple：绘制 Dense / Guess(插值) / Observed(稀疏) 以便检查采样效果
"""

import argparse
import os
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a .pth dataset file (either full dataset dict or a saved test batch tuple)."
    )
    parser.add_argument(
        "--path",
        type=str,
        default="Dataset/test_20240711_B100_l512_E05.pth",
        help="Path to .pth file",
    )
    parser.add_argument("--idx", type=int, default=0, help="Trajectory/sample index to plot")
    parser.add_argument(
        "--out",
        type=str,
        default="output/show_dataset.png",
        help="Output image path",
    )
    parser.add_argument("--show", action="store_true", help="Show figure window")
    return parser.parse_args()


def _ensure_out_dir(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def _plot_apartment_dataset_dict(data: Dict[str, Any], idx: int) -> None:
    trajs = data["trajs"]
    traj = trajs[idx].float()
    lon = traj[0].cpu().numpy()
    lat = traj[1].cpu().numpy()
    plt.plot(lon, lat, linewidth=1)
    plt.axis("equal")
    plt.title(f"apartment_dataset | traj={idx} | len={traj.shape[1]}")


def _plot_test_batch_tuple(data: Tuple[Any, ...], idx: int) -> None:
    if len(data) < 7:
        raise ValueError(f"Unexpected test batch tuple length: {len(data)}")

    # Common order used by eval.py / CreateTestSet.py:
    # loc_0, loc_T, loc_guess, loc_mean, meta, time, mask, ...
    loc_0 = data[0]  # (B, 2, L)
    loc_guess = data[2]  # (B, 2, L)
    loc_mean = data[3]  # (B, 2, 1) or scalar/zeros
    mask = data[6]  # (B, 1, L)

    if idx < 0 or idx >= loc_0.shape[0]:
        raise IndexError(f"--idx out of range: idx={idx}, batch_size={loc_0.shape[0]}")

    loc_mean_i = loc_mean[idx] if isinstance(loc_mean, torch.Tensor) else 0.0
    dense = (loc_0[idx] + loc_mean_i).detach()
    guess = (loc_guess[idx] + loc_mean_i).detach()

    mask_i = mask[idx, 0].detach()
    valid_positions = mask_i >= 0
    observed_positions = (mask_i <= 0.1) & valid_positions

    dense_xy = dense[:, valid_positions].cpu().numpy()
    guess_xy = guess[:, valid_positions].cpu().numpy()
    obs_xy = dense[:, observed_positions].cpu().numpy()

    plt.plot(dense_xy[0], dense_xy[1], color="tab:blue", linewidth=1, label="Dense (gt)")
    plt.plot(guess_xy[0], guess_xy[1], color="tab:gray", linewidth=1, linestyle="--", label="Guess (interp)")
    if obs_xy.shape[1] > 0:
        plt.scatter(obs_xy[0], obs_xy[1], s=10, color="tab:orange", label="Observed", zorder=3)
    plt.axis("equal")
    plt.title(f"test_batch | sample={idx} | valid_len={int(valid_positions.sum())}")
    plt.legend()


def main() -> None:
    args = parse_args()
    data = torch.load(args.path, map_location="cpu")

    print(f"path={args.path}")
    print(f"type={type(data)}")

    plt.figure(figsize=(7, 7))
    if isinstance(data, dict):
        print(f"keys={list(data.keys())}")
        _plot_apartment_dataset_dict(data, args.idx)
    elif isinstance(data, tuple):
        shapes = [
            tuple(x.shape) if isinstance(x, torch.Tensor) else type(x) for x in data
        ]
        print(f"tuple_len={len(data)}")
        print(f"tuple_shapes={shapes}")
        _plot_test_batch_tuple(data, args.idx)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    _ensure_out_dir(args.out)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved: {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
