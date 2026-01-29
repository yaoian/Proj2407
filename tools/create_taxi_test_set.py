"""
从任意 TaxiDataset 缓存（list[(traj, ...)]）生成 eval.py 可直接读取的 test batch `.pth`。

生成的 tuple 字段顺序（与 `eval.py` Xian/Chengdu 分支一致）：
0 loc_0:      (B, 2, L)
1 loc_T:      (B, 2, L)  # query 部分加高斯噪声，其余保持原值
2 loc_guess:  (B, 2, L)  # 由时间插值得到的初始猜测
3 time:       (B, 1, L)
4 mask:       (B, 1, L)  # 1=erase, 0=observe, -1=padding
5 bool_mask:  (B, 2, L)  # query 部分的布尔 mask（repeat 到 2 通道）
6 query_len:  (B,)
7 observe_len:(B,)
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Dataset.DatasetTaxi import TaxiDataset
from device_utils import get_default_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a saved taxi test batch tuple for eval.py")
    p.add_argument("--cache", type=str, required=True, help="Path to taxi cache .pth, e.g. Dataset/TDrive_nov_cache.pth")
    p.add_argument("--out", type=str, required=True, help="Output test batch .pth path")
    p.add_argument("--batch", type=int, default=100, help="Batch size B")
    p.add_argument("--traj-len", type=int, default=512, help="Padded trajectory length L")
    p.add_argument("--erase-rate", type=float, default=0.5, help="Erase rate (query ratio), e.g. 0.5")
    p.add_argument("--device", type=str, default="", help="Override device, e.g. cuda/cpu; empty uses auto")
    p.add_argument(
        "--sample-with-replacement",
        action="store_true",
        help="If cache has fewer trajectories than --batch, sample indices with replacement instead of failing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device.strip() or get_default_device()
    B = int(args.batch)
    L = int(args.traj_len)
    erase_rate = float(args.erase_rate)

    dataset = TaxiDataset(max_len=L, load_path=args.cache)
    dataset.device = device
    dataset.resetSampleLength(L)
    dataset.resetEraseRate(erase_rate)

    n_trajs = len(dataset)
    if n_trajs <= 0:
        raise ValueError(f"Empty cache: {args.cache}")
    if n_trajs < B and not args.sample_with_replacement:
        raise ValueError(
            f"Cache has only {n_trajs} trajectories, but --batch={B}. "
            "Reduce --batch, or regenerate cache with more trajectories, "
            "or add --sample-with-replacement."
        )

    query_len = int(L * erase_rate)
    observe_len = L - query_len
    batch_query_len = torch.ones(B, device=device, dtype=torch.long) * query_len
    batch_observe_len = torch.ones(B, device=device, dtype=torch.long) * observe_len

    batch_loc_0 = torch.zeros(B, 2, L, device=device, dtype=torch.float32)
    batch_loc_guess = torch.zeros(B, 2, L, device=device, dtype=torch.float32)
    batch_time = torch.zeros(B, 1, L, device=device, dtype=torch.float32)
    batch_mask = torch.zeros(B, 1, L, device=device, dtype=torch.float32)

    for j in range(B):
        idx = j if n_trajs >= B else random.randrange(n_trajs)
        traj_0, mask, loc_guess = dataset[idx]
        traj_0 = traj_0.to(device)
        mask = mask.to(device)
        loc_guess = loc_guess.to(device)

        batch_loc_0[j] = traj_0[:2]
        batch_time[j] = traj_0[2:]
        batch_loc_guess[j] = loc_guess
        batch_mask[j] = mask.reshape(1, -1)

    batch_bool_mask = (batch_mask > 0.1).repeat(1, 2, 1)  # (B, 2, L)
    batch_loc_T = batch_loc_0.clone()
    batch_loc_T[batch_bool_mask] = torch.randn_like(batch_loc_T[batch_bool_mask])

    batch_data = (
        batch_loc_0,
        batch_loc_T,
        batch_loc_guess,
        batch_time,
        batch_mask,
        batch_bool_mask,
        batch_query_len,
        batch_observe_len,
    )
    torch.save(batch_data, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
