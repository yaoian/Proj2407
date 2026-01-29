"""
单条轨迹可视化 Demo。

- 从 `Dataset/test_*.pth` 里随机抽一条轨迹
- 按 `--keep-rate` 生成稀疏观测点（其余视为缺失）
- 运行 DDIM 反推恢复轨迹
- 输出对比图（Dense/Sparse/Recovered）以及可选 CSV（Dense + Sparse 标记）
"""

import argparse
import os
import random
import sys
import csv
import math
import json
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Configs import (
    Trace as DefaultTrace,
    Trace_args as default_trace_args,
    Linkage as DefaultLinkage,
    link_args as default_link_args,
    dataset_name,
    diffusion_args,
)
from DDM import DDIM
from Dataset.DatasetApartments import DatasetApartments
from Dataset.DatasetTaxi import TaxiDataset
from Models import (
    Embedder,
    Trace_MultiSeq_Add,
    Trace_MultiSeq_Add_Linkage,
    Trace_MultiSeq_Cat,
    Trace_MultiSeq_Cat_Linkage,
    Trace_MultiSeq_CA,
    Trace_MultiSeq_CA_Linkage,
    Trace_MultiVec_Add,
    Trace_MultiVec_Add_Linkage,
    Trace_Seq_Cat,
    Trace_Seq_Cat_Linkage,
)
from device_utils import get_default_device

MODEL_REGISTRY = {
    "Trace_MultiSeq_Add": (Trace_MultiSeq_Add, Trace_MultiSeq_Add_Linkage),
    "Trace_MultiSeq_Cat": (Trace_MultiSeq_Cat, Trace_MultiSeq_Cat_Linkage),
    "Trace_MultiSeq_CA": (Trace_MultiSeq_CA, Trace_MultiSeq_CA_Linkage),
    "Trace_SingleSeq": (Trace_Seq_Cat, Trace_Seq_Cat_Linkage),
    "Trace_MultiVec_Add": (Trace_MultiVec_Add, Trace_MultiVec_Add_Linkage),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-trajectory visualization demo: Dense vs Sparse vs Recovered"
    )
    parser.add_argument("--ckpt", type=str, default=None, help="Path to a checkpoint .pth")
    parser.add_argument("--test-file", type=str, default=None, help="Path to a test .pth batch")
    parser.add_argument(
        "--keep-rate",
        type=float,
        default=0.5,
        help="Fraction of points to keep (observed ratio). Overrides --erase-rate when set.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cpu/cuda")
    parser.add_argument("--out", type=str, default="output/single_traj_demo.png", help="Output image path")
    parser.add_argument(
        "--csv-out",
        type=str,
        default="output/single_traj_demo.csv",
        help="Optional CSV output path (contains dense points + sparse subset flag)",
    )
    parser.add_argument("--show", action="store_true", help="Show the figure in a window")
    parser.add_argument("--verbose", action="store_true", help="Show recovery progress bar")
    return parser.parse_args()


def find_latest_ckpt() -> str:
    candidates = []
    runs_dir = os.path.join(PROJECT_ROOT, "Runs")
    for root, _, files in os.walk(runs_dir):
        if "best.pth" in files:
            candidates.append(os.path.join(root, "best.pth"))
    if candidates:
        return max(candidates, key=os.path.getmtime)
    for root, _, files in os.walk(runs_dir):
        if "last.pth" in files:
            candidates.append(os.path.join(root, "last.pth"))
    if candidates:
        return max(candidates, key=os.path.getmtime)
    raise FileNotFoundError("No checkpoint found under Runs/. Please pass --ckpt.")


def default_test_file() -> str:
    if dataset_name == "apartments":
        return os.path.join(PROJECT_ROOT, "Dataset/test_20240711_B100_l512_E05.pth")
        #return os.path.join(PROJECT_ROOT, f"Dataset/test_GeolifeNew_B100_l512_E05.pth")
    #return os.path.join(PROJECT_ROOT, f"Dataset/test_GeolifeNew_B100_l512_E05.pth")
    #return os.path.join(PROJECT_ROOT, f"Dataset/test_tdrive_B100_l512_E0.5.pth")
    candidates = [
        os.path.join(PROJECT_ROOT, f"Dataset/test_{dataset_name}_B100_l512_E05.pth"),
        os.path.join(PROJECT_ROOT, f"Dataset/test_{dataset_name}_B100_l512_E0.5.pth"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    dataset_dir = os.path.join(PROJECT_ROOT, "Dataset")
    for p in sorted(Path(dataset_dir).glob(f"test_{dataset_name}_B*_l512_E*.pth")):
        if p.is_file():
            return str(p)
    return candidates[0]


def build_erase_mask(valid_len: int, keep_rate: float, device: torch.device) -> torch.Tensor:
    if valid_len <= 0:
        raise ValueError("valid_len must be > 0")
    if valid_len == 1:
        return torch.zeros(1, device=device)

    if not (0.0 <= keep_rate <= 1.0):
        raise ValueError("keep_rate must be within [0, 1]")

    n_remain = int(math.ceil(valid_len * keep_rate))
    n_remain = max(2, min(valid_len, n_remain))

    if valid_len == 2 or n_remain == 2:
        remain_indices = torch.tensor([0, valid_len - 1], device=device)
    else:
        remain_indices = torch.randperm(valid_len - 2, device=device)[: n_remain - 2] + 1
        remain_indices = torch.sort(remain_indices)[0]
        remain_indices = torch.cat(
            [
                torch.tensor([0], device=device),
                remain_indices,
                torch.tensor([valid_len - 1], device=device),
            ]
        )

    mask = torch.ones(valid_len, device=device)
    mask[remain_indices] = 0
    return mask


def load_checkpoint(
    ckpt_path: str, device: torch.device, traj_len: int
) -> Tuple[torch.nn.Module, torch.nn.Module, Optional[torch.nn.Module]]:
    checkpoint = torch.load(ckpt_path, map_location=device)

    unet_state = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["unet"].items()}
    ckpt_in_c = int(unet_state["pre_embed.weight"].shape[1])

    train_cfg_path = os.path.join(os.path.dirname(ckpt_path), "train_config.json")
    train_cfg = None
    if os.path.isfile(train_cfg_path):
        try:
            with open(train_cfg_path, "r", encoding="utf-8") as f:
                train_cfg = json.load(f)
        except Exception:
            train_cfg = None

    trace_cls = DefaultTrace
    linkage_cls = DefaultLinkage
    trace_args = dict(default_trace_args)
    linkage_args = dict(default_link_args)
    if isinstance(train_cfg, dict):
        model_name = train_cfg.get("model_name")
        if isinstance(model_name, str) and model_name in MODEL_REGISTRY:
            trace_cls, linkage_cls = MODEL_REGISTRY[model_name]
            if isinstance(train_cfg.get("Trace_args"), dict):
                trace_args = dict(train_cfg["Trace_args"])
            if isinstance(train_cfg.get("link_args"), dict):
                linkage_args = dict(train_cfg["link_args"])

    trace_args["in_c"] = ckpt_in_c
    unet = trace_cls(**trace_args).to(device).eval()
    linkage_shapes = unet.getFeatureShapes(traj_len) if hasattr(unet, "getFeatureShapes") else unet.getStateShapes(traj_len)
    linkage = linkage_cls(linkage_shapes, **linkage_args).to(device).eval()

    unet.load_state_dict(unet_state)
    linkage.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in checkpoint["linkage"].items()})

    embedder = None
    if "embedder" in checkpoint:
        embed_state = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["embedder"].items()}
        embed_dim = int(embed_state["operator_embedder.weight"].shape[1])
        embedder = Embedder(embed_dim).to(device).eval()
        embedder.load_state_dict(embed_state)

    return unet, linkage, embedder


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def write_dense_and_sparse_csv(
    path: str,
    dense_xy: torch.Tensor,
    time: torch.Tensor,
    mask_1d: torch.Tensor,
) -> Tuple[int, int]:
    """
    :param dense_xy: (2, L_valid)
    :param time: (1, L_valid)
    :param mask_1d: (L_valid,) 0 for observed/kept, 1 for erased
    :return: (valid_len, observed_len)
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dense_xy = dense_xy.detach().cpu()
    time = time.detach().cpu()
    mask_1d = mask_1d.detach().cpu()

    valid_len = int(mask_1d.numel())
    observed_len = int((mask_1d <= 0.1).sum().item())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "lon",
                "lat",
                "time",
                "mask",
                "is_observed",
                "sparse_lon",
                "sparse_lat",
                "sparse_time",
            ]
        )
        for i in range(valid_len):
            lon = float(dense_xy[0, i].item())
            lat = float(dense_xy[1, i].item())
            t = float(time[0, i].item())
            mask_val = float(mask_1d[i].item())
            is_observed = 1 if mask_val <= 0.1 else 0
            if is_observed:
                sparse_lon, sparse_lat, sparse_time = lon, lat, t
            else:
                sparse_lon, sparse_lat, sparse_time = "", "", ""
            writer.writerow(
                [i, lon, lat, t, mask_val, is_observed, sparse_lon, sparse_lat, sparse_time]
            )

    return valid_len, observed_len


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else get_default_device()
    ckpt_path = resolve_path(args.ckpt) if args.ckpt else find_latest_ckpt()
    test_file = resolve_path(args.test_file) if args.test_file else default_test_file()

    if not os.path.isfile(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    batch_data = torch.load(test_file, map_location="cpu")
    if not isinstance(batch_data, tuple):
        raise TypeError(f"Unexpected test batch type: {type(batch_data)}")

    # apartments: (loc_0, loc_T, loc_guess, loc_mean, meta, time, mask, bool_mask, query_len, observe_len)
    # taxi:      (loc_0, loc_T, loc_guess, time, mask, bool_mask, query_len, observe_len)
    if len(batch_data) == 10:
        loc_0_all, _, _, loc_mean_all, meta_all, time_all, raw_mask_all, _, _, _ = batch_data
    elif len(batch_data) == 8:
        loc_0_all, _, _, time_all, raw_mask_all, _, _, _ = batch_data
        loc_mean_all = torch.zeros(loc_0_all.shape[0], 2, 1)
        meta_all = None
    else:
        raise ValueError(f"Unexpected test batch tuple length: {len(batch_data)}")

    batch_size = loc_0_all.shape[0]
    idx = torch.randint(0, batch_size, (1,)).item()

    loc_0 = loc_0_all[idx : idx + 1].to(device)
    time = time_all[idx : idx + 1].to(device)
    loc_mean = loc_mean_all[idx : idx + 1].to(device)
    meta = meta_all[idx : idx + 1].to(device) if meta_all is not None else None

    raw_mask = raw_mask_all[idx]
    if raw_mask.ndim == 2:
        raw_mask = raw_mask[0]
    valid_positions = raw_mask >= 0
    valid_len = int(valid_positions.sum().item())
    if valid_len <= 1:
        raise ValueError("Valid trajectory length is too short for visualization.")

    keep_rate = args.keep_rate
    if not (0.0 <= keep_rate <= 1.0):
        raise ValueError("--keep-rate must be within [0, 1]")

    traj_len = loc_0.shape[-1]
    traj_valid = torch.cat([loc_0[:, :, :valid_len], time[:, :, :valid_len]], dim=1)
    mask_valid = build_erase_mask(valid_len, keep_rate, device)

    loc_guess_valid = TaxiDataset.guessTraj(traj_valid[0], mask_valid)

    loc_guess = torch.zeros(1, 2, traj_len, device=device)
    loc_guess[:, :, :valid_len] = loc_guess_valid

    mask = torch.full((1, 1, traj_len), -1.0, device=device)
    mask[:, :, :valid_len] = mask_valid.view(1, 1, -1)

    loc_T = loc_0.clone()
    mask_2d = (mask > 0.1).repeat(1, 2, 1)
    loc_T[mask_2d] = torch.randn_like(loc_T[mask_2d])

    unet, linkage, embedder = load_checkpoint(ckpt_path, device, traj_len)
    diff_manager = DDIM(**diffusion_args, device=device)

    s_T = []
    for shape in unet.getStateShapes(traj_len):
        s_T.append(torch.zeros(1, *shape, dtype=torch.float32, device=device))

    with torch.no_grad():
        in_c = int(unet.pre_embed.in_channels)
        extra_c = in_c - 6
        if extra_c > 0:
            if meta is None:
                meta = torch.zeros(1, 4, traj_len, dtype=torch.long, device=device)
            if loc_mean is None:
                loc_mean = torch.zeros(1, 2, 1, dtype=torch.float32, device=device)

            if embedder is not None:
                embed = embedder(meta, loc_mean)
            else:
                embed = torch.zeros(1, extra_c, traj_len, dtype=torch.float32, device=device)
            loc_rec = diff_manager.diffusionBackwardWithE(
                unet, linkage, embed, loc_T, s_T, time, loc_guess, mask, verbose=args.verbose
            )
        else:
            loc_rec = diff_manager.diffusionBackward(
                unet, linkage, loc_T, s_T, time, loc_guess, mask, verbose=args.verbose
            )

    dense = (loc_0 + loc_mean)[0]
    recovered = (loc_rec + loc_mean)[0]
    valid_positions = mask[0, 0] >= 0
    observed_positions = (mask[0, 0] <= 0.1) & valid_positions

    if args.csv_out:
        csv_path = resolve_path(args.csv_out)
        dense_valid = dense[:, :valid_len]
        time_valid = time[0, :, :valid_len]
        mask_valid_1d = mask[0, 0, :valid_len]
        csv_valid_len, csv_observed_len = write_dense_and_sparse_csv(
            csv_path, dense_valid, time_valid, mask_valid_1d
        )
        print(f"saved: {csv_path}")
        effective_keep_rate = csv_observed_len / max(1, csv_valid_len)
        print(
            "csv_valid_len={} csv_observed_len={} keep_rate~{:.6f}".format(
                csv_valid_len,
                csv_observed_len,
                effective_keep_rate,
            )
        )

    dense_xy = dense[:, valid_positions].detach().cpu().numpy()
    sparse_xy = dense[:, observed_positions].detach().cpu().numpy()
    recovered_xy = recovered[:, valid_positions].detach().cpu().numpy()

    all_x = np.concatenate([dense_xy[0], sparse_xy[0], recovered_xy[0]])
    all_y = np.concatenate([dense_xy[1], sparse_xy[1], recovered_xy[1]])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    pad_x = max(1e-6, (x_max - x_min) * 0.02)
    pad_y = max(1e-6, (y_max - y_min) * 0.02)

    def format_axes(ax, title: str) -> None:
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax_all = axes[0, 0]
    ax_all.plot(dense_xy[0], dense_xy[1], color="tab:blue", linewidth=1, label="Dense")
    ax_all.scatter(sparse_xy[0], sparse_xy[1], s=12, color="tab:orange", label="Sparse", zorder=3)
    ax_all.plot(recovered_xy[0], recovered_xy[1], color="tab:red", linewidth=1, label="Recovered")
    title_parts = [f"All | {dataset_name} | idx={idx}", f"keep_rate={keep_rate:.4f}"]
    format_axes(ax_all, " | ".join(title_parts))
    ax_all.legend()

    ax_dense = axes[0, 1]
    ax_dense.plot(dense_xy[0], dense_xy[1], color="tab:blue", linewidth=1)
    format_axes(ax_dense, "Dense")

    ax_sparse = axes[1, 0]
    ax_sparse.scatter(sparse_xy[0], sparse_xy[1], s=12, color="tab:orange", zorder=3)
    format_axes(ax_sparse, "Sparse")

    ax_rec = axes[1, 1]
    ax_rec.plot(recovered_xy[0], recovered_xy[1], color="tab:red", linewidth=1)
    format_axes(ax_rec, "Recovered")

    fig.tight_layout()

    out_path = resolve_path(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")
    print(f"dataset={dataset_name} ckpt={ckpt_path} valid_len={valid_len}")

    erased_positions = mask[0, 0] > 0.1
    if erased_positions.any():
        mse = torch.nn.functional.mse_loss(
            loc_rec[0, :, erased_positions], loc_0[0, :, erased_positions]
        ) * 1000
        print(f"recovery_mse_x1000={mse.item():.6f}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
