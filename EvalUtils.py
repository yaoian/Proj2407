import ctypes
import os
from typing import Tuple

import numpy as np
import torch


def _load_ndtw_lib() -> ctypes.CDLL:
    lib_name = "NDTW.dll" if os.name == "nt" else "NDTW.so"
    lib_path = os.path.join(os.path.dirname(__file__), lib_name)
    if not os.path.isfile(lib_path):
        raise FileNotFoundError(
            f"Native DTW library not found: {lib_path}. Expected {lib_name} in repo root."
        )
    lib = ctypes.cdll.LoadLibrary(lib_path)
    lib.NDTW.restype = ctypes.c_float
    return lib


eval_c_lib = _load_ndtw_lib()

Tensor = torch.Tensor



def JSD(original_data: Tensor, generated_data: Tensor, n_grids: int = 64, normalize: bool = True):
    """
    Jensen–Shannon Divergence between 2D point distributions.

    Expected input shapes:
    - (B, 2, L) tensors (common in this repo), or
    - (N, 2) tensors (flattened points)

    Note: Padding/filtering should be done by the caller if needed.
    """

    def _to_points(x: Tensor) -> np.ndarray:
        x = x.detach().to("cpu")
        if x.ndim == 3:
            x = x.transpose(1, 2).reshape(-1, 2)
        elif x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"Unexpected input shape for JSD: {tuple(x.shape)}")
        return x.numpy()

    orig = _to_points(original_data)
    gen = _to_points(generated_data)

    all_points = np.concatenate([orig, gen], axis=0)
    min_lon = float(all_points[:, 0].min())
    max_lon = float(all_points[:, 0].max())
    min_lat = float(all_points[:, 1].min())
    max_lat = float(all_points[:, 1].max())

    eps = 1e-12
    lng_interval = (max_lon - min_lon) / n_grids
    lat_interval = (max_lat - min_lat) / n_grids
    lng_interval = max(lng_interval, eps)
    lat_interval = max(lat_interval, eps)

    def _hist(points: np.ndarray) -> np.ndarray:
        lng_idx = np.clip(((points[:, 0] - min_lon) / lng_interval).astype(np.int64), 0, n_grids - 1)
        lat_idx = np.clip(((points[:, 1] - min_lat) / lat_interval).astype(np.int64), 0, n_grids - 1)
        flat = lng_idx * n_grids + lat_idx
        counts = np.bincount(flat, minlength=n_grids * n_grids).astype(np.float64)
        counts = counts.reshape(n_grids, n_grids)
        if normalize:
            counts = (counts + 1.0) / float(counts.sum() + n_grids * n_grids)
        return counts

    P = _hist(orig)
    Q = _hist(gen)
    M = 0.5 * (P + Q)

    def _kl(A: np.ndarray, B: np.ndarray) -> float:
        return float(np.sum(A * np.log((A + eps) / (B + eps))))

    return 0.5 * (_kl(P, M) + _kl(Q, M))


def NDTW(target_traj, compare_traj):
    """
    This function calculates the Dynamic Time Warping (DTW) distance between two trajectories.
    :param target_traj: trajectory 1 (3, N)
    :param compare_traj: trajectory 2 (3, M)
    :return: DTW distance
    """
    n = target_traj.shape[1]
    m = compare_traj.shape[1]
    dtw = torch.zeros((n + 1, m + 1))
    dtw[1:, 0] = torch.inf
    dtw[0, 1:] = torch.inf
    dtw[0, 0] = 0

    lng_lat_A = target_traj[:2, :].unsqueeze(2)
    lng_lat_B = compare_traj[:2, :].unsqueeze(1)
    squared_dist = torch.sum((lng_lat_A - lng_lat_B) ** 2, dim=0)
    dist_mat = torch.sqrt(squared_dist)

    dist_mat_ptr = dist_mat.cpu().numpy().astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    dtw_ptr = dtw.cpu().numpy().astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    return float(eval_c_lib.NDTW(dist_mat_ptr, dtw_ptr, n, m))
