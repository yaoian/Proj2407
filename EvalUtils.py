import ctypes
import os
from typing import Optional, Tuple

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


_eval_c_lib: Optional[ctypes.CDLL] = None
_eval_c_lib_error: Optional[str] = None
try:
    _eval_c_lib = _load_ndtw_lib()
except Exception as e:
    _eval_c_lib = None
    _eval_c_lib_error = f"{type(e).__name__}: {e}"

NDTW_BACKEND = "native" if _eval_c_lib is not None else "python"
NDTW_NATIVE_ERROR = _eval_c_lib_error

Tensor = torch.Tensor



def JSD(
    original_data: Tensor,
    generated_data: Tensor,
    n_grids: int = 64,
    normalize: bool = True,
    bounds: Optional[Tuple[float, float, float, float]] = None,
):
    """
    Jensen–Shannon Divergence between 2D point distributions.

    Expected input shapes:
    - (B, 2, L) tensors (common in this repo), or
    - (N, 2) tensors (flattened points)

    Note: Padding/filtering should be done by the caller if needed.
    If `bounds` is provided, it must be (min_x, max_x, min_y, max_y) and will be used
    to build the shared discretization grid (helps align experiments across models).
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

    if bounds is None:
        all_points = np.concatenate([orig, gen], axis=0)
        min_lon = float(all_points[:, 0].min())
        max_lon = float(all_points[:, 0].max())
        min_lat = float(all_points[:, 1].min())
        max_lat = float(all_points[:, 1].max())
    else:
        min_lon, max_lon, min_lat, max_lat = bounds

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


def _ndtw_python(target_traj: Tensor, compare_traj: Tensor) -> float:
    n = int(target_traj.shape[1])
    m = int(compare_traj.shape[1])
    if n <= 0 or m <= 0:
        return 0.0

    lng_lat_A = target_traj[:2, :].T.contiguous()  # (n, 2)
    lng_lat_B = compare_traj[:2, :].T.contiguous()  # (m, 2)
    dist_mat = torch.cdist(lng_lat_A, lng_lat_B, p=2).detach().to("cpu").numpy().astype(np.float64)  # (n, m)

    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        prev_i = dtw[i - 1]
        cur_i = dtw[i]
        dist_i = dist_mat[i - 1]
        for j in range(1, m + 1):
            cost = dist_i[j - 1]
            cur_i[j] = cost + min(cur_i[j - 1], prev_i[j], prev_i[j - 1])

    # normalize by sequence length to keep scale comparable across different lengths
    return float(dtw[n, m] / max(1, max(n, m)))


def NDTW(target_traj, compare_traj):
    """
    This function calculates the Dynamic Time Warping (DTW) distance between two trajectories.
    :param target_traj: trajectory 1 (3, N)
    :param compare_traj: trajectory 2 (3, M)
    :return: DTW distance
    """
    if _eval_c_lib is None:
        return _ndtw_python(target_traj, compare_traj)

    n = int(target_traj.shape[1])
    m = int(compare_traj.shape[1])
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

    return float(_eval_c_lib.NDTW(dist_mat_ptr, dtw_ptr, n, m))


# Backward/typo compatibility (some notes/scripts may call it NTDW).
NTDW = NDTW
