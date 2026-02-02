from __future__ import annotations

import torch


def guess_traj_time_interp(traj_0: torch.Tensor, erase_mask: torch.Tensor) -> torch.Tensor:
    """
    Obtain the guessed trajectory from the original trajectory and the erase mask
    :param traj_0: (3, L)
    :param erase_mask: (L,) float, 1=erase, 0=observe
    :return: guessed locations: (2, L)
    """
    if traj_0.numel() == 0:
        return torch.zeros((2, 0), dtype=traj_0.dtype, device=traj_0.device)

    boolean_mask = erase_mask > 0.1  # (L,)
    if not torch.any(boolean_mask):
        return traj_0[:2].clone()

    erased_subtraj = traj_0[:, boolean_mask]  # (3, L_erased)
    remain_subtraj = traj_0[:, ~boolean_mask]  # (3, L_remain)

    L_remain = remain_subtraj.shape[-1]
    if L_remain == 0:
        return torch.zeros_like(traj_0[:2])

    time_interp = erased_subtraj[2]  # (L_erased)
    time_remain = remain_subtraj[2]  # (L_remain)
    ids_right = torch.searchsorted(time_remain, time_interp).to(torch.long)  # (L_erased)
    ids_left = ids_right - 1  # (L_erased)

    ids_left = torch.clamp(ids_left, 0, L_remain - 1)
    ids_right = torch.clamp(ids_right, 0, L_remain - 1)

    traj_left = remain_subtraj[:, ids_left]
    traj_right = remain_subtraj[:, ids_right]

    denom = traj_right[2] - traj_left[2]
    ratio = (time_interp - traj_left[2]) / denom  # (L_erased)
    ratio = torch.where(torch.isfinite(ratio), ratio, torch.zeros_like(ratio))

    erased_loc_guess = traj_left[:2] * (1 - ratio) + traj_right[:2] * ratio  # (2, L_erased)

    loc_guess = traj_0[:2].clone()  # (2, L)
    loc_guess[:, boolean_mask] = erased_loc_guess

    nan_mask = torch.isnan(loc_guess)
    if torch.any(nan_mask):
        loc_guess[nan_mask] = torch.zeros_like(loc_guess[nan_mask])

    return loc_guess
