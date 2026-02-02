import torch
import torch.utils.data as data
from typing import *
import random
from rich.status import Status

from device_utils import get_default_device
from Priors import guess_traj_time_interp

class TaxiDataset(data.Dataset):

    def __init__(self, max_len: int, load_path: str):
        self.device = get_default_device()

        with Status(f'Loading {load_path} from disk...'):
            dataset_part = torch.load(load_path, map_location="cpu")
            # Keep trajectories on CPU to avoid OOM for large datasets; move to device in __getitem__.
            self.trajs = [sample[0].to(torch.float32) for sample in dataset_part]

        self.max_length = max_len
        self.sample_length = max_len
        self.erase_rate = 0.0
        self.shift_rate = 0.0


    def resetSampleLength(self, length: int):
        if length > self.max_length:
            length = self.max_length
        elif length < 1:
            length = 1
        self.sample_length = length


    def resetEraseRate(self, elim_rate: float):
        self.erase_rate = min(1.0, max(0.0, elim_rate))


    def resetShiftRate(self, shift_rate: float):
        self.shift_rate = min(1.0, max(0.0, shift_rate))

    def state_dict(self) -> dict:
        return {
            "max_length": int(self.max_length),
            "sample_length": int(self.sample_length),
            "erase_rate": float(self.erase_rate),
            "shift_rate": float(self.shift_rate),
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if "sample_length" in state:
            self.resetSampleLength(int(state["sample_length"]))
        if "erase_rate" in state:
            self.resetEraseRate(float(state["erase_rate"]))
        if "shift_rate" in state:
            self.resetShiftRate(float(state["shift_rate"]))

    @staticmethod
    # @torch.compile
    def guessTraj(traj_0, erase_mask):
        """
        Obtain the guessed trajectory from the original trajectory and the erase mask
        :param traj_0: (3, L)
        :param mask:  (L,)
        :return: guessed locations: (2, L)
        """
        return guess_traj_time_interp(traj_0, erase_mask)



    def __len__(self):
        return len(self.trajs)


    def __getitem__(self, index: Any) -> Any:
        """
        :param index: The index of the trajectory in dataset_part
        :return: lon_lat: (2, N), attr: (3,), times: (N,)
        """
        traj = self.trajs[index]
        max_start = max(0, traj.shape[1] - self.sample_length)
        sample_start = random.randint(0, max_start)
        traj = traj[:, sample_start:sample_start + self.sample_length].to(self.device)

        actual_sample_length = traj.shape[1]
        if actual_sample_length < 2:
            raise ValueError(f"Trajectory too short: len={actual_sample_length}")

        n_remain = actual_sample_length - int(actual_sample_length * self.erase_rate)
        # select n_remain indices to remain, sorted, and exclude the first and last point
        remain_indices = torch.randperm(actual_sample_length - 2)[:n_remain - 2].to(self.device) + 1
        remain_indices = torch.sort(remain_indices)[0]
        # add firsst and the last point
        remain_indices = torch.cat(
            [
                torch.tensor([0], device=self.device),
                remain_indices,
                torch.tensor([actual_sample_length - 1], device=self.device),
            ]
        )

        # binary_mask is 0 means the broken traj, 1 means erased
        mask = torch.ones(actual_sample_length, dtype=torch.float32, device=self.device)
        mask[remain_indices] = 0

        loc_guess = self.guessTraj(traj, mask)

        pad_size = self.max_length - actual_sample_length
        mask = torch.nn.functional.pad(mask, (0, pad_size), mode="constant", value=-1)
        traj = torch.nn.functional.pad(traj, (0, pad_size))
        loc_guess = torch.nn.functional.pad(loc_guess, (0, pad_size))

        return traj, mask, loc_guess

def collate_fn(batch):
    """
    :param batch: list of (traj, mask, loc_guess)
    :return: traj: (B, 3, L), mask: (B, L), loc_guess: (B, 2, L)
    """
    traj, mask, loc_guess = zip(*batch)
    traj = torch.stack(traj, dim=0)
    mask = torch.stack(mask, dim=0)
    loc_guess = torch.stack(loc_guess, dim=0)

    return traj, mask, loc_guess



if __name__ == "__main__":
    from Configs import dataset_args
    dataset = TaxiDataset(**dataset_args)
    for i in range(10):
        test = dataset[i]
