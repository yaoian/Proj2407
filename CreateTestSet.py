from Configs import *
from Dataset.DatasetApartments import DatasetApartments
from device_utils import get_default_device

B = 100
erase_rate = 0.5
l = 512
L = TRAJ_LEN

device = get_default_device()

dataset = DatasetApartments(**dataset_args)
dataset.resetSampleLength(l)
dataset.resetEraseRate(erase_rate)


query_len = int(l * erase_rate)
batch_query_len = torch.ones(B, device=device, dtype=torch.long) * query_len
observe_len = l - query_len
batch_observe_len = torch.ones(B, device=device, dtype=torch.long) * observe_len

def getData():
    batch_loc_0 = torch.zeros(B, 2, L, device=device, dtype=torch.float32)
    batch_loc_guess = torch.zeros(B, 2, L, device=device, dtype=torch.float32)
    batch_time = torch.zeros(B, 1, L, device=device, dtype=torch.float32)
    batch_mask = torch.zeros(B, 1, L, device=device, dtype=torch.float32)
    batch_loc_mean = torch.zeros(B, 2, 1, device=device, dtype=torch.float32)
    batch_meta = torch.zeros(B, 4, L, device=device, dtype=torch.long)
    for j in range(B):
        # traj_0, _, _, _, _, mask, _, loc_guess = dataset[j]
        traj_0, mask, loc_guess, loc_mean, meta = dataset[j]

        batch_loc_0[j] = traj_0[:2]
        batch_time[j] = traj_0[2:]
        batch_loc_guess[j] = loc_guess
        batch_mask[j] = mask
        batch_loc_mean[j] = loc_mean
        batch_meta[j] = meta

    batch_bool_mask = (batch_mask > 0.1).repeat(1, 2, 1)    # (B, 2, 512)

    batch_loc_T = batch_loc_0.clone()
    batch_loc_T[batch_bool_mask] = torch.randn_like(batch_loc_T[batch_bool_mask])

    return batch_loc_0, batch_loc_T, batch_loc_guess, batch_loc_mean, batch_meta, batch_time, batch_mask, batch_bool_mask, batch_query_len, batch_observe_len

batch_data = getData()
torch.save(batch_data, "Dataset/test_20240711_B100_l512_E05.pth")
