import torch


def get_default_device() -> torch.device:
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

