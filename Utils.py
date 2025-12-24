from Configs import *
import atexit
import os
import threading


_save_lock = threading.Lock()
_save_threads: dict[str, threading.Thread] = {}


def _join_save_threads() -> None:
    with _save_lock:
        threads = list(_save_threads.values())
    for t in threads:
        if t.is_alive():
            t.join()


atexit.register(_join_save_threads)


def saveModel(path: str, async_write: bool = False, **models) -> None:
    models = {name: model for name, model in models.items() if model is not None}

    def _do_save() -> None:
        payload = {}
        for name, model in models.items():
            state = model.state_dict()
            payload[name] = {k: v.detach().to("cpu") for k, v in state.items()}
        tmp_path = f"{path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    if not async_write:
        _do_save()
        return

    with _save_lock:
        prev = _save_threads.get(path)
        if prev is not None and prev.is_alive():
            return
        t = threading.Thread(target=_do_save, name=f"saveModel:{os.path.basename(path)}")
        _save_threads[path] = t
        t.start()


def loadModel(path: str, **models) -> List[Module]:
    checkpoint = torch.load(path)
    for name, model in models.items():
        if model is None:
            continue
        if name not in checkpoint:
            continue
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in checkpoint[name].items()})
    return list(models.values())


class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.avg = 0
        self.size = 0

    def __lshift__(self, number: float) -> None:
        if self.size < self.window_size:
            moving_sum = self.avg * self.size + number
            self.size += 1
        else:
            moving_sum = (self.avg * self.size - self.avg + number)
        self.avg = moving_sum / self.size

    def __float__(self) -> float:
        return self.avg

    def __str__(self) -> str:
        return str(self.avg)

    def __repr__(self) -> str:
        return str(self.avg)

    def __format__(self, format_spec: str) -> str:
        return self.avg.__format__(format_spec)

class MaskedMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = torch.nn.MSELoss()

    def forward(self, output, eps, mask):
        return torch.mean(torch.stack([self.mse(output[b][mask[b, :2, :]], eps[b].flatten()) for b in range(batch_size)]))
