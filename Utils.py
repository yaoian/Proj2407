from Configs import *
import atexit
import os
import threading
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch


_save_lock = threading.Lock()
_save_threads: dict[str, threading.Thread] = {}


def _join_save_threads() -> None:
    with _save_lock:
        threads = list(_save_threads.values())
    for t in threads:
        if t.is_alive():
            t.join()


atexit.register(_join_save_threads)

_FULL_CKPT_FORMAT = "trace_full_checkpoint_v1"


def _module_state_to_cpu(module: Module) -> Dict[str, torch.Tensor]:
    state = module.state_dict()
    return {k: v.detach().to("cpu") for k, v in state.items()}


def _checkpoint_models_view(checkpoint: Any) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
    """
    Backward/forward compatible view:
    - legacy: {"unet": state_dict, "linkage": state_dict, ...}
    - full:   {"format":..., "models": {...}, ...}
    """
    if not isinstance(checkpoint, dict):
        return None
    if "models" in checkpoint and isinstance(checkpoint["models"], dict):
        return checkpoint["models"]
    # legacy
    if any(k in checkpoint for k in ("unet", "linkage", "embedder")):
        return checkpoint  # type: ignore[return-value]
    return None


def _recursive_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _recursive_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_recursive_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_recursive_to_device(v, device) for v in obj)
    return obj


def _optimizer_state_to_cpu(state: Dict[str, Any]) -> Dict[str, Any]:
    # Optimizer state_dict contains tensors in nested dicts/lists; make it device-agnostic.
    return _recursive_to_device(state, torch.device("cpu"))


def _capture_rng_state() -> Dict[str, Any]:
    rng: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            rng["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            # In some environments CUDA is partially available; don't fail checkpointing.
            rng["torch_cuda_all"] = None
    return rng


def _restore_rng_state(state: Dict[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    try:
        if "python_random" in state:
            random.setstate(state["python_random"])
        if "numpy_random" in state:
            np.random.set_state(state["numpy_random"])
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda_all") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    except Exception:
        # RNG restoration is best-effort; training can still proceed.
        return


def saveCheckpointFull(
    path: str,
    *,
    async_write: bool = False,
    models: Dict[str, Optional[Module]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[Any] = None,
    train_state: Optional[Dict[str, Any]] = None,
    dataset_state: Optional[Dict[str, Any]] = None,
    batch_manager_state: Optional[Dict[str, Any]] = None,
    run_config: Optional[Dict[str, Any]] = None,
    include_rng_state: bool = True,
) -> None:
    models = {name: model for name, model in models.items() if model is not None}

    def _do_save() -> None:
        payload: Dict[str, Any] = {
            "format": _FULL_CKPT_FORMAT,
            "models": {name: _module_state_to_cpu(model) for name, model in models.items()},
            "train_state": train_state or {},
        }
        if optimizer is not None:
            payload["optimizer"] = _optimizer_state_to_cpu(optimizer.state_dict())
        if lr_scheduler is not None:
            try:
                payload["lr_scheduler"] = lr_scheduler.state_dict()
            except Exception:
                payload["lr_scheduler"] = None
        if dataset_state is not None:
            payload["dataset_state"] = dataset_state
        if batch_manager_state is not None:
            payload["batch_manager_state"] = batch_manager_state
        if run_config is not None:
            payload["run_config"] = run_config
        if include_rng_state:
            payload["rng_state"] = _capture_rng_state()

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
        t = threading.Thread(target=_do_save, name=f"saveCheckpointFull:{os.path.basename(path)}")
        _save_threads[path] = t
        t.start()


def loadCheckpointFull(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != _FULL_CKPT_FORMAT:
        raise ValueError(f"Not a full checkpoint: {path}")
    return checkpoint


def saveModel(path: str, async_write: bool = False, **models) -> None:
    models = {name: model for name, model in models.items() if model is not None}

    def _do_save() -> None:
        payload = {}
        for name, model in models.items():
            payload[name] = _module_state_to_cpu(model)
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
    checkpoint = torch.load(path, map_location="cpu")
    models_ckpt = _checkpoint_models_view(checkpoint)
    if models_ckpt is None:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    for name, model in models.items():
        if model is None:
            continue
        if name not in models_ckpt:
            continue
        model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in models_ckpt[name].items()})
    return list(models.values())


def restore_training_extras_from_full_checkpoint(
    checkpoint: Dict[str, Any],
    *,
    optimizer: Optional[torch.optim.Optimizer],
    lr_scheduler: Optional[Any],
    device: torch.device,
    restore_rng_state: bool = True,
) -> Dict[str, Any]:
    """
    Restore optimizer/scheduler and RNG state. Returns train_state dict (may be empty).
    Best-effort: if optimizer param groups mismatch (e.g. finetune/freezing), it will skip.
    """
    if restore_rng_state and isinstance(checkpoint.get("rng_state"), dict):
        _restore_rng_state(checkpoint["rng_state"])

    if optimizer is not None and isinstance(checkpoint.get("optimizer"), dict):
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            # Move optimizer states to the current device.
            for state in optimizer.state.values():
                for k, v in list(state.items()):
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        except Exception as e:
            print(f"warn: failed to restore optimizer state ({e}); continue without it.")

    if lr_scheduler is not None and checkpoint.get("lr_scheduler") is not None:
        try:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        except Exception as e:
            print(f"warn: failed to restore lr_scheduler state ({e}); continue without it.")

    train_state = checkpoint.get("train_state")
    return train_state if isinstance(train_state, dict) else {}


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

    def state_dict(self) -> dict:
        return {"window_size": int(self.window_size), "avg": float(self.avg), "size": int(self.size)}

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if "window_size" in state and int(state["window_size"]) != int(self.window_size):
            # Keep current window_size; treat as incompatible.
            return
        if "avg" in state:
            self.avg = float(state["avg"])
        if "size" in state:
            self.size = int(state["size"])

class MaskedMSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = torch.nn.MSELoss()

    def forward(self, output, eps, mask):
        B = output.shape[0]
        losses = []
        for b in range(B):
            m = mask[b, :2, :]
            pred = output[b][m]
            if isinstance(eps, (list, tuple)):
                target = eps[b].reshape(-1)
            else:
                target = eps[b][m]
            if pred.numel() != target.numel():
                raise ValueError(
                    f"MaskedMSE shape mismatch: pred={pred.numel()} target={target.numel()} "
                    f"(mask_true={int(m.sum().item())})"
                )
            losses.append(self.mse(pred, target))
        return torch.mean(torch.stack(losses))
