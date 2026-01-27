import torch
import torch.nn as nn
import torch.nn.functional as func
from typing import *
import math
from einops import rearrange

Tensor = torch.Tensor


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class NumberEmbedder(nn.Module):
    def __init__(self, max_num: int, hidden_dim: int = 256, embed_dim: int = 128) -> None:
        super().__init__()

        # --- Diffusion step Encoding ---
        position = torch.arange(max_num, dtype=torch.float32).unsqueeze(1)  # (max_time, 1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32) * -(math.log(1.0e4) / hidden_dim)
        )  # (hidden_dim / 2)
        encodings = torch.zeros((max_num, hidden_dim), dtype=torch.float32)  # (max_time, hidden_dim)
        encodings[:, 0::2] = torch.sin(position * div_term)
        encodings[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encodings", encodings, persistent=False)

        self.proj = nn.Linear(hidden_dim, embed_dim)  # (B, embed_dim)

    def forward(self, num: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        num = num.to(dtype=torch.long)
        time_embed = self.encodings[num, :]  # (B, hidden_dim)
        return self.proj(time_embed)  # (B, embed_dim, 1)
