"""
快速打印 `Dataset/apartment_dataset.pth` 的顶层 keys（临时调试脚本）。
"""

import torch

temp = torch.load("./Dataset/apartment_dataset.pth")
print(temp.keys())
