"""
快速查看 checkpoint（.pth）里包含的模块与参数量（临时调试脚本）。

默认读取 `Runs/.../best.pth`，打印 unet/linkage/embedder 的 state_dict 键数量与前缀情况。
"""

import torch
ckpt = torch.load('Runs/2025-12-20_16-16-45/best.pth', map_location='cpu')
print(ckpt.keys())            # dict_keys(['embedder', 'linkage', 'unet'])
print(len(ckpt['unet']))  
for k in ckpt['unet'].keys():
    print(k.startswith('module.'))
print(len(ckpt['linkage']))
print(len(ckpt['embedder']))
