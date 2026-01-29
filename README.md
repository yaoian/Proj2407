# Proj2407

## 训练入口

- 训练脚本：`train.py`
- 主要配置：`Configs.py`

## “带语义信息”的数据在本项目中的形态

本项目目前已经内置了一条“语义信息/元数据(meta)参与模型输入”的路径（用于 `dataset_name == "apartments"`）：

- 数据集：`Dataset/DatasetApartments.py` 返回 `(traj, mask, loc_guess, loc_mean, meta)`
  - `traj`: `(3, L)`，float32，(lng, lat, sec_of_day) 已归一化
  - `mask`: `(L,)`，float32，1 表示被擦除点，0 表示保留点，`-1` 表示 padding
  - `loc_guess`: `(2, L)`，float32，擦除点的插值初猜
  - `loc_mean`: `(2, 1)`，float32，用于可选的中心化/反中心化
  - `meta`: `(4, L)`，int64，按时间步对齐的离散语义字段（见 `Models/EmbeddingModule.py`）
- BatchManager：`BatchManagers/BatchManagerApartments.py` 会把 `loc_mean/meta` 组织进 batch 并在训练循环中使用
- Embedder：`Models/EmbeddingModule.py::Embedder` 把 `meta + loc_mean` 映射为 `(B, embed_dim, L)`，再与轨迹输入拼接

如果你要“添加一个新的带语义信息的数据集”，建议直接复用上述形态：

1. 新增 `Dataset/DatasetXXX.py`，保证 `__getitem__` 返回 `traj/mask/loc_guess/loc_mean/meta`（语义字段数量/取值范围不同时需要同步改 `Embedder`）。
2. 新增对应的 `BatchManagers/BatchManagerXXX.py`（可以从 `BatchManagerApartments.py` 复制改最少量代码）。
3. 在 `Configs.py` 增加新的 `dataset_name` 分支：
   - `dataset_args["load_path"]` 指向你的数据文件
   - `get_batch_manager_class()` 返回你的 BatchManager
   - 设置 `embed_dim > 0`，并确保 `base_in_c = 6 + embed_dim` 与已训练 checkpoint 的模型结构一致

## 冻结训练（基于已训练 checkpoint 的 finetune）

当你希望在“加载已训练模型”的基础上，只训练部分模块（例如只训练 embedder/新加模块）时：

1. 在 `Configs.py` 设置 `resume_checkpoint = "path/to/xxx.pth"`
2. 设置冻结开关：
   - `freeze_unet = True/False`
   - `freeze_linkage = True/False`
   - `freeze_embedder = True/False`
3. 运行 `python train.py`

注意：如果你“新增语义通道”导致 `in_c` 变化（例如从 6 变成 `6 + embed_dim`），那么旧 checkpoint 通常无法直接加载（形状不匹配）。这种情况下要么：

- 重新训练一份匹配新结构的模型；要么
- 改成“不改变输入通道”的条件注入方式（例如在网络内部加条件/adapter），并允许旧权重以 `strict=False` 方式加载（需要进一步改代码）。
