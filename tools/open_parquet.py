"""
读取 Parquet 数据集并打印内容与字段类型（临时查看脚本）。

默认读取 `Dataset/apartment_dataset.parquet`。
"""

import pandas as pd

df = pd.read_parquet("Dataset/apartment_dataset.parquet")
print(df)
print("====================================")
print(df.dtypes)
