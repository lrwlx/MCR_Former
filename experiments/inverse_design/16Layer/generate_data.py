# generate_v13_dataset.py
import numpy as np
import pandas as pd
from scipy.stats import qmc
from get_v13 import get_v13

# 生成Sobol序列
sampler = qmc.Sobol(d=16, scramble=True, seed=42)
samples = sampler.random(10000)

# 映射到0-180度范围
angles = samples * 180 - 90

# 计算v13值
v13_values = np.array([get_v13(angle_seq) for angle_seq in angles])

# 创建DataFrame
df = pd.DataFrame({
    'angle_1': angles[:, 0],
    'angle_2': angles[:, 1],
    'angle_3': angles[:, 2],
    'angle_4': angles[:, 3],
    'angle_5': angles[:, 4],
    'angle_6': angles[:, 5],
    'angle_7': angles[:, 6],
    'angle_8': angles[:, 7],
    'angle_9': angles[:, 8],
    'angle_10': angles[:, 9],
    'angle_11': angles[:, 10],
    'angle_12': angles[:, 11],
    'angle_13': angles[:, 12],
    'angle_14': angles[:, 13],
    'angle_15': angles[:, 14],
    'angle_16': angles[:, 15],
    'v13': v13_values
})

# 保存到CSV
df.to_csv('composite_v13_dataset_16.csv', index=False)
print("数据集生成完成，已保存到 composite_v13_dataset.csv")