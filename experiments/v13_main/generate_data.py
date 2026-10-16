# generate_v13_dataset.py
import numpy as np
import pandas as pd
from scipy.stats import qmc
from get_v import get_v13

# 设置铺层层
d = 16  # 只需要修改这一行来改变层数

# 生成Sobol序列
sampler = qmc.Sobol(d=d, scramble=True, seed=42)
samples = sampler.random(3000)

# 映射到0-180度范围
angles = samples * 180

# 计算v13值
v13_values = np.array([get_v13(angle_seq) for angle_seq in angles])

# 创建DataFrame
df = pd.DataFrame()
for i in range(d):
    df[f'angle_{i+1}'] = angles[:, i]
df['v13'] = v13_values

# 保存到CSV
output_file = f'composite_v13_dataset.csv'
df.to_csv(output_file, index=False)
print(f"数据集生成完成，已保存到 {output_file}")
print(f"生成了 {d} 层铺层角度，共 {len(df)} 个样本")
