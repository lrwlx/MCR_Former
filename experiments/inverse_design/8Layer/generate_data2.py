# generate_v13_dataset_hybrid.py
import numpy as np
import pandas as pd
from scipy.stats import qmc
from get_v13 import get_v13

def generate_hybrid_dataset(n_samples=5000):
    """
    混合采样策略：
    1. Sobol序列保证空间均匀性
    2. 边界重点采样
    3. 对角线/角落区域采样
    4. 随机扰动探索
    """
    
    components = []
    v13_components = []
    
    # 1. Sobol序列采样 (40%)
    print("1. Sobol序列采样...")
    n_sobol = int(n_samples * 0.4)
    sampler = qmc.Sobol(d=8, scramble=True, seed=42)
    sobol_samples = sampler.random(n_sobol)
    sobol_angles = sobol_samples * 180 - 90
    sobol_v13 = np.array([get_v13(angle) for angle in sobol_angles])
    
    components.append(sobol_angles)
    v13_components.append(sobol_v13)
    
    # 2. 边界采样 (30%)
    print("2. 边界采样...")
    n_boundary = int(n_samples * 0.3)
    boundary_angles = []
    
    for _ in range(n_boundary):
        angle = np.random.uniform(-90, 90, 8)
        # 随机选择几个维度强制在边界附近
        n_boundary_dims = np.random.randint(2, 5)
        boundary_dims = np.random.choice(8, n_boundary_dims, replace=False)
        
        for dim in boundary_dims:
            # 在-90或90附近，±10度范围内
            if np.random.random() > 0.5:
                angle[dim] = -90 + np.random.uniform(0, 10)
            else:
                angle[dim] = 90 - np.random.uniform(0, 10)
        
        boundary_angles.append(angle)
    
    boundary_angles = np.array(boundary_angles)
    boundary_v13 = np.array([get_v13(angle) for angle in boundary_angles])
    
    components.append(boundary_angles)
    v13_components.append(boundary_v13)
    
    # 3. 对角区域采样 (20%)
    print("3. 对角区域采样...")
    n_diagonal = int(n_samples * 0.2)
    diagonal_angles = []
    
    for _ in range(n_diagonal):
        # 对角区域：多个维度相关
        base = np.random.uniform(-90, 90)
        direction = np.random.choice([-1, 1])
        
        angle = []
        for i in range(8):
            # 添加一些相关性
            if np.random.random() > 0.7:
                value = base + direction * np.random.uniform(0, 30)
            else:
                value = np.random.uniform(-90, 90)
            angle.append(np.clip(value, -90, 90))
        
        diagonal_angles.append(angle)
    
    diagonal_angles = np.array(diagonal_angles)
    diagonal_v13 = np.array([get_v13(angle) for angle in diagonal_angles])
    
    components.append(diagonal_angles)
    v13_components.append(diagonal_v13)
    
    # 4. 在已发现的小v13区域附近采样 (10%)
    print("4. 局部密集采样...")
    n_local = n_samples - sum(len(c) for c in components)
    
    # 找出所有样本中v13最小的点
    all_angles_temp = np.vstack(components)
    all_v13_temp = np.concatenate(v13_components)
    
    # 找出v13最小的10个点
    min_indices = np.argsort(all_v13_temp)[:10]
    min_points = all_angles_temp[min_indices]
    
    local_angles = []
    for center in min_points:
        for _ in range(n_local // 10):
            # 在中心点周围采样
            perturbation = np.random.normal(0, 5, 8)  # 小范围扰动
            new_angle = center + perturbation
            new_angle = np.clip(new_angle, -90, 90)
            local_angles.append(new_angle)
    
    local_angles = np.array(local_angles)
    local_v13 = np.array([get_v13(angle) for angle in local_angles])
    
    components.append(local_angles)
    v13_components.append(local_v13)
    
    # 合并所有样本
    all_angles = np.vstack(components)
    all_v13 = np.concatenate(v13_components)
    
    return all_angles, all_v13

# 生成数据集
angles, v13_values = generate_hybrid_dataset(4000)

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
    'v13': v13_values
})

# 保存到CSV
df.to_csv('composite_v13_dataset_8.csv', index=False)

print("\n" + "="*50)
print("数据集生成完成")
print("="*50)
print(f"总样本数: {len(v13_values)}")
print(f"v13统计信息:")
print(f"  最小值: {v13_values.min():.6f}")
print(f"  5%分位数: {np.percentile(v13_values, 5):.6f}")
print(f"  25%分位数: {np.percentile(v13_values, 25):.6f}")
print(f"  中位数: {np.median(v13_values):.6f}")
print(f"  75%分位数: {np.percentile(v13_values, 75):.6f}")
print(f"  95%分位数: {np.percentile(v13_values, 95):.6f}")
print(f"  最大值: {v13_values.max():.6f}")
print(f"  v13 < 0.001的样本数: {(v13_values < 0.001).sum()}")
print(f"  v13 < 0.01的样本数: {(v13_values < 0.01).sum()}")
print(f"  v13 < 0.1的样本数: {(v13_values < 0.1).sum()}")
print("="*50)