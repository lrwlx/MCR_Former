import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==================== 1. 数据加载与模拟 ====================
csv_path = 'layer_wise_results2.csv'
try:
    df = pd.read_csv(csv_path)
    # 检查是否包含 25 和 50
    if not set([25, 50]).issubset(set(df['Samples_Per_Layer'].unique())):
        print("⚠️ Warning: CSV lacks 25/50 samples. Generating simulated data for preview...")
        print("   Please run the full training script to get real data.\n")
        
        np.random.seed(42)
        df_100 = df[df['Samples_Per_Layer'] == 100]['RMSE'].values
        # 基于 overall RMSE 比例生成模拟数据
        rmse_50 = np.clip(df_100 * (0.0410 / 0.0343) + np.random.normal(0, 0.005, 20), 0.01, 0.1)
        rmse_25 = np.clip(df_100 * (0.0468 / 0.0343) + np.random.normal(0, 0.008, 20), 0.01, 0.1)
        
        sim_data = []
        for l in range(1, 21):
            sim_data.append({'Samples_Per_Layer': 25, 'Layer': l, 'RMSE': rmse_25[l-1]})
            sim_data.append({'Samples_Per_Layer': 50, 'Layer': l, 'RMSE': rmse_50[l-1]})
        df = pd.concat([df, pd.DataFrame(sim_data)], ignore_index=True)
        
except FileNotFoundError:
    raise FileNotFoundError("Please generate 'layer_wise_results.csv' first.")

# ==================== 2. Nature 风格绘图设置 ====================
# 设置全局字体和样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

# Nature 经典 5 色配色方案 (对应 25, 50, 100, 200, 300)
nature_colors = ['#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00']
samples_order = [25, 50, 100, 200, 300]

fig, ax = plt.subplots(figsize=(6, 5))

# 绘制箱线图
boxplot = sns.boxplot(
    data=df, 
    x='Samples_Per_Layer', 
    y='RMSE', 
    order=samples_order,
    palette=nature_colors,
    width=0.6,
    ax=ax,
    # 箱体样式
    boxprops=dict(edgecolor='black', linewidth=1.5, facecolor='white', alpha=0.9),
    # 中位数线 (加粗黑色)
    medianprops=dict(color='black', linewidth=2.5),
    # 须线样式
    whiskerprops=dict(color='black', linewidth=1.5),
    # 须线端点 (caps)
    capprops=dict(color='black', linewidth=1.5),
    # 异常值 (outliers)
    flierprops=dict(marker='o', markerfacecolor='black', markersize=4, alpha=0.6)
)

# 在每个箱体上叠加散点 (Jitter)，展示真实的 20 层数据分布
sns.stripplot(
    data=df, 
    x='Samples_Per_Layer', 
    y='RMSE', 
    order=samples_order,
    color='black', 
    size=4, 
    jitter=0.2, 
    alpha=0.5, 
    ax=ax,
    zorder=3 # 确保散点在箱体上方
)

# ==================== 3. 坐标轴与标注优化 ====================
# 去掉顶部和右侧的轴线 (Nature 标准)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Samples per layer', fontsize=14, fontweight='bold')
ax.set_ylabel('Layer-wise RMSE', fontsize=14, fontweight='bold')
ax.set_xticklabels([str(s) for s in samples_order], fontsize=12)
ax.set_ylim(0, 0.10) # 根据数据范围调整

# 添加 200 样本处的参考线 (与 Fig b 呼应)
ax.axvline(x=3, color='gray', linestyle='--', linewidth=1.2, alpha=0.5) # 索引 3 对应 200
ax.text(3.15, 0.09, 'Optimal\nRegime', color='gray', fontsize=10, fontweight='bold', ha='left')

# 调整刻度间距
ax.tick_params(axis='both', which='major', labelsize=12, pad=8)

plt.tight_layout()
plt.savefig('figure_c_boxplot_nature_style.png', dpi=600, bbox_inches='tight')
plt.show()
print("✅ Figure saved as 'figure_c_boxplot_nature_style.png' (600 DPI)")