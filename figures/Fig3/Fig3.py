import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ==========================================
# 1. Nature 风格全局样式配置
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'axes.linewidth': 0.75,
    'xtick.major.width': 0.75,
    'ytick.major.width': 0.75,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'lines.linewidth': 1.0,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# ==========================================
# 2. 数据准备
# ==========================================
models_original = ['Transformer', 'LSTM', 'CNN', 'SVR', 'Random Forest', 'Ridge', 'MLP']
# 补全数学符号
encodings = ['raw angle', r'sin$\theta$,cos$\theta$', r'sin4$\theta$,cos4$\theta$',
             r'sin2$\theta$,cos2$\theta$ (MCR)', r'sin$\theta$,cos$\theta$,sin2$\theta$,cos2$\theta$(4D)', 'Tsai 6D']

r2_data_old = [
    [0.7247, 0.9338, 0.9631, 0.1529, 0.9630, 0.9696],
    [0.7148, 0.7603, 0.9091, 0.1646, 0.9072, 0.9644],
    [0.6444, 0.8629, 0.8758, 0.0133, 0.8758, 0.8753],
    [0.2623, 0.6576, 0.7767, 0.0881, 0.7632, 0.7989],
    [0.3324, 0.5070, 0.5062, 0.1241, 0.4908, 0.4731],
    [-0.3378, 0.0303, 0.6146, -0.7104, 0.6133, 0.7570],
    [-0.0149, 0.3137, 0.4455, 0.1582, 0.6128, 0.6632]
]

# 重排顺序：Ridge 和 MLP 互换，并将 MCR 列提前
new_order_indices = [0, 1, 2, 3, 4, 6, 5]
models_reordered = [models_original[i] for i in new_order_indices]
r2_data_reordered_rows = [r2_data_old[i] for i in new_order_indices]

r2_data_final = []
for row in r2_data_reordered_rows:
    new_row = [row[0], row[1], row[3], row[2], row[4], row[5]]
    r2_data_final.append(new_row)

df_r2 = pd.DataFrame(r2_data_final, index=models_reordered, columns=encodings)
df_r2 = df_r2.iloc[::-1]  # 反转行顺序

# ==========================================
# 3. 核心修改：非对称色彩映射 (Asymmetric Colormap)
# ==========================================
# 【关键策略】将 vcenter 从 0.0 移到 0.5。
# 这样 0.5 到 1.0 的数据将占据色图的主要渐变范围（白 -> 浅红 -> 深红），
# 而负值区域被压缩为极淡的蓝色，不再抢戏。
vmin, vcenter, vmax = -0.5, 0.5, 1.0

colors_div = [
    (0.0,  (0.80, 0.85, 0.90)),  # 负值端：极淡灰蓝 (表示失效)
    (0.4,  (0.95, 0.95, 0.95)),  # 过渡
    (0.5,  (1.00, 1.00, 1.00)),  # 0.5 点：纯白 (基准线)
    (0.7,  (0.95, 0.70, 0.60)),  # 过渡：浅红
    (1.0,  (0.75, 0.15, 0.10)),  # 正值端：深红 (Nature 强调色)
]
cmap_nature = LinearSegmentedColormap.from_list('nature_div_emphasis', colors_div, N=256)
norm_nature = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

# ==========================================
# 4. 绘图
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5)) 

sns.heatmap(
    df_r2,
    annot=True,
    fmt='.3f',
    annot_kws={'size': 7.5, 'fontweight': 'normal', 'color': 'black'}, 
    cmap=cmap_nature,
    norm=norm_nature,
    linewidths=0.5,           # 细白线分隔
    linecolor='white',
    cbar=True,
    cbar_kws={
        'label': r'$R^2$',
        'shrink': 0.8,
        'pad': 0.02,
        'ticks': [0.0, 0.5, 0.8, 1.0], # 调整刻度以匹配新的色彩重心
        'format': '%.1f',
    },
    ax=ax,
)

# 轴标签
ax.set_xlabel('Encoding Type', fontsize=9, labelpad=10)
ax.set_ylabel('Model', fontsize=9, labelpad=10)

# 刻度标签旋转与对齐
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    ha='right',
    fontsize=8,
    rotation_mode='anchor'
)
ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    fontsize=8,
)

# Colorbar 样式微调 (极简)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=7, width=0, size=0) # 移除 Colorbar 的刻度线
cbar.ax.spines['top'].set_visible(False)
cbar.ax.spines['right'].set_visible(False)
cbar.ax.spines['left'].set_visible(False)
cbar.ax.spines['bottom'].set_visible(False)
cbar.outline.set_visible(False) # 移除 Colorbar 外框
cbar.set_label(r'$R^2$', fontsize=8, labelpad=8, rotation=0, va='center', ha='left')

# 移除主坐标轴边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(top=False, right=False)

plt.tight_layout(pad=0.5)

# ==========================================
# 5. 保存输出
# ==========================================
output_base = 'Fig3_Ablation_Heatmap_HighContrast'
for fmt in ['png', 'pdf', 'svg']:
    dpi = 600 if fmt == 'png' else None
    fig.savefig(f'{output_base}.{fmt}', format=fmt, dpi=dpi)
    print(f"✅ Successfully saved: {output_base}.{fmt}")

plt.close(fig)