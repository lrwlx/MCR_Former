import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Nature 风格全局样式设置 (修复顶部小点问题)
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
    'axes.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4.0,   # 主刻度长度
    'ytick.major.size': 4.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': False,        # 【关键修复】关闭顶部刻度线
    'ytick.right': False,      # 【关键修复】关闭右侧刻度线
    'xtick.minor.visible': False, # 【关键修复】彻底隐藏次刻度（小点）
    'ytick.minor.visible': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# 调色板 (Colorblind-friendly & Nature style)
COLOR_DARK_BLUE = '#0072B2'  # 21 layers (Near) / Transfer Learning
COLOR_LIGHT_BLUE = '#56B4E9' # 25 layers (Medium)
COLOR_ORANGE = '#D55E00'     # 30 layers (Far) / Emphasis
COLOR_GRAY = '#999999'       # Scratch / Baseline

# ==========================================
# 2. 数据准备
# ==========================================
# Table 4 Data (Cross-Scale: 0, 1, 5 samples)
samples_5a = [0, 1, 5]
x_pos_5a = np.arange(len(samples_5a))

layer_21 = {'mean': [0.954, 0.960, 0.955], 'std': [0.000, 0.004, 0.008]}
layer_25 = {'mean': [0.908, 0.944, 0.938], 'std': [0.000, 0.002, 0.010]}
layer_30 = {'mean': [0.856, 0.893, 0.889], 'std': [0.000, 0.002, 0.009]}

# Table 5 Data (Cross-Task: 4 to 400 samples)
samples_5b = [4, 8, 20, 40, 100, 200, 400]
transfer = {'mean': [0.557, 0.712, 0.904, 0.938, 0.957, 0.962, 0.964], 'std': [0.042, 0.035, 0.018, 0.009, 0.004, 0.002, 0.001]}
scratch = {'mean': [-0.124, -0.089, 0.453, 0.681, 0.792, 0.827, 0.844], 'std': [0.441, 0.512, 0.124, 0.076, 0.041, 0.028, 0.039]}

# ==========================================
# 3. 创建 1x2 子图布局
# ==========================================
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 4.2))

# ------------------------------------------
# Panel a: Cross-Scale Few-Shot Generalization
# ------------------------------------------
# 绘制三条线
ax_a.errorbar(x_pos_5a, layer_21['mean'], yerr=layer_21['std'], fmt='-o', color=COLOR_DARK_BLUE, 
              linewidth=1.8, markersize=7, capsize=3, elinewidth=1, label='21 layers (Near)', zorder=3)
ax_a.errorbar(x_pos_5a, layer_25['mean'], yerr=layer_25['std'], fmt='-s', color=COLOR_LIGHT_BLUE, 
              linewidth=1.8, markersize=7, capsize=3, elinewidth=1, label='25 layers (Medium)', zorder=3)
ax_a.errorbar(x_pos_5a, layer_30['mean'], yerr=layer_30['std'], fmt='-^', color=COLOR_ORANGE, 
              linewidth=1.8, markersize=7, capsize=3, elinewidth=1, label='30 layers (Far)', zorder=3)

# 关键标注 1: 21 层 Zero-shot 的卓越表现 (保留蓝色箭头，无黑色虚线)
ax_a.annotate('Near-domain zero-shot\nmatches pre-training ceiling', 
              xy=(0, 0.954), xytext=(0.8, 0.985),
              arrowprops=dict(arrowstyle='->', color=COLOR_DARK_BLUE, lw=1.2),
              fontsize=7.5, color=COLOR_DARK_BLUE, ha='center', va='bottom', fontweight='bold')

# 关键标注 2: 30 层 1-shot 的快速恢复
ax_a.annotate('Rapid recovery\nwith 1-shot', 
              xy=(1, 0.893), xytext=(1.6, 0.865),
              arrowprops=dict(arrowstyle='->', color=COLOR_ORANGE, lw=1.2),
              fontsize=7, color=COLOR_ORANGE, ha='left', va='center')

# 坐标轴设置
ax_a.set_xticks(x_pos_5a)
ax_a.set_xticklabels(['0\n(Zero-shot)', '1', '5'])
ax_a.set_xlabel('Number of Fine-tuning Samples')
ax_a.set_ylabel(r'$R^2$ Score')
ax_a.set_ylim(0.80, 1.00)
ax_a.legend(loc='lower right', fontsize=7.5)

# 样式清理 (彻底移除顶部和右侧边框及刻度)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)
ax_a.tick_params(top=False, right=False, which='both') # 确保主/次刻度都不显示在顶部/右侧
ax_a.grid(axis='y', linestyle=':', alpha=0.3, linewidth=0.8)
ax_a.set_axisbelow(True)

# ------------------------------------------
# Panel b: Cross-Task Transfer Learning & Phase Transition
# ------------------------------------------
# Transfer Learning 曲线及阴影
ax_b.plot(samples_5b, transfer['mean'], color=COLOR_DARK_BLUE, linewidth=2.0, marker='o', markersize=5, label='Transfer learning')
ax_b.fill_between(samples_5b, 
                  np.array(transfer['mean']) - np.array(transfer['std']), 
                  np.array(transfer['mean']) + np.array(transfer['std']), 
                  color=COLOR_DARK_BLUE, alpha=0.15)

# Training from Scratch 曲线及阴影
ax_b.plot(samples_5b, scratch['mean'], color=COLOR_GRAY, linewidth=1.5, linestyle='--', marker='s', markersize=5, label='Training from scratch')
ax_b.fill_between(samples_5b, 
                  np.array(scratch['mean']) - np.array(scratch['std']), 
                  np.array(scratch['mean']) + np.array(scratch['std']), 
                  color=COLOR_GRAY, alpha=0.15)

# 关键标注 1: Phase transition threshold (X=20)
ax_b.axvline(x=20, color='black', linestyle=':', linewidth=1.2, alpha=0.7)
ax_b.text(22, 0.10, 'Phase transition\nthreshold\n(5% data)', fontsize=7, ha='left', va='center')

# 关键标注 2: Full-data limit of scratch training (Y=0.844)
scratch_limit = 0.844
ax_b.axhline(y=scratch_limit, color=COLOR_ORANGE, linestyle='-.', linewidth=1.2, alpha=0.9)
ax_b.text(50, scratch_limit + 0.03, 'Full-data limit of\nscratch training', fontsize=7, color=COLOR_ORANGE, ha='left')

# 坐标轴设置
ax_b.set_xscale('log')
ax_b.set_xticks([4, 8, 20, 40, 100, 200, 400])
ax_b.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax_b.set_xlabel('Number of Training Samples')
ax_b.set_ylim(-0.2, 1.05)
ax_b.legend(loc='lower right', fontsize=7)

# 样式清理 (彻底移除顶部和右侧边框及刻度)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)
ax_b.tick_params(top=False, right=False, which='both') # 确保主/次刻度都不显示在顶部/右侧
ax_b.grid(axis='y', linestyle=':', alpha=0.3, linewidth=0.8)
ax_b.set_axisbelow(True)

# 添加子图标签 (a, b)
fig.text(0.08, 0.95, 'a', fontsize=12, fontweight='bold')
fig.text(0.52, 0.95, 'b', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.93])

# 保存为三种格式
fig.savefig('Figure_5_Combined_Clean.png', format='png')
fig.savefig('Figure_5_Combined_Clean.pdf', format='pdf')
fig.savefig('Figure_5_Combined_Clean.svg', format='svg')
print("Successfully saved Figure_5_Combined_Clean.png/pdf/svg")
plt.close()