import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Nature 风格全局样式配置 (严格极简 & 紧凑布局)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.linewidth': 0.75,
    'xtick.major.width': 0.75,
    'ytick.major.width': 0.75,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    # 彻底关闭顶部/右侧边框和次刻度
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'lines.linewidth': 1.0,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# 统一调色板 (Colorblind-friendly)
COLOR_BLUE = '#4F81BD'   # 基线/训练集
COLOR_RED = '#D9534F'    # MCR-Former 强调色
COLOR_GRAY = '#808080'   # 其他基线模型

# ==========================================
# 2. 数据准备 (直接读取真实 CSV)
# ==========================================
csv_a = 'mcr_former_best_model_training_history_with_aug.csv'
csv_b = 'mcr_former_best_model_prediction_vs_actual_with_aug.csv'

# 读取图 a 数据
df_a = pd.read_csv(csv_a)
epochs = df_a['epoch']
train_loss = df_a['train_loss']
val_loss = df_a['val_loss']

# 读取图 b 数据
df_b = pd.read_csv(csv_b)
actual = df_b['actual_v13']
predicted = df_b['predicted_v13']

# 图 c/d 数据 (Model Comparison)
models = ['Random Forest', 'LSTM', 'CNN', 'Transformer', 'Informer',
          'ModernTCN', 'TFT', 'DeepAR', 'LayupFormer', 'MCR-Former']
r2_scores = [0.328, 0.711, 0.715, 0.725, 0.891, 0.926, 0.937, 0.946, 0.972, 0.991]
rmse_scores = [0.7804, 0.0306, 0.0304, 0.0298, 0.0188, 0.0155, 0.0143, 0.0133, 0.0093, 0.0054]

# ==========================================
# 3. 创建 2x2 画布 (Nature 双栏宽度 7.1 英寸，强制正方形)
# ==========================================
fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.2))
# 紧凑布局：减小垂直和水平间距
fig.subplots_adjust(wspace=0.30, hspace=0.22) 

# 提取子图轴
ax_a, ax_b = axs[0, 0], axs[0, 1]
ax_c, ax_d = axs[1, 0], axs[1, 1]

# ==========================================
# 4. 绘制子图 a: 训练曲线
# ==========================================
ax_a.plot(epochs, train_loss, color=COLOR_BLUE, linewidth=1.2, label='Training')
ax_a.plot(epochs, val_loss, color=COLOR_RED, linewidth=1.2, label='Validation')
ax_a.set_xlabel('Epoch')
ax_a.set_ylabel('MSE Loss')
ax_a.legend(loc='upper right', fontsize=6, frameon=False)
ax_a.set_xlim(0, 100)
ax_a.tick_params(labelsize=6)

# ==========================================
# 5. 绘制子图 b: 预测散点图
# ==========================================
ax_b.scatter(actual, predicted, s=5, color=COLOR_BLUE, alpha=0.4, edgecolors='none')
min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
margin = (max_val - min_val) * 0.05
ax_b.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 
          'k--', linewidth=0.8, alpha=0.6)
ax_b.set_xlabel(r'Actual $v_{13}$')
ax_b.set_ylabel(r'Predicted $v_{13}$')
ax_b.set_xlim(min_val - margin, max_val + margin)
ax_b.set_ylim(min_val - margin, max_val + margin)
ax_b.tick_params(labelsize=6)

# 【核心修改】严格按照 Table 1 的指标格式化文本框
text_content = (
    "MCR-Former\n"
    "$R^2$ = 0.991 \n"
    "RMSE = 0.0054 "
)
ax_b.text(0.05, 0.95, text_content,
          transform=ax_b.transAxes, fontsize=6, verticalalignment='top',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5))

# ==========================================
# 6. 绘制子图 c: R² 比较 (条形图)
# ==========================================
y_pos = np.arange(len(models))
colors_c = [COLOR_GRAY] * (len(models) - 1) + [COLOR_RED]
bars_c = ax_c.barh(y_pos, r2_scores, color=colors_c, edgecolor='none', height=0.65)
ax_c.set_yticks(y_pos)
ax_c.set_yticklabels(models, fontsize=6)
ax_c.invert_yaxis()
ax_c.set_xlabel('$R^2$')
ax_c.set_xlim(0, 1.10)
ax_c.tick_params(axis='x', labelsize=6, left=False) # 移除 Y 轴刻度线

for val, bar in zip(r2_scores, bars_c):
    ax_c.text(val + 0.015, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
              va='center', ha='left', fontsize=5.5, color='black')

# ==========================================
# 7. 绘制子图 d: RMSE 比较 (对数坐标)
# ==========================================
colors_d = [COLOR_GRAY] * (len(models) - 1) + [COLOR_RED]
bars_d = ax_d.barh(y_pos, rmse_scores, color=colors_d, edgecolor='none', height=0.65)
ax_d.set_yticks(y_pos)
ax_d.set_yticklabels(models, fontsize=6)
ax_d.invert_yaxis()
ax_d.set_xscale('log')
ax_d.set_xlabel('RMSE (log scale)')
ax_d.set_xlim(1e-3, 2.0)
ax_d.set_xticks([1e-3, 1e-2, 1e-1, 1e0])
ax_d.set_xticklabels(['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$'], fontsize=6)
ax_d.tick_params(axis='x', labelsize=6, left=False)

for val, bar in zip(rmse_scores, bars_d):
    pos_x = val * 1.15 if val > 0.01 else val * 1.35
    ax_d.text(pos_x, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
              va='center', ha='left', fontsize=5.5, color='black')

# ==========================================
# 8. 添加子图标签 (a, b, c, d)
# ==========================================
label_style = {'fontsize': 11, 'fontweight': 'bold', 'fontfamily': 'Arial'}

# 使用相对坐标，确保标签紧贴子图左上角外部
ax_a.text(-0.15, 1.05, 'a', transform=ax_a.transAxes, **label_style)
ax_b.text(-0.15, 1.05, 'b', transform=ax_b.transAxes, **label_style)
ax_c.text(-0.15, 1.05, 'c', transform=ax_c.transAxes, **label_style)
ax_d.text(-0.15, 1.05, 'd', transform=ax_d.transAxes, **label_style)

# ==========================================
# 9. 保存输出 (PNG, PDF, SVG)
# ==========================================
output_base = 'Figure_2_Combined_Final'
for fmt in ['png', 'pdf', 'svg']:
    dpi = 600 if fmt == 'png' else None
    fig.savefig(f'{output_base}.{fmt}', format=fmt, dpi=dpi)
    print(f"✅ Successfully saved: {output_base}.{fmt}")

plt.close(fig)