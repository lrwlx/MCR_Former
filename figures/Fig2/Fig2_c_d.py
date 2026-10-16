import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# --- Nature Style Configuration (Unified with Fig 2a/2b) ---
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 7
rcParams['axes.linewidth'] = 0.75
rcParams['xtick.major.width'] = 0.75
rcParams['ytick.major.width'] = 0.75
rcParams['xtick.minor.width'] = 0.75
rcParams['ytick.minor.width'] = 0.75
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['axes.spines.top'] = True
rcParams['axes.spines.right'] = True
rcParams['xtick.minor.visible'] = True
rcParams['ytick.minor.visible'] = True
rcParams['xtick.major.size'] = 3.5
rcParams['ytick.major.size'] = 3.5
rcParams['xtick.minor.size'] = 2
rcParams['ytick.minor.size'] = 2
rcParams['lines.linewidth'] = 1
rcParams['savefig.dpi'] = 600
rcParams['figure.dpi'] = 300

# --- Data Preparation ---
models = ['Random Forest', 'LSTM', 'CNN', 'Transformer', 'Informer', 
          'ModernTCN', 'TFT', 'DeepAR', 'LayupFormer', 'MCR-Former']
r2_scores = [0.328, 0.711, 0.715, 0.725, 0.891, 0.926, 0.937, 0.946, 0.972, 0.991]
rmse_scores = [0.7804, 0.0306, 0.0304, 0.0298, 0.0188, 0.0155, 0.0143, 0.0133, 0.0093, 0.0054]

# Muted/Pale Colors (Matching Fig 2a/2b style)
color_baseline = '#4F81BD'  # Soft blue
color_mcr = '#D9534F'       # Soft red
colors = [color_baseline] * (len(models) - 1) + [color_mcr]

y_pos = np.arange(len(models))

# ==========================================
# Subplot (c): R² Comparison
# ==========================================
fig_c = plt.figure(figsize=(3.5, 3))
ax1 = fig_c.add_axes([0.18, 0.18, 0.75, 0.75])  

bars1 = ax1.barh(y_pos, r2_scores, color=colors, edgecolor='black', linewidth=0.4, height=0.65)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(models, fontsize=7)
ax1.invert_yaxis()  

ax1.set_xlabel('$R^2$', fontsize=7)
ax1.set_xlim(0, 1.12)  
ax1.set_xticks([0.3, 0.5, 0.7, 0.9, 1.0])

# X轴保留刻度线，Y轴移除所有刻度线
ax1.tick_params(axis='x', which='major', labelsize=6)
ax1.tick_params(axis='y', which='major', labelsize=6, left=False)
ax1.tick_params(axis='y', which='minor', left=False)
ax1.yaxis.set_minor_locator(plt.NullLocator())

# Grid (very faint)
ax1.grid(axis='x', linestyle='--', alpha=0.3, color='gray', linewidth=0.5)
ax1.set_axisbelow(True)

# Add value labels (线性坐标使用加法偏移)
for i, (val, bar) in enumerate(zip(r2_scores, bars1)):
    label = f'{val:.3f}'
    ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, label, 
             va='center', ha='left', fontsize=6, color='black')

# 【已移除子图标签 'c' 的绘制代码】

# Save Subplot c
for fmt in ['png', 'pdf', 'svg']:
    dpi = 600 if fmt == 'png' else None
    fig_c.savefig(f'fig2c.{fmt}', dpi=dpi, bbox_inches='tight')
plt.close(fig_c)

# ==========================================
# Subplot (d): RMSE Comparison (Log Scale)
# ==========================================
fig_d = plt.figure(figsize=(3.5, 3))
ax2 = fig_d.add_axes([0.18, 0.18, 0.75, 0.75])

bars2 = ax2.barh(y_pos, rmse_scores, color=colors, edgecolor='black', linewidth=0.4, height=0.65)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(models, fontsize=7) 
ax2.invert_yaxis()

ax2.set_xscale('log')
ax2.set_xlabel('RMSE (log scale)', fontsize=7)
ax2.set_xlim(1e-3, 2.0)  
ax2.set_xticks([1e-3, 1e-2, 1e-1, 1e0])
ax2.set_xticklabels(['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$'], fontsize=6)

# X轴保留刻度线，Y轴移除所有刻度线
ax2.tick_params(axis='x', which='major', labelsize=6)
ax2.tick_params(axis='y', which='major', labelsize=6, left=False)
ax2.tick_params(axis='y', which='minor', left=False)
ax2.yaxis.set_minor_locator(plt.NullLocator())

ax2.grid(axis='x', linestyle='--', alpha=0.3, color='gray', linewidth=0.5)
ax2.set_axisbelow(True)

# Add value labels - 动态乘法偏移
for i, (val, bar) in enumerate(zip(rmse_scores, bars2)):
    label = f'{val:.4f}'
    # 动态调整偏移比例
    if val > 0.5:
        pos_x = val * 1.05  # 对于 Random Forest (0.7804)，仅偏移 5%，紧贴柱子
    elif val > 0.1:
        pos_x = val * 1.15  # 中等数值偏移 15%
    else:
        pos_x = val * 1.35  # 小数值偏移 35%
        
    ax2.text(pos_x, bar.get_y() + bar.get_height()/2, label, 
             va='center', ha='left', fontsize=6, color='black')

# 【已移除子图标签 'd' 的绘制代码】

# Save Subplot d
for fmt in ['png', 'pdf', 'svg']:
    dpi = 600 if fmt == 'png' else None
    fig_d.savefig(f'fig2d.{fmt}', dpi=dpi, bbox_inches='tight')
plt.close(fig_d)

print("Figures c and d saved successfully!")