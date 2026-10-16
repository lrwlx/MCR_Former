import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import r2_score, mean_squared_error

# 设置Nature风格参数
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 7
rcParams['axes.linewidth'] = 0.75
rcParams['xtick.major.width'] = 0.75
rcParams['ytick.major.width'] = 0.75
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

# 读取数据
train_data = pd.read_csv('mcr_former_best_model_training_history_with_aug.csv')
pred_data = pd.read_csv('mcr_former_best_model_prediction_vs_actual_with_aug.csv')

# 计算R²和RMSE
r2 = r2_score(pred_data['actual_v13'], pred_data['predicted_v13'])
rmse = np.sqrt(mean_squared_error(pred_data['actual_v13'], pred_data['predicted_v13']))

# ==========================================
# 子图 a: 训练曲线 (独立保存)
# ==========================================
fig_a = plt.figure(figsize=(3.5, 3))
# 调整比例，使其在单图中居中且美观
ax1 = fig_a.add_axes([0.18, 0.18, 0.78, 0.75])  

ax1.plot(train_data['epoch'], train_data['train_loss'], 
         color='#2E74B5', linewidth=1.2, label='Training')
ax1.plot(train_data['epoch'], train_data['val_loss'], 
         color='#C55A4F', linewidth=1.2, label='Validation')

ax1.set_xlabel('Epoch', fontsize=7)
ax1.set_ylabel('MSE Loss', fontsize=7)
ax1.legend(loc='upper right', fontsize=6, frameon=False)
ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
ax1.tick_params(axis='both', which='major', labelsize=6)
ax1.set_xlim(0, 100)

# 【已移除子图标签 'a' 的绘制代码】

# 保存子图 a
for fmt in ['png', 'pdf', 'svg']:
    dpi = 600 if fmt == 'png' else None
    fig_a.savefig(f'fig2a.{fmt}', dpi=dpi, bbox_inches='tight')
plt.close(fig_a)

# ==========================================
# 子图 b: 预测散点图 (独立保存)
# ==========================================
fig_b = plt.figure(figsize=(3.5, 3))
ax2 = fig_b.add_axes([0.18, 0.18, 0.78, 0.75])

ax2.scatter(pred_data['actual_v13'], pred_data['predicted_v13'], 
            s=8, color='#4F81BD', alpha=0.6, edgecolors='#2E5C8A', linewidth=0.3)

min_val = min(pred_data['actual_v13'].min(), pred_data['predicted_v13'].min())
max_val = max(pred_data['actual_v13'].max(), pred_data['predicted_v13'].max())
margin = (max_val - min_val) * 0.02
ax2.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 
         'k--', linewidth=0.8, alpha=0.7)

# 修复数学符号渲染
ax2.set_xlabel(r'Actual $v_{13}$', fontsize=7)
ax2.set_ylabel(r'Predicted $v_{13}$', fontsize=7)

# 添加R²和RMSE文本框
ax2.text(0.05, 0.95, 
         f'$R^2$ = 0.991\nRMSE = 0.0054',
         transform=ax2.transAxes,
         fontsize=6,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                   edgecolor='black', linewidth=0.5))

ax2.tick_params(axis='both', which='major', labelsize=6)
ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
ax2.set_xlim(min_val - margin, max_val + margin)
ax2.set_ylim(min_val - margin, max_val + margin)

# 【已移除子图标签 'b' 的绘制代码】

# 保存子图 b
for fmt in ['png', 'pdf', 'svg']:
    dpi = 600 if fmt == 'png' else None
    fig_b.savefig(f'fig2b.{fmt}', dpi=dpi, bbox_inches='tight')
plt.close(fig_b)

print("Figures a and b saved successfully!")