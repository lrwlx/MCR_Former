import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv('overall_results.csv')

# ==========================================
# 1. 全局样式设置 (Nature/学术风格)
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2  # 坐标轴线宽
plt.rcParams['lines.linewidth'] = 2.0 # 数据线宽
plt.rcParams['lines.markersize'] = 7  # 标记点大小

# ==========================================
# 2. 颜色定义 (一体两面：和谐配色)
# ==========================================
# R² 使用深黑色/深灰色，代表主要指标，稳重
color_r2 = '#000000'      
# RMSE 使用钢蓝色 (SteelBlue)，与黑色和谐，体现“一体”的冷色调感，避免黑黄冲突
color_rmse = '#4682B4'    
# 垂直参考线使用浅灰色或淡蓝色，不抢戏
color_vline = '#A9A9A9'   

fig, ax1 = plt.subplots(figsize=(7, 5))

# ==========================================
# 3. 绘制 R² (左 Y 轴)
# ==========================================
line1, = ax1.plot(df['Samples_Per_Layer'], df['R2'], 
                  marker='o', linestyle='-',
                  color=color_r2, linewidth=2.2, 
                  label='$R^2$ score')

ax1.set_xlabel('Samples per layer', fontsize=13, fontweight='bold')
ax1.set_ylabel('$R^2$ score', color=color_r2, fontsize=13, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_r2, direction='out', length=5)
ax1.set_ylim(0.58, 1.02)
ax1.set_xticks([25, 50, 100, 200, 300])
ax1.set_xlim(15, 315) # 稍微留白，防止边缘数据点被切

# 左侧和底部边框加粗，顶部和右侧隐藏 (L型框架，更现代)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False) 
ax1.spines['left'].set_linewidth(1.2)
ax1.spines['bottom'].set_linewidth(1.2)

# ==========================================
# 4. 绘制 RMSE (右 Y 轴)
# ==========================================
ax2 = ax1.twinx()
line2, = ax2.plot(df['Samples_Per_Layer'], df['RMSE'], 
                  marker='s', linestyle='--',
                  color=color_rmse, linewidth=2.2, 
                  label='RMSE')

ax2.set_ylabel('RMSE', color=color_rmse, fontsize=13, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_rmse, direction='out', length=5)
ax2.set_ylim(0.015, 0.050)

# 修复右侧轴显示问题：显示右侧边框，并设为 RMSE 的颜色
ax2.spines['right'].set_visible(True)
ax2.spines['right'].set_color(color_rmse)
ax2.spines['right'].set_linewidth(1.2)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False) # 隐藏左侧边框，避免与 ax1 重叠

# ==========================================
# 5. 添加垂直参考线 (200 samples)
# ==========================================
# 使用浅灰色虚线，标注文字放在右上方，避免遮挡曲线
ax1.axvline(x=200, color=color_vline, linestyle='--', linewidth=1.5, alpha=0.8)
#ax1.text(208, 0.98, '200\nsamples', color=color_vline, fontsize=10, fontweight='bold', va='top')

# ==========================================
# 6. 图例设置 (解决遮挡问题)
# ==========================================
# 将图例放在左下角内部，稍微往右上方偏移，避开 x=25 的数据点
# bbox_to_anchor 使用相对坐标 (0,0)是左下角，(1,1)是右上角
ax1.legend([line1, line2], ['$R^2$ score', 'RMSE'],
           loc='lower left',
           bbox_to_anchor=(0.15, 0.08),  # 向右和向上微调，避开数据点
           fontsize=11,
           frameon=False,                # 无边框，更清爽
           handlelength=2.5)             # 图例线段长度

# ==========================================
# 7. 标题与保存
# ==========================================
# 标题加粗，与参考图一致
plt.title('b  $R^2$ and RMSE vs. training samples', fontsize=14, fontweight='bold', pad=15)

fig.tight_layout()

# 保存为高清 PNG 和 PDF
plt.savefig('figure_b_harmonious.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_b_harmonious.pdf', format='pdf', bbox_inches='tight')

plt.show()