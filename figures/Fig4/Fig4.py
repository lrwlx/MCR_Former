import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

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
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',      # 刻度朝内 (Nature 风格)
    'ytick.direction': 'in',
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.minor.visible': False,
    'ytick.minor.visible': False,
    'axes.spines.top': False,     # 移除顶部边框
    'axes.spines.right': False,   # 移除右侧边框
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,           # 强制 PDF 字体嵌入
    'ps.fonttype': 42,
})

# 调色板 (Okabe-Ito 色盲安全配色)
COLOR_BLACK = '#000000'
COLOR_ORANGE = '#E69F00'
COLOR_BLUE = '#56B4E9'
COLOR_GRAY = '#999999'

# ==========================================
# 2. 数据准备 (智能 Fallback 机制)
# ==========================================
def load_data():
    """尝试读取 CSV，若不存在则生成模拟数据"""
    csv_overall = 'overall_results.csv'
    csv_layer = 'layer_wise_results.csv'
    
    # --- 图 A/B 数据 (Overall Performance) ---
    if os.path.exists(csv_overall):
        df_overall = pd.read_csv(csv_overall)
    else:
        print("⚠️ overall_results.csv not found. Generating mock data...")
        df_overall = pd.DataFrame({
            'Samples_Per_Layer': [25, 50, 100, 200, 300],
            'R2': [0.59, 0.80, 0.88, 0.96, 0.965],
            'RMSE': [0.047, 0.041, 0.034, 0.018, 0.017]
        })
        
    # --- 图 C 数据 (Layer-wise Performance) ---
    if os.path.exists(csv_layer):
        df_layer = pd.read_csv(csv_layer)
    else:
        print("⚠️ layer_wise_results.csv not found. Generating mock data...")
        np.random.seed(42)
        layers = np.arange(1, 21)
        data_c = []
        for s in [100, 200, 300]:
            if s == 100:
                r2_vals = 0.90 + np.random.normal(0, 0.05, 20)
                r2_vals = np.clip(r2_vals, 0.75, 0.98)
            elif s == 200:
                r2_vals = 0.95 + np.random.normal(0, 0.01, 20)
                r2_vals = np.clip(r2_vals, 0.90, 0.99)
            else:
                r2_vals = 0.96 + np.random.normal(0, 0.005, 20)
                r2_vals = np.clip(r2_vals, 0.92, 0.99)
            
            for l, r in zip(layers, r2_vals):
                data_c.append({'Layer': l, 'Samples_Per_Layer': s, 'R2': r})
        df_layer = pd.DataFrame(data_c)
        
    return df_overall, df_layer

df_overall, df_layer = load_data()

# ==========================================
# 3. 创建画布与布局 (紧凑化调整)
# ==========================================
fig = plt.figure(figsize=(7.5, 7.0)) 
# 【核心修改】将 hspace 从 0.35 降至 0.22，大幅缩短上下行间距；wspace 微调至 0.30
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.22, wspace=0.30)

ax_a = fig.add_subplot(gs[0, 0]) 
ax_b = fig.add_subplot(gs[0, 1]) 
ax_c = fig.add_subplot(gs[1, :]) 

# ==========================================
# 4. 绘制 Panel A: R² vs Samples
# ==========================================
ax_a.plot(df_overall['Samples_Per_Layer'], df_overall['R2'], 
          color=COLOR_BLACK, marker='o', linestyle='-', linewidth=1.5, markersize=6)
ax_a.set_xlabel('Samples per layer')
ax_a.set_ylabel('$R^2$ score')
ax_a.set_ylim(0.55, 1.02)
ax_a.set_xticks([25, 50, 100, 200, 300])
ax_a.axvline(x=200, color=COLOR_GRAY, linestyle='--', linewidth=1.0, alpha=0.6)

# ==========================================
# 5. 绘制 Panel B: RMSE vs Samples
# ==========================================
ax_b.plot(df_overall['Samples_Per_Layer'], df_overall['RMSE'], 
          color=COLOR_BLUE, marker='s', linestyle='--', linewidth=1.5, markersize=6)
ax_b.set_xlabel('Samples per layer')
ax_b.set_ylabel('RMSE')
ax_b.set_ylim(0.015, 0.050) 
ax_b.set_xticks([25, 50, 100, 200, 300])
ax_b.axvline(x=200, color=COLOR_GRAY, linestyle='--', linewidth=1.0, alpha=0.6)

# ==========================================
# 6. 绘制 Panel C: Layer-wise R²
# ==========================================
samples_to_plot = [100, 200, 300]
colors_c = [COLOR_BLACK, COLOR_ORANGE, COLOR_BLUE]
markers_c = ['o', 's', '^']

for i, s in enumerate(samples_to_plot):
    subset = df_layer[df_layer['Samples_Per_Layer'] == s].sort_values('Layer')
    ax_c.plot(subset['Layer'], subset['R2'], 
              color=colors_c[i], marker=markers_c[i], linestyle='-', 
              linewidth=1.2, markersize=5, label=f'{s} samples/layer')

ax_c.set_xlabel('Number of layers')
ax_c.set_ylabel('$R^2$ score')
ax_c.set_xticks(range(0, 21, 5))
ax_c.set_ylim(0.70, 1.02)
ax_c.legend(loc='lower left')

# ==========================================
# 7. 添加子图标签 (a, b, c) - 坐标同步微调
# ==========================================
label_style = {'fontsize': 11, 'fontweight': 'bold', 'fontfamily': 'Arial'}
# 由于 hspace 减小，第二行上移，c 的 Y 坐标从 0.52 调整为 0.48 以保持相对位置
fig.text(0.08, 0.93, 'a', **label_style)
fig.text(0.52, 0.93, 'b', **label_style)
fig.text(0.08, 0.48, 'c', **label_style)

# ==========================================
# 8. 保存输出 (PNG, PDF, SVG)
# ==========================================
output_base = 'Figure_Cross_Layer_Compact'
for fmt in ['png', 'pdf', 'svg']:
    dpi = 600 if fmt == 'png' else None
    fig.savefig(f'{output_base}.{fmt}', format=fmt, dpi=dpi)
    print(f"✅ Successfully saved: {output_base}.{fmt}")

plt.close(fig)