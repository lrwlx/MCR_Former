import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ==========================================
# 1. 设置 Nature 期刊全局样式参数
# ==========================================
# 字体设置：优先使用 Arial，若无则使用 Helvetica 或 DejaVu Sans
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# 关键：强制 PDF 使用 TrueType 字体 (Type 42)，Nature 投稿强制要求
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 字号设置 - 适当增大，与上方子图视觉权重匹配（Nature 最低要求 7pt）
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9      # 轴标签 9pt（原 7pt → 增大）
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8     # 刻度标签 8pt（原 6pt → 增大）
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 6     # 图例 8pt（原 6pt → 增大）

# 线条与刻度线宽 - 全面加粗，解决"太浅"问题
plt.rcParams['axes.linewidth'] = 1.0        # 坐标轴线宽 1.0 pt（原 0.5 → 加倍）
plt.rcParams['xtick.major.width'] = 1.0     # 刻度线宽 1.0 pt（原 0.5 → 加倍）
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['lines.linewidth'] = 1.5       # 数据线宽 1.5 pt（原 1.0 → 加粗）
plt.rcParams['lines.markersize'] = 5        # 标记点 5 pt（原 4 → 增大）

# ==========================================
# 2. 数据读取与画布初始化
# ==========================================
df = pd.read_csv('layer_wise_results.csv')

# Nature 单栏图标准宽度 89 mm (~3.5 英寸)，此处适当加宽以容纳更大字体
fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=300)

# ==========================================
# 3. 数据绘制（色盲安全配色 + 形状区分）
# ==========================================
colors = ['#000000', '#E69F00', '#56B4E9']  # Okabe-Ito 色盲安全配色
markers = ['o', 's', '^']
samples = sorted(df['Samples_Per_Layer'].unique())

for i, sample in enumerate(samples):
    subset = df[df['Samples_Per_Layer'] == sample]
    ax.plot(subset['Layer'], subset['R2'],
            color=colors[i],
            marker=markers[i],
            markersize=5,               # 标记点增大（原 3.5 → 5）
            markeredgewidth=1.0,        # 【新增】标记点边框加粗，使轮廓更清晰
            linewidth=1.5,              # 数据线加粗（原 1.0 → 1.5）
            markevery=2,
            label=f'{sample} samples/layer')

# ==========================================
# 4. 坐标轴与细节美化
# ==========================================
# 去除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 【关键修改】加粗左侧和底部边框，与上方子图一致
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)

# 刻度线朝外，加宽加粗
ax.tick_params(axis='both', which='major',
               direction='out', length=3, width=1.0, pad=2)

# 设置 X 轴刻度为整数
ax.set_xticks(range(0, 21, 5))

# 【关键修改】坐标轴标签使用粗体，与上方子图匹配
ax.set_xlabel('Number of layers', fontweight='bold', fontsize=9, fontstyle='normal')
ax.set_ylabel('$R^2$ score', fontweight='bold', fontsize=9, fontstyle='normal')

# Y 轴范围与刻度
ax.set_ylim(0.72, 1.01)
ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))

# 【关键修改】刻度标签设为粗体
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(8)
    label.set_fontstyle('normal')

# ==========================================
# 5. 图例设置 - 粗体，与上方子图匹配
# ==========================================
leg = ax.legend(frameon=False, loc='lower left', borderaxespad=0.5)
for text in leg.get_texts():
    text.set_fontweight('bold')    # 图例文字加粗
    text.set_fontsize(8)
    text.set_fontstyle('normal')

# ==========================================
# 6. 导出与保存
# ==========================================
plt.tight_layout()

# 保存为矢量图 PDF（投稿首选）
plt.savefig('Nature_Fig_R2.pdf', format='pdf', bbox_inches='tight', dpi=300)
# 保存为高清晰度 PNG（备用）
plt.savefig('Nature_Fig_R2.png', format='png', bbox_inches='tight', dpi=600)

plt.show()