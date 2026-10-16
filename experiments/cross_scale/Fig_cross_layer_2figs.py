import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('overall_results.csv')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 1.8
plt.rcParams['lines.markersize'] = 6

color_r2 = '#000000'
color_rmse = '#4682B4'
color_vline = '#999999'

fig1, ax1 = plt.subplots(figsize=(4, 3.5))

ax1.plot(df['Samples_Per_Layer'], df['R2'], 
         marker='o', linestyle='-',
         color=color_r2, linewidth=1.8, 
         label='$R^2$ score')

ax1.set_ylabel('$R^2$ score', color=color_r2, fontsize=11, fontweight='bold')
ax1.set_xlabel('Samples per layer', fontsize=11, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_r2, direction='out', length=4)
ax1.set_ylim(0.55, 1.02)
ax1.set_xlim(15, 315)
ax1.set_xticks([25, 50, 100, 200, 300])

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.axvline(x=200, color=color_vline, linestyle='--', linewidth=1.2, alpha=0.8)

ax1.legend(loc='upper left', fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig('figure_r2.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_r2.pdf', format='pdf', bbox_inches='tight')
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(4, 3.5))

ax2.plot(df['Samples_Per_Layer'], df['RMSE'], 
         marker='s', linestyle='--',
         color=color_rmse, linewidth=1.8, 
         label='RMSE')

ax2.set_ylabel('RMSE', color=color_rmse, fontsize=11, fontweight='bold')
ax2.set_xlabel('Samples per layer', fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_rmse, direction='out', length=4)
ax2.set_ylim(0.015, 0.050)
ax2.set_xlim(15, 315)
ax2.set_xticks([25, 50, 100, 200, 300])

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.axvline(x=200, color=color_vline, linestyle='--', linewidth=1.2, alpha=0.8)

ax2.legend(loc='upper right', fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig('figure_rmse.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_rmse.pdf', format='pdf', bbox_inches='tight')
plt.close(fig2)