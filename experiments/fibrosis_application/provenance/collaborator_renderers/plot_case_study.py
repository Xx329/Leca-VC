import matplotlib
matplotlib.use('Agg')  # 强制使用纯后台渲染引擎，必须在 import pyplot 之前！

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ... [下面的代码保持完全不变] ...

# 设置学术级绘图风格
plt.style.use('default')
sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.linewidth'] = 1.2

# ==========================================
# 1. 填入从日志中提取的真实数据
# ==========================================
# Input: 微环境信号 (为了归一化展示，筛选 0-1 范围的信号)
input_labels = [
    'TGFB_signal', 
    'inflammatory_signal', 
    'injury_signal', 
    'ECM_fibrosis',
    'fibrosis_memory\n(Internal State)' # 特别标注内部记忆
]
input_values = [0.566932, 0.525749, 0.455098, 0.349001, 0.102145]

# Output: LLM 决策程序 (挑选与肌成纤维细胞最相关的关键动作)
output_labels = [
    'ecm_deposition', 
    'survival', 
    'tgfb_response', 
    'memory_gain', 
    'homeostasis_repair'
]
output_values = [0.9, 0.8, 0.7, 0.7, 0.1]

# ==========================================
# 2. 构建双面板可视化画布
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})
fig.subplots_adjust(wspace=0.4)

# 绘制左侧面板：Input Context (深蓝色系)
y_pos_input = np.arange(len(input_labels))
colors_input = sns.color_palette("Blues_r", len(input_labels))
bars1 = ax1.barh(y_pos_input, input_values, color=colors_input, edgecolor='black', height=0.6)
ax1.set_yticks(y_pos_input)
ax1.set_yticklabels(input_labels, fontweight='bold')
ax1.invert_yaxis()  # 让重要的排在上面
ax1.set_xlim(0, 1.0)
ax1.set_title("Input: Microenvironment & Memory\n(Step 60)", fontweight='bold', pad=15)
ax1.set_xlabel("Signal Strength (0.0 - 1.0)", fontweight='bold')

# 绘制右侧面板：Output Program (深红色系，突出突变)
y_pos_output = np.arange(len(output_labels))
colors_output = sns.color_palette("Reds_r", len(output_labels))
bars2 = ax2.barh(y_pos_output, output_values, color=colors_output, edgecolor='black', height=0.6)
ax2.set_yticks(y_pos_output)
ax2.set_yticklabels(output_labels, fontweight='bold')
ax2.invert_yaxis()
ax2.set_xlim(0, 1.0)
ax2.set_title("Output: Generated Biological Program\n(LLM Decision Engine)", fontweight='bold', pad=15)
ax2.set_xlabel("Action Policy Score (0.0 - 1.0)", fontweight='bold')

# 添加数值标签
for bar in bars1:
    ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2., 
             f'{bar.get_width():.2f}', va='center', fontweight='bold')
for bar in bars2:
    ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2., 
             f'{bar.get_width():.2f}', va='center', fontweight='bold', 
             color='red' if bar.get_width() >= 0.8 else 'black')

# 去除顶部和右侧边框
sns.despine(ax=ax1)
sns.despine(ax=ax2)

# 保存高分辨率图片
plt.savefig("Leca_VC_Case_Study_Step60.pdf", bbox_inches='tight', dpi=300)
plt.savefig("Leca_VC_Case_Study_Step60.png", bbox_inches='tight', dpi=300)
print("图表已保存为 Leca_VC_Case_Study_Step60.pdf 和 .png")