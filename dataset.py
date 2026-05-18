import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------- 数据准备（无人机目标尺寸分布） ----------------------
labels = ['Small', 'Tiny', 'Medium', 'Large']
sizes = [37.7, 30.8, 28.7, 2.8]

# 学术风专业配色（低饱和、高区分度，对应原分类色系）
colors = ['#d6604d', '#4393c3', '#929599', '#fdbf6f']

# 仅突出占比最低的Large类别，避免视觉干扰
explode = (0, 0, 0, 0.08)

# ---------------------- 全局学术风格配置（兼容多系统） ----------------------
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.unicode_minus': False
})

# ---------------------- 创建画布与绘图 ----------------------
fig, ax = plt.subplots(figsize=(6, 5))

# 绘制学术风环形饼图
wedges, texts, autotexts = ax.pie(
    sizes,
    explode=explode,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    pctdistance=0.82,
    labeldistance=1.1,
    shadow=False,
    startangle=315,
    wedgeprops=dict(
        width=0.3,
        edgecolor='white',
        linewidth=1.5,
        antialiased=True
    ),
    textprops=dict(
        fontweight='normal'
    )
)

# ---------------------- 文本样式精细化优化 ----------------------
# 百分比标签：黑色加粗，提升数据可读性
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

# 分类标签：统一学术样式
for text in texts:
    text.set_fontsize(10)
    text.set_fontweight('normal')

# ---------------------- 图表标注与布局 ----------------------
# 修复+更新标题（Visdrone 2019 论文专用）
ax.set_title('Distribution of Object Sizes in Visdrone 2019 Dataset', pad=20, weight='bold')

# 强制正圆，避免变形
ax.axis('equal')

# 优化布局，防止标签截断
plt.tight_layout()

# ---------------------- 导出/显示 ----------------------
# 直接保存为300DPI高清图，可直接插入论文
plt.savefig('object_size_distribution.png', dpi=300, bbox_inches='tight')
plt.show()