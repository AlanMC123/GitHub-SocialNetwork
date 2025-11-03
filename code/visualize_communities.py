import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据 - 使用相对路径
file_path = '../community/community_results.csv'
try:
    df = pd.read_csv(file_path)
    print("数据加载成功！")
    print(f"数据形状: {df.shape}")
    print(f"数据前5行:\n{df.head()}")
except Exception as e:
    print(f"读取文件时出错: {e}")
    exit()

# 数据基本统计
print("\n数据基本统计:")
print(f"唯一节点数: {df['node_id'].nunique()}")
print(f"唯一社区数: {df['community_id'].nunique()}")

# 计算每个社区的节点数量
community_sizes = df['community_id'].value_counts().sort_values(ascending=False)
print(f"\n最大的10个社区大小:\n{community_sizes.head(10)}")

# 创建图形目录 - 使用相对路径
import os
output_dir = '../community_visualizations'
os.makedirs(output_dir, exist_ok=True)

# 1. 社区大小分布条形图（前30个社区）
plt.figure(figsize=(12, 8))
top_n = min(30, len(community_sizes))
sns.barplot(x=community_sizes.index[:top_n], y=community_sizes.values[:top_n])
plt.title('前30个最大社区的节点数量')
plt.xlabel('社区ID')
plt.ylabel('节点数量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_communities_bar.png'), dpi=300)
plt.close()

# 2. 社区大小分布直方图
plt.figure(figsize=(10, 6))
plt.hist(community_sizes, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('社区大小分布直方图')
plt.xlabel('社区大小')
plt.ylabel('社区数量')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'community_size_distribution.png'), dpi=300)
plt.close()

# 3. 累计分布曲线
plt.figure(figsize=(10, 6))
cumulative = np.cumsum(community_sizes.values / community_sizes.values.sum())
plt.plot(range(len(cumulative)), cumulative, marker='o', linestyle='-', markersize=2)
plt.title('社区大小累计分布')
plt.xlabel('社区排名（按大小降序）')
plt.ylabel('累计节点比例')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'cumulative_distribution.png'), dpi=300)
plt.close()

# 4. 社区大小的对数分布
plt.figure(figsize=(10, 6))
plt.loglog(range(1, len(community_sizes) + 1), community_sizes.values, 'bo', markersize=4)
plt.title('社区大小的对数分布（Zipf图）')
plt.xlabel('社区排名（按大小降序）')
plt.ylabel('社区大小')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'zipf_distribution.png'), dpi=300)
plt.close()

# 5. 饼图显示大社区的占比（前10个社区和其他）
plt.figure(figsize=(10, 10))
top_10 = community_sizes.head(10)
others = pd.Series([community_sizes[10:].sum()], index=['其他'])
sizes = pd.concat([top_10, others])

# 为了更好的可视化效果，设置颜色
colors = plt.cm.tab20(np.linspace(0, 1, len(sizes)))

plt.pie(sizes, labels=sizes.index, autopct='%1.1f%%', startangle=90, colors=colors)
plt.axis('equal')
plt.title('前10个社区和其他社区的节点占比')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'community_pie_chart.png'), dpi=300)
plt.close()

# 6. 社区大小统计信息保存到CSV
stats = pd.DataFrame({
    'community_id': community_sizes.index,
    'size': community_sizes.values,
    'percentage': (community_sizes.values / len(df) * 100).round(4)
})
stats.to_csv(os.path.join(output_dir, 'community_statistics.csv'), index=False)

print(f"\n可视化完成！")
print(f"图表保存在: {output_dir}")
print(f"统计数据保存在: {os.path.join(output_dir, 'community_statistics.csv')}")

# 显示统计摘要
print("\n社区统计摘要:")
print(f"平均社区大小: {community_sizes.mean():.2f}")
print(f"中位数社区大小: {community_sizes.median():.2f}")
print(f"标准差: {community_sizes.std():.2f}")
print(f"最大社区大小: {community_sizes.max()}")
print(f"最小社区大小: {community_sizes.min()}")
print(f"\n前5个最大的社区占总节点数的比例: {(community_sizes.head(5).sum() / len(df) * 100):.2f}%")