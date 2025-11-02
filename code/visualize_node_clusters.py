import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取npy文件 - 使用相对路径
file_path = '../community/node_cluster_labels.npy'
try:
    cluster_labels = np.load(file_path)
    print("数据加载成功！")
    print(f"数据形状: {cluster_labels.shape}")
    print(f"数据类型: {cluster_labels.dtype}")
    print(f"前10个聚类标签: {cluster_labels[:10]}")
except Exception as e:
    print(f"读取文件时出错: {e}")
    exit()

# 数据基本统计
print("\n数据基本统计:")
print(f"节点总数: {len(cluster_labels)}")
print(f"唯一聚类数: {len(np.unique(cluster_labels))}")
print(f"聚类标签范围: {cluster_labels.min()} - {cluster_labels.max()}")

# 计算每个聚类的节点数量
cluster_counts = Counter(cluster_labels)
sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
print(f"\n最大的10个聚类大小:")
for cluster_id, count in sorted_clusters[:10]:
    print(f"聚类 {cluster_id}: {count} 个节点")

# 创建图形目录 - 使用相对路径
output_dir = '../cluster_visualizations'
os.makedirs(output_dir, exist_ok=True)

# 转换为DataFrame以便处理
import pandas as pd
df_clusters = pd.DataFrame({
    'node_index': range(len(cluster_labels)),
    'cluster_id': cluster_labels
})

# 1. 聚类大小分布条形图（前30个聚类）
plt.figure(figsize=(12, 8))
top_n = min(30, len(sorted_clusters))
cluster_ids, counts = zip(*sorted_clusters[:top_n])
sns.barplot(x=list(cluster_ids), y=list(counts))
plt.title('前30个最大聚类的节点数量')
plt.xlabel('聚类ID')
plt.ylabel('节点数量')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top_clusters_bar.png'), dpi=300)
plt.close()

# 2. 聚类大小分布直方图
plt.figure(figsize=(10, 6))
sizes = [count for _, count in sorted_clusters]
plt.hist(sizes, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('聚类大小分布直方图')
plt.xlabel('聚类大小')
plt.ylabel('聚类数量')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'cluster_size_distribution.png'), dpi=300)
plt.close()

# 3. 累计分布曲线
plt.figure(figsize=(10, 6))
cumulative = np.cumsum(sizes) / np.sum(sizes)
plt.plot(range(len(cumulative)), cumulative, marker='o', linestyle='-', markersize=2)
plt.title('聚类大小累计分布')
plt.xlabel('聚类排名（按大小降序）')
plt.ylabel('累计节点比例')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'cumulative_distribution.png'), dpi=300)
plt.close()

# 4. 聚类大小的对数分布（Zipf图）
plt.figure(figsize=(10, 6))
plt.loglog(range(1, len(sizes) + 1), sizes, 'bo', markersize=4)
plt.title('聚类大小的对数分布（Zipf图）')
plt.xlabel('聚类排名（按大小降序）')
plt.ylabel('聚类大小')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'zipf_distribution.png'), dpi=300)
plt.close()

# 5. 饼图显示大聚类的占比（前10个聚类和其他）
plt.figure(figsize=(10, 10))
top_10_sizes = sizes[:10]
others = sum(sizes[10:])
pie_sizes = top_10_sizes + [others]
pie_labels = [f'聚类 {i}' for i in cluster_ids[:10]] + ['其他']

# 为了更好的可视化效果，设置颜色
colors = plt.cm.tab20(np.linspace(0, 1, len(pie_sizes)))

plt.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.axis('equal')
plt.title('前10个聚类和其他聚类的节点占比')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cluster_pie_chart.png'), dpi=300)
plt.close()

# 6. 聚类分布的箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(y=sizes)
plt.title('聚类大小分布箱线图')
plt.ylabel('聚类大小')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'cluster_boxplot.png'), dpi=300)
plt.close()

# 7. 聚类大小的小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(y=sizes)
plt.title('聚类大小分布小提琴图')
plt.ylabel('聚类大小')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'cluster_violinplot.png'), dpi=300)
plt.close()

# 8. 聚类标签分布散点图（前1000个节点作为示例）
plt.figure(figsize=(12, 8))
sample_size = min(1000, len(df_clusters))
sample_df = df_clusters.sample(sample_size, random_state=42)
plt.scatter(sample_df['node_index'], sample_df['cluster_id'], alpha=0.6, s=10)
plt.title('节点聚类标签分布（随机采样1000个节点）')
plt.xlabel('节点索引')
plt.ylabel('聚类ID')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'cluster_scatter_sample.png'), dpi=300)
plt.close()

# 9. 聚类大小统计信息保存到CSV
stats = pd.DataFrame({
    'cluster_id': [cid for cid, _ in sorted_clusters],
    'size': sizes,
    'percentage': (np.array(sizes) / len(cluster_labels) * 100).round(4)
})
stats.to_csv(os.path.join(output_dir, 'cluster_statistics.csv'), index=False)

# 10. 聚类大小的热力图矩阵（如果聚类数量适中）
if len(sorted_clusters) <= 50:  # 限制聚类数量以避免图表过大
    plt.figure(figsize=(12, 10))
    # 创建一个简单的热力图数据（这里只是聚类大小的可视化）
    size_matrix = np.array(sizes[:50]).reshape(-1, 1)
    sns.heatmap(size_matrix, annot=True, fmt='d', cmap='viridis', 
                xticklabels=['聚类大小'], yticklabels=[f'聚类 {i}' for i, _ in sorted_clusters[:50]])
    plt.title('聚类大小热力图（前50个聚类）')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_heatmap.png'), dpi=300)
    plt.close()

print(f"\n可视化完成！")
print(f"图表保存在: {output_dir}")
print(f"统计数据保存在: {os.path.join(output_dir, 'cluster_statistics.csv')}")

# 显示统计摘要
print("\n聚类统计摘要:")
print(f"平均聚类大小: {np.mean(sizes):.2f}")
print(f"中位数聚类大小: {np.median(sizes):.2f}")
print(f"标准差: {np.std(sizes):.2f}")
print(f"最大聚类大小: {np.max(sizes)}")
print(f"最小聚类大小: {np.min(sizes)}")
print(f"\n前5个最大的聚类占总节点数的比例: {(np.sum(sizes[:5]) / len(cluster_labels) * 100):.2f}%")
print(f"\n聚类大小分布的四分位数:")
print(f"第一四分位数 (Q1): {np.percentile(sizes, 25):.2f}")
print(f"第二四分位数 (Q2/中位数): {np.percentile(sizes, 50):.2f}")
print(f"第三四分位数 (Q3): {np.percentile(sizes, 75):.2f}")