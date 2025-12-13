# Community Analysis Results

本目录包含社区分析和节点集群分析的结果，通过多种可视化方法展示了GitHub社交网络的社区结构特征。

## 1. 社区检测结果

### 1.1 社区基本信息
- **community_results.csv**：包含社区ID、节点ID列表、社区大小等基本信息
- **community_statistics.csv**：社区统计数据，包括平均大小、最大/最小社区、社区密度等
- **node_cluster_labels.npy**：节点对应的社区标签，用于后续分析

### 1.2 社区大小分布
- **community_size_distribution.png**：社区大小的分布直方图
- **community_pie_chart.png**：主要社区占比饼图
- **top_communities_bar.png**：前N个最大社区的大小对比柱状图
- **cumulative_distribution.png**：社区大小的累积分布曲线
- **zipf_distribution.png**：社区大小的Zipf分布拟合

## 2. 节点集群分析

### 2.1 集群统计信息
- **cluster_statistics.csv**：集群的统计数据，包括平均大小、密度、中心性等

### 2.2 集群可视化
- **cluster_boxplot.png**：不同集群的特征分布箱线图
- **cluster_heatmap.png**：集群间关系热力图
- **cluster_pie_chart.png**：主要集群占比饼图
- **cluster_scatter_sample.png**：集群样本的散点图可视化
- **cluster_size_distribution.png**：集群大小的分布直方图
- **cluster_violinplot.png**：不同集群的特征分布小提琴图
- **top_clusters_bar.png**：前N个最大集群的大小对比柱状图

## 3. 结果解读

### 3.1 社区结构特征
- **幂律分布**：社区大小呈现典型的幂律分布（Zipf分布），少数大型社区包含大部分节点
- **层次结构**：网络中存在清晰的社区层次结构，从大型社区到小型子社区
- **模块化程度**：社区间连接相对稀疏，社区内连接密集，显示出较强的模块化特征

### 3.2 集群特性
- **异质性**：不同集群具有显著不同的特征分布
- **核心-边缘结构**：大型集群具有明显的核心-边缘结构
- **专业化趋势**：部分集群表现出明显的专业化特征，可能对应特定技术领域或兴趣群体

## 4. 应用价值

1. **网络结构理解**：帮助理解GitHub社交网络的层次化组织
2. **社区发现**：识别具有相似兴趣或专业背景的开发者群体
3. **信息传播**：为影响力传播研究提供社区结构基础
4. **推荐系统**：基于社区结构改进开发者推荐算法
5. **异常检测**：识别偏离正常社区结构的异常节点

## 5. 生成方法

所有结果由以下脚本生成：
- `visualize_communities.py`：社区检测和可视化
- `visualize_node_clusters.py`：节点集群分析和可视化

使用的主要算法：
- 社区检测：Louvain算法
- 节点聚类：K-means、谱聚类
- 可视化：Matplotlib、Seaborn
