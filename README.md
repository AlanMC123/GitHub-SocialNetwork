# GitHub-SocialNetwork

## 项目概述
这是一个用于分析GitHub社交网络的综合性研究项目。该项目提供了完整的网络分析工具链，包括结构分析、差异分析、节点层级分析、社区检测等功能，旨在揭示GitHub开发者之间的社交关系模式和社区结构特征。

## 数据集介绍
项目使用了开放的GitHub社交网络数据集，主要包含以下文件：

- **musae_git_target.csv**: 节点表，包含每个开发者的基本信息
  - id: 开发者唯一标识符
  - name: 开发者用户名
  - ml_target: 开发者类型（0表示Web开发者，1表示机器学习开发者）
  - ...其他开发者属性

- **musae_git_edges_fixed.csv**: 边表，包含开发者之间的关注关系
  - source: 关注者ID
  - target: 被关注者ID

- **musae_git_features.json**: 每个节点的特征向量

- **dataset-README.txt**: 数据集说明文档

## 项目结构

```
GitHub-SocialNetwork/
├── code/                  # 核心分析代码
│   ├── structure_analysis.py       # 网络结构分析
│   ├── discrepancy_analysis.py     # 开发者群体差异分析
│   ├── node_level_analysis.py      # 节点层级指标分析
│   ├── visualize_communities.py    # 社区可视化
│   ├── visualize_node_clusters.py  # 节点聚类可视化
│   ├── ergm_graphtool_analysis.py  # 指数随机图模型分析
│   └── gnn_network_analysis.py     # 图神经网络分析
├── data/                  # 数据集
├── graph_structure/       # 结构分析结果和可视化
├── discrepancy_analysis/  # 差异分析结果和可视化
├── node_level/            # 节点分析结果
├── community/             # 社区检测结果
├── community_visualizations/ # 社区可视化结果
├── cluster_visualizations/   # 聚类分析可视化结果
├── ERGM/                  # 指数随机图模型分析结果
└── README.md              # 项目说明文档
```

## 安装依赖

```bash
pip install pandas igraph tqdm numpy scipy seaborn matplotlib reportlab collections concurrent.futures
```

## 分析模块详解

### 1. 结构分析 (structure_analysis.py)
该模块用于分析网络的基本结构特征，支持多种后端（networkx、igraph、cugraph）以提高计算效率。

**主要功能：**
- 计算基本网络指标：节点数、边数、平均出入度、网络密度、聚类系数、平均路径长度、直径
- 生成入度和出度分布统计（CSV格式）
- 绘制度分布的对数-对数散点图（PNG格式）
- 支持多进程加速计算

**运行方式：**
```bash
python code/structure_analysis.py
```

### 2. 差异分析 (discrepancy_analysis.py)
该模块比较Web开发者和机器学习开发者两个群体在网络中心性指标上的差异。

**主要功能：**
- 计算四种中心性指标：度中心性、接近中心性、介数中心性、特征向量中心性
- 进行t检验和Cohen's d效应量计算，评估群体差异的显著性
- 生成各中心性指标的小提琴图可视化
- 生成详细的PDF分析报告

**运行方式：**
```bash
python code/discrepancy_analysis.py
```

### 3. 节点层级分析 (node_level_analysis.py)
该模块计算每个节点的高级网络指标，揭示节点在网络中的重要性和位置特征。

**主要功能：**
- 计算PageRank值，衡量节点的影响力
- 计算K-core (coreness)，识别网络中的核心节点
- 计算结构洞指标（Burt's constraint和effective_size）
- 使用多进程加速计算，提高大规模网络的处理效率

**运行方式：**
```bash
python code/node_level_analysis.py
```

### 4. 社区检测与可视化
通过community文件夹中的结果和visualize_communities.py脚本，可以进行社区分析和可视化。

**主要功能：**
- 识别网络中的社区结构
- 生成社区大小分布、社区统计信息
- 提供多种可视化方式展示社区特征

### 5. 聚类分析
通过cluster_visualizations文件夹中的结果，可以了解节点聚类的情况。

**主要结果：**
- 聚类大小分布
- 聚类统计信息
- 各类可视化图表（散点图、箱线图、热力图等）

### 6. 指数随机图模型分析 (ERGM)
使用ergm_graphtool_analysis.py进行网络形成机制的统计建模。

**主要功能：**
- 拟合网络形成的统计模型
- 分析网络中的结构模式和形成机制
- 生成度分布和边共享伙伴分布等可视化

## 可视化结果说明

### 结构分析可视化
- **in_degree_distribution.png**：入度分布的对数-对数图
- **out_degree_distribution.png**：出度分布的对数-对数图

### 差异分析可视化
- **betweenness.png**、**closeness.png**、**degree_centrality.png**、**eigenvector.png**：各类中心性指标的分布对比图
- **network_analysis_report_igraph.pdf**：完整的分析报告

### 社区与聚类可视化
- 饼图、柱状图、热力图等多种方式展示社区和聚类特征
- 累计分布和Zipf分布分析

## 使用建议

1. **数据准备**：确保data目录下有正确的数据集文件
2. **分析顺序**：建议按照结构分析→差异分析→节点层级分析→社区检测的顺序进行
3. **可视化**：生成的可视化文件可用于进一步分析和论文撰写
4. **自定义分析**：可以修改各脚本中的参数以适应特定的分析需求

## 扩展与开发

该项目提供了基础的网络分析框架，可以通过以下方式进行扩展：

1. 添加新的网络指标计算方法
2. 实现更高级的社区检测算法
3. 整合深度学习方法进行节点分类和链接预测
4. 开发交互式可视化界面

## 注意事项

1. 大型网络分析可能需要较长时间，请耐心等待
2. 部分分析支持多进程加速，可以根据硬件情况调整参数
3. 对于GPU加速，需要安装相应的依赖（如cugraph）
4. 生成的结果文件会保存在相应的输出目录中