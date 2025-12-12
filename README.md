# GitHub-SocialNetwork
开放的数据集，用于分析GitHub的社交网络，重点研究Web开发者与机器学习开发者的网络结构差异。

## 项目概述
本项目对GitHub社交网络进行了全面的分析，包括网络结构特征、节点级指标、社区结构、链接预测、节点分类和影响传播等方面。通过多种分析方法，揭示了不同类型开发者在网络中的特征和差异，为理解开源社区的结构和演化提供了深入的洞察。

## 目录结构

```
GitHub-SocialNetwork/
├── analysis/              # 分析脚本和结果目录
│   ├── initial_analysis/         # 初始分析
│   │   ├── primary_analysis.py   # 初始分析代码
│   │   └── outputs/              # 初始分析结果
│   ├── structure_analysis/       # 网络结构分析
│   │   ├── structure_analysis.py # 结构分析代码
│   │   └── outputs/              # 结构分析结果
│   ├── node_level/               # 节点级分析
│   │   ├── node_level_analysis.py # 节点级分析代码
│   │   └── outputs/              # 节点级分析结果
│   ├── ergm_analysis/            # ERGM模型分析
│   │   ├── ergm_graphtool_analysis.py # ERGM分析代码
│   │   └── outputs/              # ERGM分析结果
│   ├── gnn_analysis/             # GNN网络分析
│   │   ├── gnn_network_analysis.py # GNN分析代码
│   │   └── outputs/              # GNN分析结果
│   ├── community_analysis/       # 社区分析
│   │   ├── visualize_communities.py # 社区可视化代码
│   │   ├── visualize_node_clusters.py # 聚类可视化代码
│   │   └── outputs/              # 社区分析结果
│   ├── link_prediction/          # 链接预测研究
│   │   ├── link_prediction.py    # 链接预测代码
│   │   └── outputs/              # 链接预测结果
│   ├── node_classification/      # 节点分类研究
│   │   ├── node_classification.py # 节点分类代码
│   │   └── outputs/              # 节点分类结果
│   └── influence_propagation/    # 影响传播研究
│       ├── influence_propagation.py # 影响传播代码
│       └── outputs/              # 影响传播结果
├── data/                 # 原始数据集
│   ├── musae_git_edges.csv         # 边数据
│   ├── musae_git_edges_fixed.csv   # 修复后的边数据
│   ├── musae_git_features.json     # 节点特征
│   ├── musae_git_target.csv         # 节点标签
│   └── dataset-README.txt          # 数据集说明
├── README.md             # 项目说明
├── research_guide.md     # 研究指南
└── 项目说明文档.md         # 中文项目说明
```

## 数据说明

### 节点数据 `musae_git_target.csv`
包含GitHub开发者节点信息，每行代表一个开发者：
- `id`：开发者唯一标识符
- `name`：GitHub用户名
- `ml_target`：开发者类型（0 = Web开发者，1 = 机器学习开发者）

### 边数据 `musae_git_edges_fixed.csv`
包含开发者之间的关注关系：
- `source`：关注者ID
- `target`：被关注者ID

### 特征数据 `musae_git_features.json`
包含每个节点的256维二进制特征向量。

## 安装依赖

### 基本依赖
```bash
pip install pandas numpy networkx matplotlib seaborn tqdm scikit-learn scipy igraph reportlab
```

### GNN相关依赖
对于链接预测和节点分类脚本，还需要安装PyTorch和PyTorch Geometric：
```bash
pip install torch torch_geometric
```

## 研究方向

### 1. 初始分析
- **内容**：分析Web开发者与机器学习开发者的网络结构差异
- **方法**：计算中心性指标（度中心性、介数中心性、接近中心性、特征向量中心性）
- **结果**：生成中心性指标可视化和网络分析报告

### 2. 网络结构分析
- **内容**：分析GitHub社交网络的结构特征
- **方法**：计算度分布、密度、聚类系数、平均路径长度等
- **结果**：生成度分布图表和网络结构分析报告

### 3. 节点级分析
- **内容**：计算节点级指标
- **方法**：计算PageRank、K-core（核数）和结构洞指标
- **结果**：生成节点级指标数据集

### 4. ERGM模型分析
- **内容**：使用ERGM模型拟合GitHub社交网络
- **方法**：拟合指数随机图模型
- **结果**：生成度分布和边共享伙伴分布图表

### 5. GNN网络分析
- **内容**：使用图神经网络进行网络分析
- **方法**：训练GNN模型生成节点嵌入，进行聚类分析
- **结果**：生成聚类结果和社区检测结果

### 6. 社区分析
- **内容**：检测和分析GitHub中的开发者社区
- **方法**：使用Louvain算法和K-means聚类
- **结果**：生成社区可视化图表和统计信息

### 7. 链接预测
- **内容**：基于现有网络结构预测未来可能形成的连接
- **方法**：实现5种基于相似度的方法和3种GNN方法
- **结果**：生成方法性能比较和预测链接分析

### 8. 节点分类
- **内容**：利用GNN嵌入和节点特征进行更精确的节点分类
- **方法**：比较GCN、GraphSAGE、GAT等不同GNN模型
- **结果**：生成模型性能比较和特征重要性分析

### 9. 影响传播
- **内容**：模拟信息在GitHub网络中的传播过程
- **方法**：实现SIR和IC传播模型
- **结果**：生成传播过程可视化和关键节点分析

## 使用方法

### 运行单个分析脚本
```bash
# 运行初始分析
python analysis/initial_analysis/primary_analysis.py

# 运行链接预测
python analysis/link_prediction/link_prediction.py

# 运行节点分类
python analysis/node_classification/node_classification.py

# 运行影响传播模型
python analysis/influence_propagation/influence_propagation.py
```

### 运行完整分析
参考 `research_guide.md` 文件获取详细的运行说明和参数设置。

## 结果说明

所有分析结果保存在对应分析目录的 `outputs` 文件夹中，包括：
- CSV文件：包含模型性能指标和分析结果
- PNG文件：包含可视化图表
- PDF文件：包含详细的分析报告

## 项目特点

1. **多方法支持**：实现了多种网络分析方法，包括传统方法和深度学习方法
2. **全面的可视化**：生成多种类型的可视化图表，便于结果理解
3. **模块化设计**：各分析脚本功能明确，便于扩展和修改
4. **详细的文档**：提供完整的数据集说明和分析结果
5. **可复现性**：所有分析代码和数据均可复现

## 应用场景

1. **社交网络分析**：研究GitHub开发者之间的社交关系
2. **开发者分类**：比较Web开发者与机器学习开发者的网络行为差异
3. **社区检测**：发现GitHub中的开发者社区
4. **网络结构研究**：分析开源社区的网络结构特征
5. **预测模型**：基于网络结构预测开发者类型和未来连接
6. **信息传播研究**：模拟开源项目信息的传播过程

## 许可证

本项目采用MIT许可证，详见LICENSE文件。