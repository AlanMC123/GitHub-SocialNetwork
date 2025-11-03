# GitHub-SocialNetwork
开放的数据集，用于分析GitHub的社交网络。

## 注意事项
1. ml_target列：0是Web开发者，1是机器学习开发者。
2. musae_git_features.json是每个节点的特征向量。
3. musae_git_target.csv是节点表，包含了每个节点的基本信息。
4. musae_git_edges_fixed.csv是边表，包含了所有边的信息，该表可直接导入到Gephi。

## 安装依赖
pandas, igraph, tqdm, numpy, scipy.stats, seaborn, matplotlib, reportlab, collections, concurrent.futures

## 研究步骤
### 1. 结构分析structure.analysis.py
输出节点数、边数、平均出入度、网络密度、聚类系数、平均路径长度、直径。
输出出入度分布，并进行可视化

### 2. 差异分析discrepancy_analysis.py
对比两个群体的中心性指标，进行t检验和Cohen's d计算。
中心性指标包括：度中心性、接近中心性、介数中心性、特征向量中心性。

### 3. 节点层级分析node_level_analysis.py
输出节点的PageRank、Coreness、Constraint、Effective_size。

### 4. 社区检测community_detection.py


### 5. 聚类分析clustering_analysis.py


### 6. 模块分析modularity_analysis.py


### 7. 可视化