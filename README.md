# GitHub-SocialNetwork
开放的数据集，用于分析GitHub的社交网络。

## 注意事项
1. ml_target列：0是Web开发者，1是机器学习开发者。
2. musae_git_features.json是每个节点的特征向量。

## 安装依赖
pandas, igraph, tqdm, numpy, scipy.stats, seaborn, matplotlib, reportlab, collections, concurrent.futures

## 研究步骤
### 1. 初始分析 primary_analysis.py
该程序分析Web开发者与机器学习开发者的网络结构差异。
结果保存在primary_analysis文件夹。

### 2. 结构分析 structure_analysis.py
该程序分析GitHub社交网络的结构特征，如度分布、密度、聚类系数等等。结果保存在graph_structure文件夹。

### 3. 模型拟合 model_fitting.py
该程序使用ERGM模型拟合GitHub社交网络，分析不同开发者类型之间的联系。结果保存在ERGM文件夹。
随后使用