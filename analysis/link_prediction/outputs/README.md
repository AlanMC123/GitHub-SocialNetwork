# Link Prediction Results

本目录包含使用多种算法对GitHub社交网络进行链接预测的结果，包括传统相似性算法和GNN模型。

## 1. 链接预测算法概述

链接预测的目标是预测网络中可能存在但尚未观察到的连接。本次分析使用了以下两类算法：

### 1.1 相似性算法
- 共同邻居（Common Neighbors）
- Jaccard系数
- Adamic-Adar指数
- 优先连接（Preferential Attachment）
- 资源分配指数

### 1.2 GNN模型
- Graph Attention Network（GAT）

## 2. 分析结果

### 2.1 GAT模型预测结果
- **GAT_all_predictions.csv**：GAT模型对所有可能边的预测分数，包括源节点ID、目标节点ID和预测概率
- **GAT_predicted_links.csv**：根据预测概率筛选出的最可能存在的链接，包括源节点、目标节点和预测概率

### 2.2 方法性能比较
- **method_performance.csv**：不同链接预测方法的性能指标比较，包括AUC、准确率、精确率、召回率、F1分数等

## 3. 结果解读

### 3.1 GAT模型预测质量
- GAT模型能够捕捉到网络的复杂结构信息，生成高质量的链接预测
- GAT_predicted_links.csv中包含的链接具有较高的可信度，可用于实际应用

### 3.2 算法性能对比
- 从method_performance.csv可以看出不同算法的优缺点
- GNN模型（如GAT）通常比传统相似性算法表现更好，尤其是在复杂网络中
- 传统相似性算法在计算效率上具有优势，适合大规模网络的快速预测

## 4. 应用价值

1. **开发者推荐**：基于预测的链接为开发者推荐潜在的合作对象
2. **网络演化预测**：预测社交网络的未来发展趋势
3. **异常检测**：识别异常的链接模式
4. **信息传播优化**：基于预测的链接结构优化信息传播策略

## 5. 生成方法

所有结果由`link_prediction.py`脚本生成，使用以下技术栈：
- **传统算法**：NetworkX
- **GNN模型**：PyTorch Geometric
- **评估指标**：scikit-learn

## 6. 使用说明

### 6.1 预测链接使用
- GAT_predicted_links.csv包含了按预测概率排序的潜在链接
- 可以根据实际需求筛选不同置信度阈值的链接

### 6.2 方法选择参考
- 对于大规模网络，优先考虑传统相似性算法
- 对于需要高精度预测的场景，推荐使用GNN模型
- 根据method_performance.csv中的指标选择最适合特定任务的算法
