#!/usr/bin/env python3
"""
GitHub社交网络节点分类研究

研究内容：利用GNN嵌入和节点特征进行更精确的节点分类
技术路径：比较GCN、GraphSAGE、GAT等不同GNN模型的分类性能
创新点：研究特征重要性，确定哪些特征对区分Web开发者和机器学习开发者最有帮助
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class NodeClassification:
    def __init__(self, data_dir):
        """初始化节点分类模型"""
        self.data_dir = data_dir
        self.nodes_df = None
        self.edges_df = None
        self.G = None
        self.node_features = None
        self.node_labels = None
        self.node_id_map = None
        self.id_node_map = None
        self.device = device
        self.features_df = None
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载节点和边数据"""
        print(f"正在从 {self.data_dir} 加载数据...")
        
        # 读取节点数据
        nodes_path = os.path.join(self.data_dir, 'musae_git_target.csv')
        self.nodes_df = pd.read_csv(nodes_path)
        
        # 读取边数据
        edges_path = os.path.join(self.data_dir, 'musae_git_edges_fixed.csv')
        self.edges_df = pd.read_csv(edges_path)
        
        # 创建NetworkX图
        self.G = nx.from_pandas_edgelist(self.edges_df, source='source', target='target', create_using=nx.Graph())
        
        # 读取特征数据
        features_path = os.path.join(self.data_dir, 'musae_git_features.json')
        import json
        with open(features_path, 'r') as f:
            features_data = json.load(f)
        
        # 转换特征数据为DataFrame
        features_list = []
        for node, features in features_data.items():
            node_id = int(node)
            feature_dict = {f'feature_{i}': 1 if i in features else 0 for i in range(256)}  # 256维特征
            feature_dict['id'] = node_id
            features_list.append(feature_dict)
        
        self.features_df = pd.DataFrame(features_list)
        
        # 合并节点数据和特征数据
        self.nodes_df = pd.merge(self.nodes_df, self.features_df, on='id', how='left')
        
        # 创建节点ID映射
        all_nodes = list(self.G.nodes())
        self.node_id_map = {node: idx for idx, node in enumerate(all_nodes)}
        self.id_node_map = {idx: node for node, idx in self.node_id_map.items()}
        
        print(f"节点数: {len(all_nodes)}, 边数: {self.G.number_of_edges()}")
        print(f"特征维度: {self.features_df.shape[1] - 1}")
    
    def _prepare_node_features(self, include_degree=True, include_basic=True):
        """准备节点特征"""
        print(f"准备节点特征，include_degree: {include_degree}, include_basic: {include_basic}")
        
        # 基础特征：256维二进制特征
        feature_cols = [col for col in self.nodes_df.columns if col.startswith('feature_')]
        features = self.nodes_df[['id'] + feature_cols].copy()
        
        # 添加度特征
        if include_degree:
            degrees = dict(self.G.degree())
            features['degree'] = features['id'].map(degrees)
        
        # 确保所有节点都有特征
        all_nodes_df = pd.DataFrame({'id': list(self.G.nodes())})
        features = pd.merge(all_nodes_df, features, on='id', how='left').fillna(0)
        
        # 按节点ID映射排序
        features['node_idx'] = features['id'].map(self.node_id_map)
        features = features.sort_values('node_idx').reset_index(drop=True)
        
        # 提取特征矩阵
        feature_cols = [col for col in features.columns if col not in ['id', 'node_idx']]
        X = features[feature_cols].values
        
        # 提取标签
        labels = []
        for node in features['id']:
            ml_target = self.nodes_df[self.nodes_df['id'] == node]['ml_target'].values[0] if node in self.nodes_df['id'].values else 0
            labels.append(ml_target)
        y = np.array(labels)
        
        return X, y, feature_cols
    
    def _prepare_gnn_data(self, test_size=0.2, seed=42, include_degree=True, include_basic=True):
        """准备GNN数据"""
        print(f"准备GNN数据，测试集比例: {test_size}")
        
        # 准备节点特征
        X, y, feature_cols = self._prepare_node_features(include_degree, include_basic)
        
        # 创建边索引
        edge_index = []
        for u, v in self.G.edges():
            edge_index.append([self.node_id_map[u], self.node_id_map[v]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # 创建节点特征张量
        x = torch.tensor(X, dtype=torch.float)
        
        # 创建标签张量
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # 划分训练集、验证集和测试集
        num_nodes = x.size(0)
        indices = torch.randperm(num_nodes)
        test_size = int(num_nodes * test_size)
        val_size = int(num_nodes * 0.1)  # 10%作为验证集
        
        test_indices = indices[:test_size]
        val_indices = indices[test_size:test_size+val_size]
        train_indices = indices[test_size+val_size:]
        
        # 创建掩码
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        return Data(x=x, edge_index=edge_index, y=y_tensor, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask), X, y, feature_cols
    
    class GCNClassifier(torch.nn.Module):
        """基于GCN的节点分类模型"""
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x
    
    class GraphSAGEClassifier(torch.nn.Module):
        """基于GraphSAGE的节点分类模型"""
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
            self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x
    
    class GATClassifier(torch.nn.Module):
        """基于GAT的节点分类模型"""
        def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
            self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x
    
    def gnn_node_classification(self, model_type='GCN', epochs=100, lr=0.01, hidden_channels=64, test_size=0.2):
        """基于GNN的节点分类"""
        print(f"\n=== 使用{model_type}进行节点分类 ===")
        
        # 准备数据
        data, X, y, feature_cols = self._prepare_gnn_data(test_size=test_size)
        data = data.to(self.device)
        
        # 初始化模型
        in_channels = data.x.size(1)
        out_channels = 2  # Web开发者 vs ML开发者
        
        if model_type == 'GCN':
            model = self.GCNClassifier(in_channels, hidden_channels, out_channels)
        elif model_type == 'GraphSAGE':
            model = self.GraphSAGEClassifier(in_channels, hidden_channels, out_channels)
        elif model_type == 'GAT':
            model = self.GATClassifier(in_channels, hidden_channels, out_channels)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 移至设备
        model = model.to(self.device)
        
        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        # 训练模型
        best_val_acc = 0
        best_model = None
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            # 验证模型
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    pred = out.argmax(dim=1)
                    
                    train_acc = accuracy_score(data.y[data.train_mask].cpu(), pred[data.train_mask].cpu())
                    val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu())
                    
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_model = model.state_dict()
                
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, 训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}')
                model.train()
        
        # 加载最佳模型
        model.load_state_dict(best_model)
        
        # 测试模型
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            test_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
            test_precision = precision_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
            test_recall = recall_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
            test_f1 = f1_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
        
        # 打印测试结果
        print(f"\n{model_type} 测试结果:")
        print(f"准确率: {test_acc:.4f}")
        print(f"精确率: {test_precision:.4f}")
        print(f"召回率: {test_recall:.4f}")
        print(f"F1分数: {test_f1:.4f}")
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu(), target_names=['Web开发者', 'ML开发者']))
        
        return {
            'model_type': model_type,
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'predictions': pred.cpu().numpy(),
            'true_labels': data.y.cpu().numpy(),
            'test_mask': data.test_mask.cpu().numpy()
        }
    
    def analyze_feature_importance(self):
        """分析特征重要性"""
        print("\n=== 分析特征重要性 ===")
        
        # 准备数据
        X, y, feature_cols = self._prepare_node_features(include_degree=True, include_basic=True)
        
        # 使用随机森林进行特征重要性分析
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # 获取特征重要性
        importances = rf.feature_importances_
        feature_importance = pd.DataFrame({'feature': feature_cols, 'importance': importances})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # 打印前20个重要特征
        print("\n前20个重要特征:")
        top_features = feature_importance.head(20)
        print(top_features)
        
        # 可视化特征重要性
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('特征重要性排名（前20）')
        plt.tight_layout()
        
        # 保存可视化结果
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'node_classification_results')
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        
        # 保存特征重要性结果
        feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        print(f"\n特征重要性结果已保存到 {os.path.join(output_dir, 'feature_importance.csv')}")
        
        return feature_importance
    
    def compare_gnn_models(self, epochs=100, lr=0.01, hidden_channels=64, test_size=0.2):
        """比较不同GNN模型的性能"""
        print("\n=== 比较不同GNN模型的性能 ===")
        
        # 运行所有GNN模型
        models = ['GCN', 'GraphSAGE', 'GAT']
        results = []
        
        for model_type in models:
            result = self.gnn_node_classification(model_type, epochs, lr, hidden_channels, test_size)
            results.append(result)
        
        # 转换为DataFrame
        df_results = pd.DataFrame(results)
        df_results = df_results[['model_type', 'accuracy', 'precision', 'recall', 'f1']]
        
        # 打印比较结果
        print("\n不同GNN模型性能比较:")
        print(df_results.sort_values(by='accuracy', ascending=False))
        
        # 保存比较结果
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'node_classification_results')
        os.makedirs(output_dir, exist_ok=True)
        
        df_results.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
        print(f"\n模型比较结果已保存到 {os.path.join(output_dir, 'model_comparison.csv')}")
        
        return df_results
    
    def run_all_experiments(self):
        """运行所有实验"""
        print("运行所有节点分类实验...")
        
        # 比较不同GNN模型
        self.compare_gnn_models()
        
        # 分析特征重要性
        self.analyze_feature_importance()
        
        print("\n=== 节点分类研究完成 ===")

# 主函数
def main():
    # 获取数据目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "..", "data")
    
    # 创建节点分类实例
    nc = NodeClassification(data_dir)
    
    # 运行所有实验
    nc.run_all_experiments()

if __name__ == "__main__":
    main()
