#!/usr/bin/env python3
"""
GitHub社交网络链接预测研究

研究内容：基于现有网络结构预测未来可能形成的连接
技术路径：实现多种链接预测算法并比较性能
创新点：分析预测的连接是否与开发者类型相关
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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import average_precision_score
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class LinkPrediction:
    def __init__(self, data_dir):
        """初始化链接预测模型"""
        self.data_dir = data_dir
        self.nodes_df = None
        self.edges_df = None
        self.G = None
        self.train_edges = None
        self.test_edges = None
        self.test_neg_edges = None
        self.node_features = None
        self.node_id_map = None
        self.id_node_map = None
        self.device = device
        
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
        
        # 创建节点ID映射
        all_nodes = list(self.G.nodes())
        self.node_id_map = {node: idx for idx, node in enumerate(all_nodes)}
        self.id_node_map = {idx: node for node, idx in self.node_id_map.items()}
        
        print(f"节点数: {len(all_nodes)}, 边数: {self.G.number_of_edges()}")
    
    def _generate_train_test_split(self, test_size=0.2, seed=42):
        """生成训练集和测试集"""
        print(f"生成训练集和测试集，测试集比例: {test_size}")
        
        # 随机分割边为训练集和测试集
        edges = list(self.G.edges())
        train_edges, test_edges = train_test_split(edges, test_size=test_size, random_state=seed)
        
        # 创建训练图
        train_G = nx.Graph()
        train_G.add_nodes_from(self.G.nodes())
        train_G.add_edges_from(train_edges)
        
        # 生成负样本
        def generate_neg_edges(positive_edges, num_neg, G):
            """生成负样本"""
            neg_edges = []
            all_nodes = list(G.nodes())
            total_nodes = len(all_nodes)
            
            while len(neg_edges) < num_neg:
                u = np.random.choice(all_nodes)
                v = np.random.choice(all_nodes)
                if u != v and (u, v) not in G.edges() and (v, u) not in G.edges():
                    neg_edges.append((u, v))
            return neg_edges
        
        # 生成与正样本数量相等的负样本
        test_neg_edges = generate_neg_edges(test_edges, len(test_edges), train_G)
        
        self.train_edges = train_edges
        self.test_edges = test_edges
        self.test_neg_edges = test_neg_edges
        
        print(f"训练边数: {len(train_edges)}, 测试边数: {len(test_edges)}, 测试负样本数: {len(test_neg_edges)}")
        
        return train_G, train_edges, test_edges, test_neg_edges
    
    def _calculate_similarity(self, method, G, node1, node2):
        """计算两个节点之间的相似度"""
        if method == 'common_neighbors':
            # 共同邻居数量
            neighbors1 = set(G.neighbors(node1))
            neighbors2 = set(G.neighbors(node2))
            return len(neighbors1.intersection(neighbors2))
        
        elif method == 'jaccard':
            # Jaccard系数
            neighbors1 = set(G.neighbors(node1))
            neighbors2 = set(G.neighbors(node2))
            union = len(neighbors1.union(neighbors2))
            if union == 0:
                return 0.0
            return len(neighbors1.intersection(neighbors2)) / union
        
        elif method == 'adamic_adar':
            # Adamic-Adar指数
            neighbors1 = set(G.neighbors(node1))
            neighbors2 = set(G.neighbors(node2))
            common = neighbors1.intersection(neighbors2)
            if not common:
                return 0.0
            return sum(1 / np.log(G.degree(v)) for v in common if G.degree(v) > 1)
        
        elif method == 'preferential_attachment':
            # 优先连接
            return G.degree(node1) * G.degree(node2)
        
        elif method == 'resource_allocation':
            # 资源分配指数
            neighbors1 = set(G.neighbors(node1))
            neighbors2 = set(G.neighbors(node2))
            common = neighbors1.intersection(neighbors2)
            if not common:
                return 0.0
            return sum(1 / G.degree(v) for v in common if G.degree(v) > 0)
        
        else:
            raise ValueError(f"不支持的相似度方法: {method}")
    
    def similarity_based_prediction(self, method='jaccard'):
        """基于相似度的链接预测"""
        print(f"\n=== 使用{method}方法进行链接预测 ===")
        
        # 生成训练测试分割
        train_G, train_edges, test_edges, test_neg_edges = self._generate_train_test_split()
        
        # 计算测试集和负样本的相似度分数
        scores = []
        labels = []
        
        # 计算正样本分数
        print("计算正样本相似度...")
        for u, v in tqdm(test_edges, desc="正样本"):
            score = self._calculate_similarity(method, train_G, u, v)
            scores.append(score)
            labels.append(1)
        
        # 计算负样本分数
        print("计算负样本相似度...")
        for u, v in tqdm(test_neg_edges, desc="负样本"):
            score = self._calculate_similarity(method, train_G, u, v)
            scores.append(score)
            labels.append(0)
        
        # 计算评估指标
        self._evaluate(scores, labels, method)
        
        return scores, labels, method
    
    class GCNLinkPredictor(torch.nn.Module):
        """基于GCN的链接预测模型"""
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        
        def encode(self, x, edge_index):
            """生成节点嵌入"""
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x
        
        def decode(self, z, edge_label_index):
            """预测边存在概率"""
            return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        
        def forward(self, x, edge_index, edge_label_index):
            z = self.encode(x, edge_index)
            return self.decode(z, edge_label_index)
    
    class GraphSAGELinkPredictor(torch.nn.Module):
        """基于GraphSAGE的链接预测模型"""
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
            self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
        
        def encode(self, x, edge_index):
            """生成节点嵌入"""
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x
        
        def decode(self, z, edge_label_index):
            """预测边存在概率"""
            return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        
        def forward(self, x, edge_index, edge_label_index):
            z = self.encode(x, edge_index)
            return self.decode(z, edge_label_index)
    
    class GATLinkPredictor(torch.nn.Module):
        """基于GAT的链接预测模型"""
        def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
            self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        
        def encode(self, x, edge_index):
            """生成节点嵌入"""
            x = self.conv1(x, edge_index).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x
        
        def decode(self, z, edge_label_index):
            """预测边存在概率"""
            return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
        
        def forward(self, x, edge_index, edge_label_index):
            z = self.encode(x, edge_index)
            return self.decode(z, edge_label_index)
    
    def _prepare_gnn_data(self, test_size=0.2, seed=42):
        """准备GNN数据"""
        # 生成训练集和测试集
        train_G, train_edges, test_edges, test_neg_edges = self._generate_train_test_split(test_size, seed)
        
        # 创建节点特征：度特征和开发者类型特征
        features = []
        for node in self.G.nodes():
            # 度特征
            degree = train_G.degree(node)
            # 开发者类型特征
            ml_type = self.nodes_df[self.nodes_df['id'] == node]['ml_target'].values[0] if node in self.nodes_df['id'].values else 0
            features.append([degree, ml_type])
        
        x = torch.tensor(features, dtype=torch.float)
        
        # 创建训练边索引
        train_edge_index = []
        for u, v in train_edges:
            train_edge_index.append([self.node_id_map[u], self.node_id_map[v]])
        train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).t().contiguous()
        
        # 准备链接预测的正样本和负样本
        all_test_edges = test_edges + test_neg_edges
        edge_label = [1] * len(test_edges) + [0] * len(test_neg_edges)
        
        edge_label_index = []
        for u, v in all_test_edges:
            edge_label_index.append([self.node_id_map[u], self.node_id_map[v]])
        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long).t().contiguous()
        
        edge_label = torch.tensor(edge_label, dtype=torch.float)
        
        return x, train_edge_index, edge_label_index, edge_label, all_test_edges
    
    def gnn_link_prediction(self, model_type='GCN', epochs=100, lr=0.01, hidden_channels=64, out_channels=32):
        """基于GNN的链接预测"""
        print(f"\n=== 使用{model_type}进行链接预测 ===")
        
        # 准备数据
        x, train_edge_index, edge_label_index, edge_label, all_test_edges = self._prepare_gnn_data()
        
        # 初始化模型
        in_channels = x.size(1)
        if model_type == 'GCN':
            model = self.GCNLinkPredictor(in_channels, hidden_channels, out_channels)
        elif model_type == 'GraphSAGE':
            model = self.GraphSAGELinkPredictor(in_channels, hidden_channels, out_channels)
        elif model_type == 'GAT':
            model = self.GATLinkPredictor(in_channels, hidden_channels, out_channels)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 移至设备
        model = model.to(self.device)
        x = x.to(self.device)
        train_edge_index = train_edge_index.to(self.device)
        edge_label_index = edge_label_index.to(self.device)
        edge_label = edge_label.to(self.device)
        
        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        # 训练模型
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(x, train_edge_index, edge_label_index)
            loss = criterion(out, edge_label)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
        
        # 评估模型
        model.eval()
        with torch.no_grad():
            out = model(x, train_edge_index, edge_label_index)
            scores = out.sigmoid().cpu().numpy()
            labels = edge_label.cpu().numpy()
        
        # 计算评估指标
        self._evaluate(scores, labels, model_type)
        
        # 返回结果用于后续分析
        return scores, labels, model_type, all_test_edges
    
    def _evaluate(self, scores, labels, method):
        """评估链接预测结果"""
        # 计算AUC-ROC
        roc_auc = roc_auc_score(labels, scores)
        
        # 计算AUC-PR
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        
        # 计算平均精确率
        avg_precision = average_precision_score(labels, scores)
        
        print(f"{method} 评估结果:")
        print(f"AUC-ROC: {roc_auc:.4f}")
        print(f"AUC-PR: {pr_auc:.4f}")
        print(f"平均精确率: {avg_precision:.4f}")
        
        return roc_auc, pr_auc, avg_precision
    
    def analyze_predicted_links(self, scores, labels, all_test_edges, method, threshold=0.5):
        """分析预测的链接"""
        print(f"\n=== 分析 {method} 预测的链接 ===")
        
        # 根据阈值获取预测为正的链接
        predicted_pos = []
        predicted_scores = []
        for i in range(len(scores)):
            if scores[i] >= threshold:
                predicted_pos.append(all_test_edges[i])
                predicted_scores.append(scores[i])
        
        # 获取真实正链接
        true_pos = [all_test_edges[i] for i in range(len(labels)) if labels[i] == 1]
        
        # 获取预测为正的真实正链接
        correct_pred = list(set(predicted_pos) & set(true_pos))
        
        print(f"预测正链接数: {len(predicted_pos)}")
        print(f"真实正链接数: {len(true_pos)}")
        print(f"正确预测数: {len(correct_pred)}")
        
        # 分析预测的链接与开发者类型的关系
        def get_link_type(u, v):
            """获取链接类型"""
            u_type = self.nodes_df[self.nodes_df['id'] == u]['ml_target'].values[0] if u in self.nodes_df['id'].values else -1
            v_type = self.nodes_df[self.nodes_df['id'] == v]['ml_target'].values[0] if v in self.nodes_df['id'].values else -1
            return (u_type, v_type)
        
        # 统计链接类型分布
        link_types = defaultdict(int)
        predicted_links_with_info = []
        for i, (u, v) in enumerate(predicted_pos):
            link_type = get_link_type(u, v)
            link_types[link_type] += 1
            
            # 添加链接信息
            type_str = "其他"
            if link_type == (0, 0):
                type_str = "Web-Web"
            elif link_type in [(0, 1), (1, 0)]:
                type_str = "Web-ML"
            elif link_type == (1, 1):
                type_str = "ML-ML"
            
            predicted_links_with_info.append({
                'source': u,
                'target': v,
                'score': predicted_scores[i],
                'type': type_str
            })
        
        print(f"预测的链接类型分布:")
        for link_type, count in link_types.items():
            if link_type == (0, 0):
                print(f"  Web-Web: {count} ({count/len(predicted_pos)*100:.2f}%)")
            elif link_type == (0, 1) or link_type == (1, 0):
                print(f"  Web-ML: {count} ({count/len(predicted_pos)*100:.2f}%)")
            elif link_type == (1, 1):
                print(f"  ML-ML: {count} ({count/len(predicted_pos)*100:.2f}%)")
            else:
                print(f"  其他: {count} ({count/len(predicted_pos)*100:.2f}%)")
        
        # 保存预测链接结果到文件
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存所有预测结果
        all_prediction_df = pd.DataFrame({
            'source': [edge[0] for edge in all_test_edges],
            'target': [edge[1] for edge in all_test_edges],
            'score': scores,
            'true_label': labels,
            'predicted_label': [1 if s >= threshold else 0 for s in scores]
        })
        all_prediction_file = os.path.join(output_dir, f'{method}_all_predictions.csv')
        all_prediction_df.to_csv(all_prediction_file, index=False)
        print(f"所有预测结果已保存到 {all_prediction_file}")
        
        # 保存预测为正的链接
        if predicted_links_with_info:
            predicted_df = pd.DataFrame(predicted_links_with_info)
            predicted_file = os.path.join(output_dir, f'{method}_predicted_links.csv')
            predicted_df.to_csv(predicted_file, index=False)
            print(f"预测为正的链接已保存到 {predicted_file}")
        
        return link_types
    
    def run_all_methods(self):
        """运行所有链接预测方法"""
        print("运行所有链接预测方法...")
        
        # 基于相似度的方法
        similarity_methods = ['common_neighbors', 'jaccard', 'adamic_adar', 'preferential_attachment', 'resource_allocation']
        results = {}
        
        for method in similarity_methods:
            scores, labels, _ = self.similarity_based_prediction(method)
            results[method] = (scores, labels)
        
        # GNN方法
        gnn_methods = ['GCN', 'GraphSAGE', 'GAT']
        for method in gnn_methods:
            scores, labels, _, all_test_edges = self.gnn_link_prediction(method)
            results[method] = (scores, labels, method, all_test_edges)
        
        return results
    
    def compare_methods(self, results):
        """比较不同方法的性能"""
        print("\n=== 比较不同方法的性能 ===")
        
        # 计算各方法的评估指标
        performance = []
        for method, data in results.items():
            if len(data) == 2:
                scores, labels = data
                all_test_edges = None
            else:
                scores, labels, _, all_test_edges = data
            
            roc_auc = roc_auc_score(labels, scores)
            precision, recall, _ = precision_recall_curve(labels, scores)
            pr_auc = auc(recall, precision)
            avg_precision = average_precision_score(labels, scores)
            
            performance.append({
                '方法': method,
                'AUC-ROC': roc_auc,
                'AUC-PR': pr_auc,
                '平均精确率': avg_precision
            })
        
        # 转换为DataFrame
        df_performance = pd.DataFrame(performance)
        print(df_performance.sort_values(by='AUC-ROC', ascending=False))
        
        # 保存结果
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'link_prediction_results')
        os.makedirs(output_dir, exist_ok=True)
        
        performance_path = os.path.join(output_dir, 'method_performance.csv')
        df_performance.to_csv(performance_path, index=False)
        print(f"性能比较结果已保存到 {performance_path}")
        
        return df_performance

# 主函数
def main():
    # 获取数据目录
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 从当前脚本目录向上两级到达项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
    data_dir = os.path.join(PROJECT_ROOT, "data")
    
    print(f"脚本目录: {BASE_DIR}")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {data_dir}")
    
    # 创建链接预测实例
    lp = LinkPrediction(data_dir)
    
    # 运行所有方法
    results = lp.run_all_methods()
    
    # 比较不同方法
    lp.compare_methods(results)
    
    # 分析GAT模型的预测结果
    gat_scores, gat_labels, gat_method, all_test_edges = results['GAT']
    lp.analyze_predicted_links(gat_scores, gat_labels, all_test_edges, 'GAT')
    
    print("\n=== 链接预测研究完成 ===")

if __name__ == "__main__":
    main()