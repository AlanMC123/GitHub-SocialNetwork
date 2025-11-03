import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv
from torch_geometric.transforms import NormalizeFeatures
from sklearn.cluster import KMeans
import networkx as nx
from collections import defaultdict
import json
import os
import warnings
warnings.filterwarnings('ignore')

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class GNNNetworkAnalyzer:
    def __init__(self, edges_file_path):
        self.edges_file_path = edges_file_path
        self.G = None  # NetworkX图对象
        self.data = None  # PyTorch Geometric数据对象
        self.node_id_map = None  # 节点ID映射
        self.n_nodes = 0
        self.n_edges = 0
        self.embeddings = None  # 节点嵌入
        self.device = device  # 使用全局设备设置
        
        # 加载数据
        self._load_data()
        
    def _load_data(self):
        """加载边数据并构建图"""
        print(f"正在从 {self.edges_file_path} 加载数据...")
        
        # 读取边数据
        edges_df = pd.read_csv(self.edges_file_path)
        
        # 创建NetworkX图
        self.G = nx.from_pandas_edgelist(edges_df, source='source', target='target', create_using=nx.Graph())
        
        # 获取所有节点
        all_nodes = list(self.G.nodes())
        self.n_nodes = len(all_nodes)
        self.n_edges = self.G.number_of_edges()
        
        # 创建节点ID映射
        self.node_id_map = {node: idx for idx, node in enumerate(all_nodes)}
        
        # 转换为PyTorch Geometric数据格式
        edge_index = []
        for _, row in edges_df.iterrows():
            source_idx = self.node_id_map[row['source']]
            target_idx = self.node_id_map[row['target']]
            edge_index.append([source_idx, target_idx])
        
        # 创建边索引张量
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # 为节点创建简单的特征（度）
        degrees = []
        for node in all_nodes:
            degrees.append([self.G.degree(node)])
        
        x = torch.tensor(degrees, dtype=torch.float)
        
        # 创建数据对象
        self.data = Data(x=x, edge_index=edge_index)
        
        # 标准化特征
        transform = NormalizeFeatures()
        self.data = transform(self.data)
        
        # 将数据移至GPU（如果可用）
        self.data = self.data.to(self.device)
        
        print(f"数据加载完成：{self.n_nodes} 个节点，{self.n_edges} 条边")
        print("数据预处理完成，已转换为PyTorch Geometric格式")
    

    
    class GCN(torch.nn.Module):
        """图卷积网络模型"""
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x
    
    class GraphSAGE(torch.nn.Module):
        """GraphSAGE模型"""
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            from torch_geometric.nn import SAGEConv
            self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
            self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')
            
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x
    
    def train_gnn_embedding(self, model_type='GCN', embedding_dim=64, epochs=100):
        """训练GNN模型生成节点嵌入"""
        print(f"\n=== 使用{model_type}训练节点嵌入 ===")
        
        # 初始化模型
        if model_type == 'GCN':
            model = self.GCN(self.data.x.size(1), embedding_dim, embedding_dim)
        elif model_type == 'GraphSAGE':
            model = self.GraphSAGE(self.data.x.size(1), embedding_dim, embedding_dim)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 将模型移至GPU（如果可用）
        model = model.to(self.device)
        
        # 由于我们只需要节点嵌入，使用无监督训练方式
        # 我们可以使用图重构损失或对比学习损失
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 前向传播获取嵌入
            embeddings = model(self.data.x, self.data.edge_index)
            
            # 使用简单的重构损失：鼓励相邻节点的嵌入相似
            pos_pairs = self.data.edge_index.t()
            
            # 计算正样本损失（相邻节点应该相似）
            pos_similarity = F.cosine_similarity(
                embeddings[pos_pairs[:, 0]], 
                embeddings[pos_pairs[:, 1]]
            )
            pos_loss = -torch.mean(pos_similarity)
            
            # 计算负样本损失（随机节点对应该不相似）
            # 为了效率，我们采样一些负样本
            neg_samples = min(10000, self.n_nodes * 10)
            neg_source = torch.randint(0, self.n_nodes, (neg_samples,), device=self.device)
            neg_target = torch.randint(0, self.n_nodes, (neg_samples,), device=self.device)
            
            # 确保负样本不是真正的边
            edge_set = set((u.item(), v.item()) for u, v in pos_pairs)
            valid_neg_pairs = []
            for i in range(neg_samples):
                if (neg_source[i].item(), neg_target[i].item()) not in edge_set:
                    valid_neg_pairs.append(i)
                if len(valid_neg_pairs) >= neg_samples // 2:
                    break
            
            if valid_neg_pairs:
                neg_source = neg_source[valid_neg_pairs]
                neg_target = neg_target[valid_neg_pairs]
                # 确保tensor在正确的设备上
                margin = torch.tensor(0.1, device=self.device)
                neg_similarity = F.cosine_similarity(
                    embeddings[neg_source], 
                    embeddings[neg_target]
                )
                neg_loss = torch.mean(F.relu(neg_similarity + margin))  # 0.1是边距
                
                # 组合损失
                loss = pos_loss + neg_loss
            else:
                loss = pos_loss
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
        
        # 获取最终嵌入
        model.eval()
        with torch.no_grad():
            # 在GPU上计算，然后移回CPU进行numpy转换
            self.embeddings = model(self.data.x, self.data.edge_index).cpu().numpy()
        
        print(f"节点嵌入生成完成，嵌入维度: {self.embeddings.shape[1]}")
        return self.embeddings
    
    def save_community_results(self, partition, file_path='d:\\code\\socianetwork\\community\\community_results.csv'):
        """保存社区检测结果到CSV文件"""
        # 创建DataFrame保存社区结果
        community_df = pd.DataFrame(list(partition.items()), columns=['node_id', 'community_id'])
        
        # 保存到CSV
        community_df.to_csv(file_path, index=False)
        print(f"社区检测结果已保存为 {file_path}")
        return community_df
    
    def cluster_nodes(self, n_clusters=10):
        """使用K-means对节点进行聚类分析"""
        if self.embeddings is None:
            raise ValueError("请先训练GNN模型生成嵌入")
        
        print(f"\n=== 使用K-means对节点进行聚类 (k={n_clusters}) ===")
        
        # 对嵌入进行K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(self.embeddings)
        
        # 分析每个聚类的特征
        cluster_stats = defaultdict(list)
        for node_idx, label in enumerate(labels):
            original_node_id = list(self.node_id_map.keys())[node_idx]
            degree = self.G.degree(original_node_id)
            cluster_stats[label].append(degree)
        
        # 打印聚类统计信息
        print("聚类统计信息:")
        for cluster_id, degrees in cluster_stats.items():
            avg_degree = np.mean(degrees)
            size = len(degrees)
            print(f"聚类 {cluster_id}: 节点数={size}, 平均度={avg_degree:.2f}")
        
        # 保存聚类结果
        community_dir = 'd:\\code\\socianetwork\\community'
        os.makedirs(community_dir, exist_ok=True)
        save_path = os.path.join(community_dir, 'node_cluster_labels.npy')
        np.save(save_path, labels)
        print(f"聚类结果已保存为 {save_path}")
        
        return labels
    
    def find_communities(self):
        """使用Louvain算法检测社区"""
        print("\n=== 使用Louvain算法检测社区 ===")
        
        try:
            import community as community_louvain
        except ImportError:
            print("请安装python-louvain包: pip install python-louvain")
            return None
        
        # 使用Louvain算法
        partition = community_louvain.best_partition(self.G)
        
        # 分析社区
        community_stats = defaultdict(list)
        for node, community_id in partition.items():
            degree = self.G.degree(node)
            community_stats[community_id].append(degree)
        
        # 打印社区统计信息
        n_communities = len(set(partition.values()))
        print(f"检测到 {n_communities} 个社区")
        print("前10个社区统计信息:")
        
        sorted_communities = sorted(community_stats.items(), key=lambda x: len(x[1]), reverse=True)
        for i, (community_id, degrees) in enumerate(sorted_communities[:10]):
            avg_degree = np.mean(degrees)
            size = len(degrees)
            print(f"社区 {community_id}: 节点数={size}, 平均度={avg_degree:.2f}")
        
        # 保存社区结果为CSV - 使用相对路径
        import os
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        community_dir = os.path.join(BASE_DIR, '..', 'community')
        os.makedirs(community_dir, exist_ok=True)
        save_path = os.path.join(community_dir, 'community_results.csv')
        self.save_community_results(partition, save_path)
        
        return partition
    

    
    def run_complete_analysis(self):
        """运行完整的GNN分析流程，生成聚类和社区结果文件"""
      
        # 1. 训练GNN模型生成节点嵌入（使用GCN）
        self.train_gnn_embedding(model_type='GCN', epochs=50)
        
        # 2. 节点聚类分析
        self.cluster_nodes()
        
        # 3. 社区检测
        self.find_communities()
        
        print("\n=== GNN分析完成 ===")
        print("聚类结果（.npy）和社区结果（.csv）已保存到community目录。")

# 主函数
def main():
    # 数据集路径
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "..", "data", "musae_git_edges_fixed.csv")
    
    # 创建分析器实例
    analyzer = GNNNetworkAnalyzer(file_path)
    
    # 运行完整分析
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()