import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 尝试导入graph-tool，如果失败提供友好提示
try:
    import graph_tool.all as gt
    GRAPH_TOOL_AVAILABLE = True
except ImportError:
    print("警告: graph-tool包未安装。graph-tool主要支持Linux系统，在Windows上安装复杂。")
    print("请考虑在Linux环境中运行此脚本，或使用WSL (Windows Subsystem for Linux)。")
    print("将使用NetworkX进行基本分析，但高级ERGM功能将不可用。")
    import networkx as nx
    GRAPH_TOOL_AVAILABLE = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class NetworkAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.G = None
        self.df = None
        self.use_graphtool = GRAPH_TOOL_AVAILABLE
        
    def load_network_data(self):
        """读取并加载网络数据"""
        print("正在读取网络数据...")
        self.df = pd.read_csv(self.file_path)
        print(f"数据加载完成，共有 {len(self.df)} 条边")
        
        if self.use_graphtool:
            # 使用graph-tool构建图
            self.G = gt.Graph(directed=False)
            # 确保节点ID是连续的整数
            all_nodes = set(self.df['source']).union(set(self.df['target']))
            node_id_map = {node: i for i, node in enumerate(sorted(all_nodes))}
            
            # 添加边
            edges = []
            for _, row in self.df.iterrows():
                edges.append((node_id_map[row['source']], node_id_map[row['target']]))
            
            self.G.add_edge_list(edges)
            print(f"使用graph-tool构建网络完成，共有 {self.G.num_vertices()} 个节点和 {self.G.num_edges()} 条边")
            self.node_id_map = node_id_map
        else:
            # 使用NetworkX作为备选
            self.G = nx.Graph()
            for _, row in self.df.iterrows():
                self.G.add_edge(row['source'], row['target'])
            print(f"使用NetworkX构建网络完成，共有 {self.G.number_of_nodes()} 个节点和 {self.G.number_of_edges()} 条边")
    
    def compute_basic_stats(self):
        """计算基本网络统计量"""
        print("\n=== 基本网络统计量 ===")
        
        if self.use_graphtool:
            n_nodes = self.G.num_vertices()
            n_edges = self.G.num_edges()
            print(f"节点数: {n_nodes}")
            print(f"边数: {n_edges}")
            
            # 密度
            density = 2 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0
            print(f"网络密度: {density:.6f}")
            
            # 度分布
            degrees = self.G.degree_property_map("total").a
            avg_degree = np.mean(degrees)
            print(f"平均度: {avg_degree:.4f}")
            print(f"最大度: {np.max(degrees)}")
            print(f"最小度: {np.min(degrees)}")
            print(f"度的标准差: {np.std(degrees):.4f}")
            
            # 连通性
            comp, hist = gt.label_components(self.G)
            n_components = len(hist)
            print(f"连通分量数: {n_components}")
            
            if hist:
                largest_cc_size = max(hist)
                print(f"最大连通分量大小: {largest_cc_size} ({largest_cc_size/n_nodes*100:.2f}%)")
            
            return {
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'density': density,
                'avg_degree': avg_degree,
                'max_degree': np.max(degrees),
                'min_degree': np.min(degrees),
                'degree_std': np.std(degrees),
                'n_components': n_components
            }
        else:
            # NetworkX实现
            n_nodes = self.G.number_of_nodes()
            n_edges = self.G.number_of_edges()
            print(f"节点数: {n_nodes}")
            print(f"边数: {n_edges}")
            
            density = nx.density(self.G)
            print(f"网络密度: {density:.6f}")
            
            degrees = [d for n, d in self.G.degree()]
            avg_degree = np.mean(degrees)
            print(f"平均度: {avg_degree:.4f}")
            print(f"最大度: {max(degrees)}")
            print(f"最小度: {min(degrees)}")
            print(f"度的标准差: {np.std(degrees):.4f}")
            
            n_components = nx.number_connected_components(self.G)
            print(f"连通分量数: {n_components}")
            
            largest_cc = max(nx.connected_components(self.G), key=len)
            print(f"最大连通分量大小: {len(largest_cc)} ({len(largest_cc)/n_nodes*100:.2f}%)")
            
            return {
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'density': density,
                'avg_degree': avg_degree,
                'max_degree': max(degrees),
                'min_degree': min(degrees),
                'degree_std': np.std(degrees),
                'n_components': n_components
            }
    
    def compute_triangle_statistics(self):
        """计算三角形相关统计量 (triangle效应)"""
        print("\n=== 三角形统计量 (Triangle Effect) ===")
        
        if self.use_graphtool:
            # 使用graph-tool计算三角形
            triangles, _ = gt.triangles(self.G)
            total_triangles = sum(triangles)
            print(f"三角形数量: {total_triangles:.0f}")
            
            # 传递性
            transitivity = gt.global_clustering(self.G)
            print(f"传递性: {transitivity:.6f}")
            
            # 平均聚类系数
            avg_clustering = gt.local_clustering(self.G).mean()
            print(f"平均聚类系数: {avg_clustering:.6f}")
            
            return {
                'triangles': total_triangles,
                'transitivity': transitivity,
                'avg_clustering': avg_clustering
            }
        else:
            # NetworkX实现
            triangles = nx.triangles(self.G)
            total_triangles = sum(triangles.values()) / 3  # 每个三角形被计算三次
            print(f"三角形数量: {total_triangles:.0f}")
            
            transitivity = nx.transitivity(self.G)
            print(f"传递性: {transitivity:.6f}")
            
            avg_clustering = nx.average_clustering(self.G)
            print(f"平均聚类系数: {avg_clustering:.6f}")
            
            return {
                'triangles': total_triangles,
                'transitivity': transitivity,
                'avg_clustering': avg_clustering
            }
    
    def compute_concurrent_statistics(self):
        """计算并发效应统计量 (concurrent effect)"""
        print("\n=== 并发效应统计量 (Concurrent Effect) ===")
        
        if self.use_graphtool:
            # 度分布的高阶矩作为并发效应的近似
            degrees = self.G.degree_property_map("total").a
            
            # 计算2-星 (degree^2)
            two_star = np.sum(degrees * (degrees - 1) / 2)
            print(f"2-星统计量: {two_star:.0f}")
            
            # 计算3-星 (degree^3)
            three_star = np.sum(degrees * (degrees - 1) * (degrees - 2) / 6)
            print(f"3-星统计量: {three_star:.0f}")
            
            # 度分布的方差
            degree_var = np.var(degrees)
            print(f"度分布方差: {degree_var:.4f}")
            
            # 度分布的偏度
            degree_skew = ((degrees - np.mean(degrees)) ** 3).mean() / (degree_var ** 1.5)
            print(f"度分布偏度: {degree_skew:.4f}")
            
            return {
                'two_star': two_star,
                'three_star': three_star,
                'degree_variance': degree_var,
                'degree_skewness': degree_skew
            }
        else:
            # NetworkX实现
            degrees = [d for n, d in self.G.degree()]
            
            two_star = sum(d * (d - 1) / 2 for d in degrees)
            print(f"2-星统计量: {two_star:.0f}")
            
            three_star = sum(d * (d - 1) * (d - 2) / 6 for d in degrees)
            print(f"3-星统计量: {three_star:.0f}")
            
            degree_var = np.var(degrees)
            print(f"度分布方差: {degree_var:.4f}")
            
            degree_skew = ((np.array(degrees) - np.mean(degrees)) ** 3).mean() / (degree_var ** 1.5)
            print(f"度分布偏度: {degree_skew:.4f}")
            
            return {
                'two_star': two_star,
                'three_star': three_star,
                'degree_variance': degree_var,
                'degree_skewness': degree_skew
            }
    
    def compute_edgesharedpartners_statistics(self):
        """计算边共享伙伴统计量 (edgesharedpartners effect)"""
        print("\n=== 边共享伙伴统计量 (Edge Shared Partners Effect) ===")
        
        esp_values = []
        esp_counts = defaultdict(int)
        
        if self.use_graphtool:
            # graph-tool实现
            # 采样部分边进行计算，避免大图计算过慢
            num_edges = self.G.num_edges()
            sample_size = min(5000, num_edges)  # 最多采样5000条边
            
            print(f"采样 {sample_size} 条边计算共享伙伴...")
            
            # 随机选择边
            edges = list(self.G.edges())
            if sample_size < num_edges:
                np.random.shuffle(edges)
                edges = edges[:sample_size]
            
            # 为每个节点创建邻居集合
            neighbors_dict = {v: set() for v in range(self.G.num_vertices())}
            for v, w in edges:
                neighbors_dict[v].add(w)
                neighbors_dict[w].add(v)
            
            # 计算每对相邻节点的共享伙伴数
            for v, w in edges:
                shared = len(neighbors_dict[v] & neighbors_dict[w]) - 1  # 减去彼此
                esp_values.append(shared)
                esp_counts[shared] += 1
        else:
            # NetworkX实现
            # 采样部分边进行计算
            num_edges = self.G.number_of_edges()
            sample_size = min(5000, num_edges)
            
            print(f"采样 {sample_size} 条边计算共享伙伴...")
            
            edges = list(self.G.edges())
            if sample_size < num_edges:
                np.random.shuffle(edges)
                edges = edges[:sample_size]
            
            # 计算每对相邻节点的共享伙伴数
            for u, v in edges:
                neighbors_u = set(self.G.neighbors(u))
                neighbors_v = set(self.G.neighbors(v))
                shared = len(neighbors_u & neighbors_v) - 1  # 减去彼此
                esp_values.append(shared)
                esp_counts[shared] += 1
        
        # 计算统计量
        if esp_values:
            avg_esp = np.mean(esp_values)
            std_esp = np.std(esp_values)
            max_esp = max(esp_values)
            min_esp = min(esp_values)
            
            print(f"平均边共享伙伴数: {avg_esp:.4f} ± {std_esp:.4f}")
            print(f"最大边共享伙伴数: {max_esp}")
            print(f"最小边共享伙伴数: {min_esp}")
            
            # 绘制共享伙伴分布
            plt.figure(figsize=(10, 6))
            sorted_esp = sorted(esp_counts.keys())
            sorted_counts = [esp_counts[k] for k in sorted_esp]
            plt.bar(sorted_esp[:20], sorted_counts[:20])  # 只显示前20个值
            plt.title('边共享伙伴数分布 (前20个值)')
            plt.xlabel('共享伙伴数')
            plt.ylabel('频数')
            plt.grid(True, alpha=0.3)
            plt.savefig('edgesharedpartners_distribution.png', dpi=300, bbox_inches='tight')
            print("边共享伙伴分布图已保存为 edgesharedpartners_distribution.png")
            
            return {
                'avg_esp': avg_esp,
                'std_esp': std_esp,
                'max_esp': max_esp,
                'min_esp': min_esp
            }
        else:
            print("未找到边或无法计算共享伙伴")
            return None
    
    def analyze_degree_distribution(self):
        """分析度分布"""
        print("\n=== 度分布分析 ===")
        
        if self.use_graphtool:
            degrees = self.G.degree_property_map("total").a
        else:
            degrees = [d for n, d in self.G.degree()]
        
        # 计算度分布
        degree_counts = defaultdict(int)
        for d in degrees:
            degree_counts[d] += 1
        
        # 排序
        degrees_sorted = sorted(degree_counts.keys())
        frequencies = [degree_counts[d]/len(degrees) for d in degrees_sorted]
        
        # 绘制度分布图
        plt.figure(figsize=(12, 6))
        
        # 线性图
        plt.subplot(1, 2, 1)
        plt.scatter(degrees_sorted, frequencies, alpha=0.6)
        plt.title('度分布 (线性尺度)')
        plt.xlabel('度')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        
        # 对数-对数图
        plt.subplot(1, 2, 2)
        plt.loglog(degrees_sorted, frequencies, 'o', alpha=0.6)
        plt.title('度分布 (对数-对数尺度)')
        plt.xlabel('度 (log)')
        plt.ylabel('频率 (log)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('degree_distribution.png', dpi=300, bbox_inches='tight')
        print("度分布图已保存为 degree_distribution.png")
        
        return degree_counts
    
    def run_analysis(self):
        """运行完整分析"""
        # 加载数据
        self.load_network_data()
        
        # 基本统计量
        basic_stats = self.compute_basic_stats()
        
        # Triangle效应
        triangle_stats = self.compute_triangle_statistics()
        
        # Concurrent效应
        concurrent_stats = self.compute_concurrent_statistics()
        
        # Edge Shared Partners效应
        esp_stats = self.compute_edgesharedpartners_statistics()
        
        # 度分布分析
        degree_dist = self.analyze_degree_distribution()
        
        print("\n=== 分析总结 ===")
        print(f"1. 网络规模: {basic_stats['n_nodes']} 节点, {basic_stats['n_edges']} 边")
        print(f"2. 网络密度: {basic_stats['density']:.6f}")
        print(f"3. 三角形统计: {triangle_stats['triangles']:.0f} 个三角形, 传递性 {triangle_stats['transitivity']:.4f}")
        print(f"4. 平均聚类系数: {triangle_stats['avg_clustering']:.4f}")
        print(f"5. 度分布方差: {concurrent_stats['degree_variance']:.4f}")
        if esp_stats:
            print(f"6. 平均边共享伙伴数: {esp_stats['avg_esp']:.4f}")
        
        print("\n图表已保存，可供进一步分析。")
        print(f"注意: {'使用graph-tool进行分析' if self.use_graphtool else 'graph-tool不可用，使用NetworkX进行基本分析'}")

# 主函数
def main():
    file_path = 'd:\\code\\socianetwork\\musae_git_edges_fixed.csv'
    
    # 创建分析器实例
    analyzer = NetworkAnalyzer(file_path)
    
    # 运行分析
    analyzer.run_analysis()

if __name__ == "__main__":
    main()