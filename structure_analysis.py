#!/usr/bin/env python3
"""
GitHub 社会网络基础结构分析脚本（带进度条与可视化）

新增功能：
  - 分析过程中添加进度条显示（tqdm）。
  - 自动生成度分布可视化图（保存为 degree_distribution.png）。
  - 保留路径直接指定模式，无需命令行参数。
"""

import os
import math
from collections import Counter
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

# 指定字体
plt.rcParams['font.family'] = ['Microsoft YaHei']

# ========== 用户可修改路径 ==========
NODES_PATH = r"musae_git_target.csv"  # 节点表路径
EDGES_PATH = r"musae_git_edges_fixed.csv"  # 边表路径
OUT_TXT_PATH = r"graph_structure/network_analysis.txt"  # 输出TXT路径
OUT_DEGREE_CSV = r"graph_structure/degree_distribution.csv"  # 度分布CSV路径
OUT_DEGREE_PNG = r"graph_structure/degree_distribution.png"  # 度分布可视化图路径
USE_GPU = True  # 是否优先使用GPU (cuGraph)
MAX_WORKERS = None  # NetworkX 多进程最大线程数
# ===================================

# try GPU / fast backends
_backend = None
try:
    import cudf
    import cugraph
    _backend = 'cugraph'
except Exception:
    try:
        import igraph as ig
        _backend = 'igraph'
    except Exception:
        import networkx as nx
        _backend = 'networkx'


def read_tables(nodes_path, edges_path):
    print("读取数据中...")
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    if 'id' not in nodes.columns:
        raise ValueError('节点表必须包含id列')
    if not {'source', 'target'}.issubset(edges.columns):
        raise ValueError('边表必须包含source和target列')
    print(f"节点数: {len(nodes)}, 边数: {len(edges)}")
    return nodes, edges


def build_nx_graph(nodes_df, edges_df, undirected=True):
    import networkx as nx
    G = nx.Graph() if undirected else nx.DiGraph()
    print("构建图中...")
    for _, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="添加节点"):
        G.add_node(row['id'], **{k: row[k] for k in nodes_df.columns if k != 'id'})
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="添加边"):
        G.add_edge(row['source'], row['target'])
    return G


def nx_degree_distribution(G):
    from collections import Counter
    degs = [d for n, d in G.degree()]
    c = Counter(degs)
    avg_deg = sum(degs) / len(degs)
    return sorted(c.items()), degs, avg_deg


def nx_basic_stats(G, max_workers=None):
    import networkx as nx
    print("计算网络指标中...")
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = (2 * m) / (n * (n - 1)) if n > 1 else 0
    comps = list(nx.connected_components(G))
    comp_count = len(comps)
    comp_sizes = sorted([len(c) for c in comps], reverse=True)
    largest = G.subgraph(comps[0])
    avg_clustering = nx.average_clustering(G)

    print("计算最短路径与直径中（可能较慢）...")

    def worker(u):
        lengths = nx.single_source_shortest_path_length(largest, u)
        s = sum(lengths.values())
        ecc = max(lengths.values())
        return s, ecc

    total_sum = 0
    max_ecc = 0
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)
    nodes_list = list(largest.nodes())

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(worker, u): u for u in nodes_list}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="最短路径计算"):
            s, ecc = fut.result()
            total_sum += s
            max_ecc = max(max_ecc, ecc)

    avg_path = total_sum / (largest.number_of_nodes() * (largest.number_of_nodes() - 1))
    return {
        'n_nodes': n,
        'n_edges': m,
        'density': density,
        'n_components': comp_count,
        'component_sizes': comp_sizes,
        'largest_cc_size': largest.number_of_nodes(),
        'avg_clustering': avg_clustering,
        'avg_path_length_lcc': avg_path,
        'diameter_lcc': max_ecc
    }


def save_degree_distribution(items, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(items, columns=['degree', 'count']).to_csv(out_csv, index=False)


def plot_degree_distribution(items, out_png):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    degrees, counts = zip(*items)
    plt.figure(figsize=(8, 6))
    plt.scatter(degrees, counts, s=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (对数)')
    plt.ylabel('Count (对数)')
    plt.title('Degree Distribution (Log-Log Scale)')
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_degree_distribution_by_class(G, out_png):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    degs_web = [d for n, d in G.degree() if G.nodes[n].get('ml_target') == 0]
    degs_ml = [d for n, d in G.degree() if G.nodes[n].get('ml_target') == 1]

    plt.figure(figsize=(8, 6))
    plt.hist([degs_web, degs_ml], bins=50, label=['Web开发者 (0)', 'ML开发者 (1)'], color=['#1f77b4', '#ff7f0e'], alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (对数)')
    plt.ylabel('Frequency (对数)')
    plt.title('Degree Distribution by Developer Type')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def write_results_txt(results, avg_degree, out_path, backend):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Backend: {backend}\n")
        f.write(f"节点数: {results['n_nodes']}\n")
        f.write(f"边数: {results['n_edges']}\n")
        f.write(f"平均度数: {avg_degree:.6f}\n")
        f.write(f"密度: {results['density']:.6f}\n")
        f.write(f"聚类系数: {results['avg_clustering']:.6f}\n")
        f.write(f"连通分量数: {results['n_components']}\n")
        f.write(f"最大连通子图节点数: {results['largest_cc_size']}\n")
        f.write(f"平均路径长度: {results['avg_path_length_lcc']:.6f}\n")
        f.write(f"直径: {results['diameter_lcc']}\n")


def main():
    nodes, edges = read_tables(NODES_PATH, EDGES_PATH)
    backend = _backend
    print(f"使用后端: {backend}")

    if backend == 'networkx':
        G = build_nx_graph(nodes, edges)
        stats = nx_basic_stats(G, max_workers=MAX_WORKERS)
        items, degs, avg_deg = nx_degree_distribution(G)
        save_degree_distribution(items, OUT_DEGREE_CSV)
        plot_degree_distribution(items, OUT_DEGREE_PNG)
        plot_degree_distribution_by_class(G, OUT_DEGREE_PNG_BY_CLASS)
        write_results_txt(stats, avg_deg, OUT_TXT_PATH, backend)

    elif backend == 'igraph':
        import igraph as ig
        g = ig.Graph.DataFrame(edges, directed=False, vertices=nodes)
        deg = g.degree()
        avg_deg = sum(deg) / len(deg)
        stats = {
            'n_nodes': g.vcount(),
            'n_edges': g.ecount(),
            'density': g.density(),
            'n_components': len(g.components()),
            'component_sizes': sorted(g.components().sizes(), reverse=True),
            'largest_cc_size': max(g.components().sizes()),
            'avg_clustering': g.transitivity_undirected(),
            'avg_path_length_lcc': g.average_path_length(),
            'diameter_lcc': g.diameter()
        }
        items = sorted(Counter(deg).items())
        save_degree_distribution(items, OUT_DEGREE_CSV)
        plot_degree_distribution(items, OUT_DEGREE_PNG)
        write_results_txt(stats, avg_deg, OUT_TXT_PATH, backend)

    elif backend == 'cugraph' and USE_GPU:
        import cudf, cugraph
        G = cugraph.Graph()
        G.from_cudf_edgelist(cudf.DataFrame.from_pandas(edges), source='source', destination='target', renumber=True)
        stats = {
            'n_nodes': G.number_of_vertices(),
            'n_edges': G.number_of_edges(),
            'density': (2 * G.number_of_edges()) / (G.number_of_vertices() * (G.number_of_vertices() - 1)),
            'n_components': int(cugraph.connected_components(G)['component'].nunique()),
            'component_sizes': [],
            'largest_cc_size': 0,
            'avg_clustering': float('nan'),
            'avg_path_length_lcc': float('nan'),
            'diameter_lcc': float('nan')
        }
        avg_deg = (2 * G.number_of_edges()) / G.number_of_vertices()
        write_results_txt(stats, avg_deg, OUT_TXT_PATH, backend)

    print(f"分析完成，结果已输出至 {OUT_TXT_PATH}")
    print(f"度分布图已保存至 {OUT_DEGREE_PNG} 和 {OUT_DEGREE_PNG_BY_CLASS}")


if __name__ == '__main__':
    main()
