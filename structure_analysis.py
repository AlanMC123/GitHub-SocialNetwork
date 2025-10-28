#!/usr/bin/env python3
"""
GitHub 社会网络结构分析（有向图版）

功能：
  - 支持 igraph / networkx / cugraph 后端。
  - 输出入度、出度分布（CSV + PNG）。
  - 输出基本网络结构指标。
"""

import os
from collections import Counter
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

# 指定中文字体
plt.rcParams['font.family'] = ['Microsoft YaHei']

# ========== 用户可修改路径 ==========
NODES_PATH = r"musae_git_target.csv"  # 节点表路径
EDGES_PATH = r"musae_git_edges_fixed.csv"  # 边表路径
OUT_TXT_PATH = r"graph_structure/network_analysis.txt"  # 输出TXT路径
OUT_IN_DEGREE_CSV = r"graph_structure/in_degree_distribution.csv"
OUT_OUT_DEGREE_CSV = r"graph_structure/out_degree_distribution.csv"
OUT_IN_DEGREE_PNG = r"graph_structure/in_degree_distribution.png"
OUT_OUT_DEGREE_PNG = r"graph_structure/out_degree_distribution.png"
USE_GPU = True  # 是否优先使用GPU (cuGraph)
MAX_WORKERS = None  # 多进程线程数
# ===================================

# 尝试加载不同后端
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


# ========== 基础函数 ==========
def read_tables(nodes_path, edges_path):
    print("读取数据中...")
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    if 'id' not in nodes.columns:
        raise ValueError('节点表必须包含 id 列')
    if not {'source', 'target'}.issubset(edges.columns):
        raise ValueError('边表必须包含 source 和 target 列')
    print(f"节点数: {len(nodes)}, 边数: {len(edges)}")
    return nodes, edges


def build_nx_graph(nodes_df, edges_df):
    import networkx as nx
    G = nx.DiGraph()
    print("构建有向图中...")
    for _, row in tqdm(nodes_df.iterrows(), total=len(nodes_df), desc="添加节点"):
        G.add_node(row['id'], **{k: row[k] for k in nodes_df.columns if k != 'id'})
    for _, row in tqdm(edges_df.iterrows(), total=len(edges_df), desc="添加边"):
        G.add_edge(row['source'], row['target'])
    return G


# ========== 有向图专用度分布 ==========
def nx_degree_distribution_directed(G):
    indeg = [d for n, d in G.in_degree()]
    outdeg = [d for n, d in G.out_degree()]

    indeg_items = sorted(Counter(indeg).items())
    outdeg_items = sorted(Counter(outdeg).items())

    avg_indeg = sum(indeg) / len(indeg)
    avg_outdeg = sum(outdeg) / len(outdeg)

    return indeg_items, outdeg_items, avg_indeg, avg_outdeg


# ========== 基本网络指标 ==========
def nx_basic_stats(G, max_workers=None):
    import networkx as nx
    print("计算网络指标中...")

    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G)

    # 使用无向图来计算聚类、路径等指标
    GU = G.to_undirected()

    comps = list(nx.connected_components(GU))
    comp_count = len(comps)
    comp_sizes = sorted([len(c) for c in comps], reverse=True)
    largest = GU.subgraph(comps[0])
    avg_clustering = nx.average_clustering(GU)

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


# ========== 保存与绘图 ==========
def save_degree_distribution(items, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(items, columns=['degree', 'count']).to_csv(out_csv, index=False)


def plot_degree_distribution(items, out_png, title):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    degrees, counts = zip(*items)
    plt.figure(figsize=(8, 6))
    plt.scatter(degrees, counts, s=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Degree (对数)')
    plt.ylabel('Count (对数)')
    plt.title(title)
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def write_results_txt(results, avg_in, avg_out, out_path, backend):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"Backend: {backend}\n")
        f.write(f"节点数: {results['n_nodes']}\n")
        f.write(f"边数: {results['n_edges']}\n")
        f.write(f"平均入度: {avg_in:.6f}\n")
        f.write(f"平均出度: {avg_out:.6f}\n")
        f.write(f"网络密度: {results['density']:.6f}\n")
        f.write(f"聚类系数: {results['avg_clustering']:.6f}\n")
        f.write(f"连通分量数: {results['n_components']}\n")
        f.write(f"最大连通子图节点数: {results['largest_cc_size']}\n")
        f.write(f"平均路径长度: {results['avg_path_length_lcc']:.6f}\n")
        f.write(f"直径: {results['diameter_lcc']}\n")


# ========== 主程序入口 ==========
def main():
    nodes, edges = read_tables(NODES_PATH, EDGES_PATH)
    backend = _backend
    print(f"使用后端: {backend}")

    if backend == 'networkx':
        G = build_nx_graph(nodes, edges)
        stats = nx_basic_stats(G, max_workers=MAX_WORKERS)
        indeg_items, outdeg_items, avg_in, avg_out = nx_degree_distribution_directed(G)

        save_degree_distribution(indeg_items, OUT_IN_DEGREE_CSV)
        save_degree_distribution(outdeg_items, OUT_OUT_DEGREE_CSV)

        plot_degree_distribution(indeg_items, OUT_IN_DEGREE_PNG, 'In-Degree Distribution (Log-Log)')
        plot_degree_distribution(outdeg_items, OUT_OUT_DEGREE_PNG, 'Out-Degree Distribution (Log-Log)')

        write_results_txt(stats, avg_in, avg_out, OUT_TXT_PATH, backend)

    elif backend == 'igraph':
        import igraph as ig
        g = ig.Graph.DataFrame(edges, directed=True, vertices=nodes)
        indeg = g.indegree()
        outdeg = g.outdegree()
        avg_in = sum(indeg) / len(indeg)
        avg_out = sum(outdeg) / len(outdeg)
        indeg_items = sorted(Counter(indeg).items())
        outdeg_items = sorted(Counter(outdeg).items())

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

        save_degree_distribution(indeg_items, OUT_IN_DEGREE_CSV)
        save_degree_distribution(outdeg_items, OUT_OUT_DEGREE_CSV)
        plot_degree_distribution(indeg_items, OUT_IN_DEGREE_PNG, 'In-Degree Distribution (Log-Log)')
        plot_degree_distribution(outdeg_items, OUT_OUT_DEGREE_PNG, 'Out-Degree Distribution (Log-Log)')
        write_results_txt(stats, avg_in, avg_out, OUT_TXT_PATH, backend)

    elif backend == 'cugraph' and USE_GPU:
        import cudf, cugraph
        G = cugraph.DiGraph()
        G.from_cudf_edgelist(cudf.DataFrame.from_pandas(edges), source='source', destination='target', renumber=True)
        stats = {
            'n_nodes': G.number_of_vertices(),
            'n_edges': G.number_of_edges(),
            'density': G.number_of_edges() / (G.number_of_vertices() * (G.number_of_vertices() - 1)),
            'n_components': 0,
            'component_sizes': [],
            'largest_cc_size': 0,
            'avg_clustering': float('nan'),
            'avg_path_length_lcc': float('nan'),
            'diameter_lcc': float('nan')
        }
        avg_in = avg_out = G.number_of_edges() / G.number_of_vertices()
        write_results_txt(stats, avg_in, avg_out, OUT_TXT_PATH, backend)

    print("分析完成 ✅")
    print(f"结果已输出至: {OUT_TXT_PATH}")
    print(f"入度分布图: {OUT_IN_DEGREE_PNG}")
    print(f"出度分布图: {OUT_OUT_DEGREE_PNG}")


if __name__ == '__main__':
    main()
