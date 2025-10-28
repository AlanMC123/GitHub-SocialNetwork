"""
Social Network Analysis script using python-igraph + multiprocessing

功能：
1. 从指定路径读取节点表(nodes.csv) 和 边表(edges.csv)
   - nodes.csv 需要包含: id,name,ml_target(可选)
   - edges.csv 需要包含: source,target[,weight]
2. 使用 igraph 构建图（可选 directed）
3. 使用多线程/多进程加速对节点级指标的计算（结构洞计算部分采用 multiprocessing）
4. 计算 PageRank、K-core (coreness)、以及结构洞指标（Burt 的 constraint 与 effective_size）
5. 输出结果到 results.csv

依赖：
pip install igraph numpy pandas
"""

import pandas as pd
import numpy as np
import igraph as ig
from multiprocessing import Pool, cpu_count
from functools import partial

# ===================== 用户配置区 =====================
NODES_FILE = 'musae_git_target.csv'   # 节点表路径
EDGES_FILE = 'musae_git_edges_fixed.csv'   # 边表路径
OUTPUT_FILE = 'node_level/results.csv'     # 输出路径
DIRECTED = True                 # 是否为有向图
PROCESSES = None                 # 并行进程数（默认 cpu_count()-1）
# =====================================================

def build_graph(nodes_df, edges_df, directed=False):
    unique_ids = pd.Index(nodes_df['id'].astype(str).unique())
    id2idx = {nid: i for i, nid in enumerate(unique_ids)}

    g = ig.Graph(directed=directed)
    g.add_vertices(len(unique_ids))
    g.vs['id'] = unique_ids.tolist()

    if 'name' in nodes_df.columns:
        id_to_name = nodes_df.set_index(nodes_df['id'].astype(str))['name'].to_dict()
        g.vs['name'] = [id_to_name.get(nid, '') for nid in unique_ids]
    if 'ml_target' in nodes_df.columns:
        id_to_ml = nodes_df.set_index(nodes_df['id'].astype(str))['ml_target'].astype(int).to_dict()
        g.vs['ml_target'] = [int(id_to_ml.get(nid, -1)) for nid in unique_ids]

    edge_tuples = []
    weights = []
    weighted = 'weight' in edges_df.columns
    for _, row in edges_df.iterrows():
        s = str(row['source'])
        t = str(row['target'])
        if s not in id2idx or t not in id2idx:
            continue
        edge_tuples.append((id2idx[s], id2idx[t]))
        if weighted:
            weights.append(float(row['weight']))

    g.add_edges(edge_tuples)
    g.es['weight'] = weights if weighted else [1.0] * g.ecount()

    return g

def compute_pagerank(g, directed=False, weight_attr='weight'):
    if weight_attr in g.es.attribute_names():
        pr = g.pagerank(directed=directed, weights=g.es[weight_attr])
    else:
        pr = g.pagerank(directed=directed)
    return pr

def compute_kcore(g):
    return g.coreness()

def node_neighbor_props(g, v_idx):
    neighbors = g.neighbors(v_idx, mode='all')
    if not neighbors:
        return {}
    total = 0.0
    neigh_w = {}
    for nb in neighbors:
        eid = g.get_eid(v_idx, nb, directed=False)
        w = float(g.es[eid]['weight']) if 'weight' in g.es.attribute_names() else 1.0
        neigh_w[nb] = w
        total += w
    return {nb: neigh_w[nb] / total for nb in neighbors} if total > 0 else {nb: 0.0 for nb in neighbors}

def compute_structural_holes_for_node(g, v_idx):
    neighbors = g.neighbors(v_idx, mode='all')
    if not neighbors:
        return {'node': g.vs[v_idx]['id'], 'constraint': 0.0, 'effective_size': 0.0, 'degree': 0}

    p_i = node_neighbor_props(g, v_idx)
    m = {q: node_neighbor_props(g, q) for q in neighbors}

    constraint_total = 0.0
    for j in neighbors:
        pij = p_i.get(j, 0.0)
        indirect = sum(p_i.get(q, 0.0) * m.get(q, {}).get(j, 0.0) for q in neighbors)
        constraint_total += (pij + indirect) ** 2

    eff = 0.0
    for j in neighbors:
        redundancy = sum(p_i.get(q, 0.0) * m.get(q, {}).get(j, 0.0) for q in neighbors)
        eff += (1 - redundancy)

    return {'node': g.vs[v_idx]['id'], 'constraint': constraint_total, 'effective_size': eff, 'degree': len(neighbors)}

def compute_structural_holes(g, processes=None):
    if processes is None:
        processes = max(1, cpu_count() - 1)
    idxs = list(range(g.vcount()))
    with Pool(processes=processes) as pool:
        func = partial(compute_structural_holes_for_node, g)
        results = pool.map(func, idxs)
    return results

def main():
    nodes_df = pd.read_csv(NODES_FILE, dtype=str)
    edges_df = pd.read_csv(EDGES_FILE, dtype=str)
    if 'weight' in edges_df.columns:
        edges_df['weight'] = edges_df['weight'].astype(float)

    g = build_graph(nodes_df, edges_df, directed=DIRECTED)
    print(f"Graph built: {g.vcount()} nodes, {g.ecount()} edges. Directed={DIRECTED}")

    pr = compute_pagerank(g, directed=DIRECTED, weight_attr='weight')
    coreness = compute_kcore(g)
    print("PageRank 和 K-core 计算完成。")

    print("开始计算结构洞（多进程）...")
    sh_results = compute_structural_holes(g, processes=PROCESSES)

    vid_to_id = g.vs['id']
    df = pd.DataFrame({'id': vid_to_id, 'pagerank': pr, 'coreness': coreness})

    sh_df = pd.DataFrame(sh_results).set_index('node')
    sh_df.index = sh_df.index.astype(str)
    df = df.set_index('id').join(sh_df).reset_index()

    if 'name' in g.vs.attribute_names():
        df = df.merge(nodes_df[['id', 'name']].drop_duplicates(), on='id', how='left')
    if 'ml_target' in g.vs.attribute_names():
        df = df.merge(nodes_df[['id', 'ml_target']].drop_duplicates(), on='id', how='left')

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"结果已保存至 {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
