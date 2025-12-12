#!/usr/bin/env python3
"""
GitHub社交网络影响传播研究

研究内容：模拟信息在GitHub网络中的传播过程
技术路径：实现SIR、IC等传播模型，分析不同类型开发者在信息传播中的作用
创新点：研究关键节点对信息传播的影响，为开源项目推广提供策略建议
"""

import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class InfluencePropagation:
    def __init__(self, data_dir):
        """初始化影响传播模型"""
        self.data_dir = data_dir
        self.nodes_df = None
        self.edges_df = None
        self.G = None
        self.node_types = None
        self.device = 'cpu'  # 传播模型主要使用NetworkX，CPU足够
        
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
        self.G = nx.from_pandas_edgelist(self.edges_df, source='source', target='target', create_using=nx.DiGraph())
        
        # 创建节点类型字典
        self.node_types = dict(zip(self.nodes_df['id'], self.nodes_df['ml_target']))
        
        print(f"节点数: {self.G.number_of_nodes()}, 边数: {self.G.number_of_edges()}")
        print(f"ML开发者数: {sum(self.node_types.values())}, Web开发者数: {len(self.node_types) - sum(self.node_types.values())}")
    
    def sir_model(self, beta=0.3, gamma=0.1, initial_infected=10, max_steps=100, seed=42):
        """
        实现SIR传播模型
        
        参数:
        - beta: 感染率
        - gamma: 恢复率
        - initial_infected: 初始感染节点数
        - max_steps: 最大模拟步数
        - seed: 随机种子
        
        返回:
        - results: 传播结果
        """
        print(f"\n=== 运行SIR模型，beta={beta}, gamma={gamma}, initial_infected={initial_infected} ===")
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 初始化节点状态
        states = {}
        for node in self.G.nodes():
            states[node] = 'S'  # 初始均为易感状态
        
        # 随机选择初始感染节点
        all_nodes = list(self.G.nodes())
        initial_nodes = random.sample(all_nodes, initial_infected)
        for node in initial_nodes:
            states[node] = 'I'  # 初始感染节点
        
        # 记录传播过程
        results = {
            'step': [],
            'S': [],
            'I': [],
            'R': [],
            'I_ml': [],  # ML开发者感染数
            'I_web': [],  # Web开发者感染数
            'R_ml': [],  # ML开发者恢复数
            'R_web': []   # Web开发者恢复数
        }
        
        # 计算初始状态
        self._record_state(states, results, 0)
        
        # 模拟传播过程
        for step in tqdm(range(1, max_steps + 1), desc="SIR模拟"):
            new_states = states.copy()
            
            # 处理感染节点
            infected_nodes = [node for node, state in states.items() if state == 'I']
            for node in infected_nodes:
                # 感染邻居
                neighbors = list(self.G.neighbors(node))
                for neighbor in neighbors:
                    if states[neighbor] == 'S' and random.random() < beta:
                        new_states[neighbor] = 'I'
                
                # 恢复
                if random.random() < gamma:
                    new_states[node] = 'R'
            
            states = new_states
            
            # 记录当前状态
            self._record_state(states, results, step)
            
            # 终止条件：没有感染节点
            if sum(1 for state in states.values() if state == 'I') == 0:
                print(f"传播在第{step}步终止，没有感染节点")
                break
        
        # 转换为DataFrame
        df_results = pd.DataFrame(results)
        
        # 保存结果
        self._save_results(df_results, 'sir')
        
        # 可视化结果
        self._visualize_propagation(df_results, 'SIR')
        
        return df_results
    
    def ic_model(self, p=0.3, initial_infected=10, max_steps=100, seed=42):
        """
        实现独立级联(IC)传播模型
        
        参数:
        - p: 传播概率
        - initial_infected: 初始感染节点数
        - max_steps: 最大模拟步数
        - seed: 随机种子
        
        返回:
        - results: 传播结果
        """
        print(f"\n=== 运行IC模型，p={p}, initial_infected={initial_infected} ===")
        
        random.seed(seed)
        np.random.seed(seed)
        
        # 初始化节点状态
        states = {}
        for node in self.G.nodes():
            states[node] = 'S'  # 初始均为易感状态
        
        # 随机选择初始感染节点
        all_nodes = list(self.G.nodes())
        initial_nodes = random.sample(all_nodes, initial_infected)
        for node in initial_nodes:
            states[node] = 'I'  # 初始感染节点
        
        # 记录传播过程
        results = {
            'step': [],
            'S': [],
            'I': [],
            'I_ml': [],  # ML开发者感染数
            'I_web': [],  # Web开发者感染数
        }
        
        # 计算初始状态
        self._record_ic_state(states, results, 0)
        
        # 模拟传播过程
        for step in tqdm(range(1, max_steps + 1), desc="IC模拟"):
            new_infections = set()
            
            # 处理感染节点
            infected_nodes = [node for node, state in states.items() if state == 'I']
            for node in infected_nodes:
                # 感染邻居
                neighbors = list(self.G.neighbors(node))
                for neighbor in neighbors:
                    if states[neighbor] == 'S' and random.random() < p:
                        new_infections.add(neighbor)
            
            # 更新状态
            for node in new_infections:
                states[node] = 'I'
            
            # 记录当前状态
            self._record_ic_state(states, results, step)
            
            # 终止条件：没有新感染节点
            if len(new_infections) == 0:
                print(f"传播在第{step}步终止，没有新感染节点")
                break
        
        # 转换为DataFrame
        df_results = pd.DataFrame(results)
        
        # 保存结果
        self._save_results(df_results, 'ic')
        
        # 可视化结果
        self._visualize_propagation(df_results, 'IC')
        
        return df_results
    
    def _record_state(self, states, results, step):
        """记录SIR模型的当前状态"""
        s_count = sum(1 for state in states.values() if state == 'S')
        i_count = sum(1 for state in states.values() if state == 'I')
        r_count = sum(1 for state in states.values() if state == 'R')
        
        # 统计ML和Web开发者的状态
        i_ml = sum(1 for node, state in states.items() if state == 'I' and self.node_types.get(node, 0) == 1)
        i_web = i_count - i_ml
        r_ml = sum(1 for node, state in states.items() if state == 'R' and self.node_types.get(node, 0) == 1)
        r_web = r_count - r_ml
        
        results['step'].append(step)
        results['S'].append(s_count)
        results['I'].append(i_count)
        results['R'].append(r_count)
        results['I_ml'].append(i_ml)
        results['I_web'].append(i_web)
        results['R_ml'].append(r_ml)
        results['R_web'].append(r_web)
    
    def _record_ic_state(self, states, results, step):
        """记录IC模型的当前状态"""
        s_count = sum(1 for state in states.values() if state == 'S')
        i_count = sum(1 for state in states.values() if state == 'I')
        
        # 统计ML和Web开发者的状态
        i_ml = sum(1 for node, state in states.items() if state == 'I' and self.node_types.get(node, 0) == 1)
        i_web = i_count - i_ml
        
        results['step'].append(step)
        results['S'].append(s_count)
        results['I'].append(i_count)
        results['I_ml'].append(i_ml)
        results['I_web'].append(i_web)
    
    def analyze_key_nodes(self, method='degree', top_k=20):
        """
        分析关键节点
        
        参数:
        - method: 关键节点评估方法 ('degree', 'betweenness', 'pagerank')
        - top_k: 前k个关键节点
        
        返回:
        - top_nodes: 前k个关键节点
        """
        print(f"\n=== 分析关键节点，方法: {method}，top_k: {top_k} ===")
        
        # 计算节点重要性
        if method == 'degree':
            importance = dict(self.G.in_degree())
        elif method == 'betweenness':
            importance = nx.betweenness_centrality(self.G)
        elif method == 'pagerank':
            importance = nx.pagerank(self.G)
        else:
            raise ValueError(f"不支持的关键节点评估方法: {method}")
        
        # 排序获取前k个关键节点
        sorted_nodes = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_nodes = sorted_nodes[:top_k]
        
        # 统计关键节点类型
        ml_count = 0
        for node, score in top_nodes:
            if self.node_types.get(node, 0) == 1:
                ml_count += 1
        
        print(f"前{top_k}个关键节点中，ML开发者: {ml_count}个，Web开发者: {top_k - ml_count}个")
        
        # 打印前10个节点
        print("\n前10个关键节点:")
        for node, score in top_nodes[:10]:
            dev_type = "ML开发者" if self.node_types.get(node, 0) == 1 else "Web开发者"
            print(f"节点 {node}: 分数={score:.4f}, 类型={dev_type}")
        
        # 保存结果
        df_top_nodes = pd.DataFrame(top_nodes, columns=['node_id', 'score'])
        df_top_nodes['type'] = df_top_nodes['node_id'].map(lambda x: 'ML' if self.node_types.get(x, 0) == 1 else 'Web')
        
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'influence_propagation_results')
        os.makedirs(output_dir, exist_ok=True)
        
        df_top_nodes.to_csv(os.path.join(output_dir, f'key_nodes_{method}.csv'), index=False)
        print(f"\n关键节点结果已保存到 {os.path.join(output_dir, f'key_nodes_{method}.csv')}")
        
        return top_nodes
    
    def compare_initial_nodes(self, model='ic', p=0.3, initial_infected=10, max_steps=100, seed=42):
        """
        比较不同类型初始节点的传播效果
        
        参数:
        - model: 传播模型 ('ic' 或 'sir')
        - p: 传播概率
        - initial_infected: 初始感染节点数
        - max_steps: 最大模拟步数
        - seed: 随机种子
        """
        print(f"\n=== 比较不同类型初始节点的传播效果，model={model} ===")
        
        # 获取ML和Web开发者节点
        ml_nodes = [node for node, is_ml in self.node_types.items() if is_ml == 1]
        web_nodes = [node for node, is_ml in self.node_types.items() if is_ml == 0]
        
        # 定义不同初始节点策略
        strategies = [
            ('random', '随机节点'),
            ('ml', 'ML开发者'),
            ('web', 'Web开发者')
        ]
        
        all_results = {}
        
        for strategy, name in strategies:
            print(f"\n--- 使用{name}作为初始感染节点 ---")
            
            # 选择初始节点
            if strategy == 'random':
                initial_nodes = random.sample(list(self.G.nodes()), initial_infected)
            elif strategy == 'ml':
                initial_nodes = random.sample(ml_nodes, initial_infected)
            elif strategy == 'web':
                initial_nodes = random.sample(web_nodes, initial_infected)
            
            # 运行传播模型
            if model == 'ic':
                results = self._run_model_with_initial_nodes('ic', p=p, initial_nodes=initial_nodes, max_steps=max_steps, seed=seed)
            elif model == 'sir':
                results = self._run_model_with_initial_nodes('sir', beta=0.3, gamma=0.1, initial_nodes=initial_nodes, max_steps=max_steps, seed=seed)
            
            all_results[strategy] = results
        
        # 比较传播效果
        self._compare_strategies(all_results)
        
        return all_results
    
    def _run_model_with_initial_nodes(self, model_type, initial_nodes, **kwargs):
        """使用指定初始节点运行传播模型"""
        
        if model_type == 'ic':
            # IC模型
            p = kwargs.get('p', 0.3)
            max_steps = kwargs.get('max_steps', 100)
            seed = kwargs.get('seed', 42)
            
            random.seed(seed)
            np.random.seed(seed)
            
            # 初始化节点状态
            states = {}
            for node in self.G.nodes():
                states[node] = 'S'
            
            # 设置初始感染节点
            for node in initial_nodes:
                states[node] = 'I'
            
            # 模拟传播
            for step in range(1, max_steps + 1):
                new_infections = set()
                infected_nodes = [node for node, state in states.items() if state == 'I']
                
                for node in infected_nodes:
                    neighbors = list(self.G.neighbors(node))
                    for neighbor in neighbors:
                        if states[neighbor] == 'S' and random.random() < p:
                            new_infections.add(neighbor)
                
                for node in new_infections:
                    states[node] = 'I'
                
                if len(new_infections) == 0:
                    break
            
            # 返回最终感染数
            total_infected = sum(1 for state in states.values() if state == 'I')
            ml_infected = sum(1 for node, state in states.items() if state == 'I' and self.node_types.get(node, 0) == 1)
            web_infected = total_infected - ml_infected
            
            return {
                'total_infected': total_infected,
                'ml_infected': ml_infected,
                'web_infected': web_infected
            }
        
        elif model_type == 'sir':
            # SIR模型
            beta = kwargs.get('beta', 0.3)
            gamma = kwargs.get('gamma', 0.1)
            max_steps = kwargs.get('max_steps', 100)
            seed = kwargs.get('seed', 42)
            
            random.seed(seed)
            np.random.seed(seed)
            
            # 初始化节点状态
            states = {}
            for node in self.G.nodes():
                states[node] = 'S'
            
            # 设置初始感染节点
            for node in initial_nodes:
                states[node] = 'I'
            
            # 模拟传播
            for step in range(1, max_steps + 1):
                new_states = states.copy()
                infected_nodes = [node for node, state in states.items() if state == 'I']
                
                for node in infected_nodes:
                    # 感染邻居
                    neighbors = list(self.G.neighbors(node))
                    for neighbor in neighbors:
                        if states[neighbor] == 'S' and random.random() < beta:
                            new_states[neighbor] = 'I'
                    
                    # 恢复
                    if random.random() < gamma:
                        new_states[node] = 'R'
                
                states = new_states
                
                # 终止条件
                if sum(1 for state in states.values() if state == 'I') == 0:
                    break
            
            # 返回最终状态数
            total_s = sum(1 for state in states.values() if state == 'S')
            total_i = sum(1 for state in states.values() if state == 'I')
            total_r = sum(1 for state in states.values() if state == 'R')
            
            return {
                'S': total_s,
                'I': total_i,
                'R': total_r,
                'total_infected': total_i + total_r
            }
    
    def _compare_strategies(self, all_results):
        """比较不同策略的传播效果"""
        print("\n=== 比较不同初始节点策略的传播效果 ===")
        
        # 打印比较结果
        print("策略\t\t总感染数")
        for strategy, results in all_results.items():
            if 'total_infected' in results:
                print(f"{strategy}\t\t{results['total_infected']}")
        
        # 可视化比较结果
        df_compare = pd.DataFrame.from_dict(all_results, orient='index')
        
        plt.figure(figsize=(10, 6))
        df_compare['total_infected'].plot(kind='bar')
        plt.title('不同初始节点策略的传播效果比较')
        plt.xlabel('初始节点策略')
        plt.ylabel('总感染数')
        plt.xticks(rotation=0)
        
        # 保存可视化结果
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'influence_propagation_results')
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'))
        plt.close()
        
        # 保存比较结果
        df_compare.to_csv(os.path.join(output_dir, 'strategy_comparison.csv'))
        print(f"\n策略比较结果已保存到 {os.path.join(output_dir, 'strategy_comparison.csv')}")
    
    def _save_results(self, df_results, model_type):
        """保存传播结果"""
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'influence_propagation_results')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{model_type}_results_{timestamp}.csv'
        df_results.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"传播结果已保存到 {os.path.join(output_dir, filename)}")
    
    def _visualize_propagation(self, df_results, model_name):
        """可视化传播过程"""
        plt.figure(figsize=(12, 8))
        
        # 绘制总人数变化
        if 'R' in df_results.columns:
            # SIR模型
            plt.plot(df_results['step'], df_results['S'], label='易感者(S)', linewidth=2)
            plt.plot(df_results['step'], df_results['I'], label='感染者(I)', linewidth=2)
            plt.plot(df_results['step'], df_results['R'], label='恢复者(R)', linewidth=2)
        else:
            # IC模型
            plt.plot(df_results['step'], df_results['S'], label='易感者(S)', linewidth=2)
            plt.plot(df_results['step'], df_results['I'], label='感染者(I)', linewidth=2)
        
        plt.title(f'{model_name}模型传播过程')
        plt.xlabel('步数')
        plt.ylabel('人数')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存可视化结果
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'influence_propagation_results')
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_propagation.png'))
        plt.close()
        
        # 绘制不同类型开发者的感染情况
        plt.figure(figsize=(12, 8))
        
        if 'I_ml' in df_results.columns:
            plt.plot(df_results['step'], df_results['I_ml'], label='ML开发者(I)', linewidth=2)
            plt.plot(df_results['step'], df_results['I_web'], label='Web开发者(I)', linewidth=2)
            
            if 'R_ml' in df_results.columns:
                plt.plot(df_results['step'], df_results['R_ml'], label='ML开发者(R)', linewidth=2, linestyle='--')
                plt.plot(df_results['step'], df_results['R_web'], label='Web开发者(R)', linewidth=2, linestyle='--')
        
        plt.title(f'{model_name}模型不同类型开发者的感染情况')
        plt.xlabel('步数')
        plt.ylabel('人数')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(output_dir, f'{model_name.lower()}_propagation_by_type.png'))
        plt.close()
    
    def run_all_experiments(self):
        """运行所有影响传播实验"""
        print("运行所有影响传播实验...")
        
        # 1. 运行不同参数的传播模型
        self.sir_model(beta=0.3, gamma=0.1, initial_infected=10)
        self.ic_model(p=0.3, initial_infected=10)
        
        # 2. 分析关键节点
        self.analyze_key_nodes(method='degree', top_k=20)
        self.analyze_key_nodes(method='betweenness', top_k=20)
        self.analyze_key_nodes(method='pagerank', top_k=20)
        
        # 3. 比较不同初始节点策略
        self.compare_initial_nodes(model='ic')
        self.compare_initial_nodes(model='sir')
        
        print("\n=== 影响传播研究完成 ===")

# 主函数
def main():
    # 获取数据目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "..", "data")
    
    # 创建影响传播实例
    ip = InfluencePropagation(data_dir)
    
    # 运行所有实验
    ip.run_all_experiments()

if __name__ == "__main__":
    main()
