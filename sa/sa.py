#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import pyzx
from typing import List,Tuple
from pyzx.graph.base import BaseGraph, VT, ET


# In[3]:


class SimulatedAnnealer:
    """
    Attributes:
        initial_temp (float): 初期温度
        final_temp (float): 最終温度
        cooling_rate (float): 冷却率 (0 < alpha < 1)
        max_iterations (int): 各温度での最大試行回数
    """

    def __init__(self, initial_temp, final_temp, cooling_rate, max_iterations):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations

    def _acceptance_probability(self, o, new_score, temp):
        """
        悪い解への遷移を受理する確率を計算する。
        exp(-delta_E / T)
        """
        if new_score < old_score:
            return 1.0
        # new_score >= old_score の場合（スコアが悪化した場合）
        delta_score = new_score - old_score
        return math.exp(-delta_score / temp)

    def solve(self, initial_state: BaseGraph, evaluate_func, get_neighbor_func):
        """
        焼きなまし法を実行して最適解を探す。

        Args:
            initial_state: 初期状態
            evaluate_func: 状態を評価する関数
            get_neighbor_func: 近傍の状態を生成する関数

        Returns:
            (best_state, best_score): 見つかった最も良い状態とその評価値
        """
        current_temp = self.initial_temp
        current_state = copy.deepcopy(initial_state)
        current_score = evaluate_func(current_state)

        best_state = current_state
        best_score = current_score

        history = {'temp': [], 'score': []}

        while current_temp > self.final_temp:
            for _ in range(self.max_iterations):
                # 1. 近傍の状態を生成
                neighbor_state = get_neighbor_func(current_state)

                # 2. 新しい状態を評価
                neighbor_score = evaluate_func(neighbor_state)

                # 3. 遷移を許容するか決定
                prob = self._acceptance_probability(current_score, neighbor_score, current_temp)
                if random.random() < prob:
                    current_state = neighbor_state
                    current_score = neighbor_score

                # 4. 最良解を更新
                if current_score < best_score:
                    best_state = current_state
                    best_score = current_score
            
            # 履歴を保存
            history['temp'].append(current_temp)
            history['score'].append(best_score)

            # 5. 温度を更新（冷却）
            current_temp *= self.cooling_rate

        return best_state, best_score, history
    
    """
    #状態のスコアを計算
    def score():

    def get_neighbor_func():
    

    #グラフに対して、簡約化操作とそのノード一覧
    def get_action():
    
        Args:
        initial_state: 初期状態
        evaluate_func: 状態を評価する関数
        get_neighbor_func: 近傍の状態を生成する関数

    Returns:
        (best_state, best_score): 見つかった最も良い状態とその評価値

"""


# In[ ]:


def generate_H_S_CNOT_T_circuit(qubits, gates, p_t=0, seed=1000):
    random.seed(seed)  
    p_s = 0.333 * (1.0 - p_t)  
    p_had = 0.333 * (1.0 - p_t)  
    p_cnot = 0.333 * (1.0 - p_t)  

    c = pyzx.Circuit(qubits) 
    for _ in range(gates):
        r = random.random() 
        if r < p_had:
            c.add_gate("HAD", random.randrange(qubits))
        elif r < p_had + p_s:
            c.add_gate("S", random.randrange(qubits))
        elif r < p_had + p_s + p_t:
            c.add_gate("T", random.randrange(qubits))
        else:
            tgt = random.randrange(qubits)
            while True:
                ctrl = random.randrange(qubits)
                if ctrl != tgt:
                    break
            c.add_gate("CNOT", tgt, ctrl)
    return c


# In[223]:


#actions
def get_actions(g:BaseGraph):
    a = pyzx.rules.match_lcomp_parallel(g)
    b = pyzx.rules.match_pivot_parallel(g)

    return a,b

def apply_lcomp(g:BaseGraph ,match: Tuple[int,List[int]]):
    print(match)
    etab, rem_verts, rem_edges, check_isolated_vertices = pyzx.rules.lcomp(g,[match])
    g.add_edge_table(etab)
    g.remove_edges(rem_edges)
    g.remove_vertices(rem_verts)
    if check_isolated_vertices: g.remove_isolated_vertices()
    return g

def apply_pivot(g:BaseGraph, match: Tuple[int,int ,List[int],List[int]]):
    print(match)
    etab, rem_verts, rem_edges, check_isolated_vertices = pyzx.rules.pivot(g, [match])
    g.add_edge_table(etab)
    g.remove_edges(rem_edges)
    g.remove_vertices(rem_verts)
    if check_isolated_vertices: g.remove_isolated_vertices()
    return g

def print_graph_data(g: BaseGraph):
    boundarycount= 0
    for v in g.vertices():

        if (g.type(v) == 0):
            vtype = "Boundary"
            boundarycount +=1
        else:
            vtype = "Z"

        phase = g.phase(v)
        neighbors = g.neighbors(v)
        print(f"Vertex {v}:")
        print(f"  Type: {vtype}")
        print(f"  Phase: {phase}")
        print(f"  Neighbors: {neighbors}")
    
    for e in g.edges():
        s, t = g.edge_st(e)
        etype = g.edge_type(e)
        print(f"Edge {e}: {s} --({etype})-- {t}")
    print("boudarycount",boundarycount)

def get_gate_num(g:BaseGraph):
    g_tmp = g.copy()
    c = pyzx.extract.extract_circuit(g_tmp,up_to_perm=True)
    c = pyzx.optimize.basic_optimization(c)
    a = c.stats_dict()
    dict = {}
    dict["all"] = a["gates"]
    dict["two"] = a["twoqubit"]
    dict["one"] = a["gates"] - a["twoqubit"]
    dict["t"] = a["tcount"]
    return dict

def get_node_and_edge_num(g:BaseGraph) -> Tuple[int, int]:
    a= g.num_vertices()
    b = g.num_edges()
    return a,b

def score(g:BaseGraph) -> int:
    a = get_gate_num(g)
    score = 10 * a["two"] + a["one"]
    return score

def get_neighbor_scores(g: BaseGraph) -> List[Tuple[BaseGraph, int]]:
    """
    現在のグラフ状態から遷移可能な全ての近傍状態を生成し、
    それぞれの状態とそのスコアのペアをリストとして返す。

    Args:
        g (BaseGraph): 現在のグラフ状態。

    Returns:
        List[Tuple[BaseGraph, int]]: (近傍のグラフ状態, そのスコア) のタプルで構成されるリスト。
    """
    # 適用可能な操作（lcompとpivot）のリストを取得

    #g.copyは、元の頂点番号を変化させる場合があるので、get_actionsとapply_rule対象のグラフは同じでなければならない
    g2 = g.copy()
    lcomp_matches, pivot_matches = get_actions(g2)
    
    neighbor_list = []

    # lcomp操作を適用した近傍を生成・評価
    for match in lcomp_matches:
        g_tmp = g2.copy()
        pyzx.rules.apply_rule(g_tmp, pyzx.rules.lcomp, [match], check_isolated_vertices=True)
        pyzx.simplify.id_simp(g_tmp,quiet=True)
        pyzx.simplify.spider_simp(g_tmp,quiet=True)
        neighbor_list.append((g_tmp, score(g_tmp)))


    print("ok lcomp")

    # pivot操作を適用した近傍を生成・評価
    for match in pivot_matches:
        g_tmp = g.copy()
        pyzx.rules.apply_rule(g_tmp, pyzx.rules.pivot, [match], check_isolated_vertices=True)
        pyzx.simplify.id_simp(g_tmp,quiet=True)
        pyzx.simplify.spider_simp(g_tmp,quiet=True)
        neighbor_list.append((g_tmp, score(g_tmp)))

    return neighbor_list


# In[ ]:


#実験用のコード
# 1. 初期グラフの準備
c = generate_H_S_CNOT_T_circuit(8, 200, p_t=0.08, seed=1000)
g = c.to_graph()
pyzx.simplify.spider_simp(g)
pyzx.simplify.to_gh(g) #red node -> green
pyzx.simplify.id_simp(g)
pyzx.draw(g)

# 2. 現在の状態のスコアを表示
current_score = score(g)
print(f"現在のスコア: {current_score}\n")

# 3. 近傍のスコア一覧を取得して表示
neighbor_scores_list = get_neighbor_scores(g)
print("--- 遷移可能な近傍のスコア一覧 ---")
if not neighbor_scores_list:
    print("遷移可能な近傍はありません。")
else:
    for i, (neighbor_state, neighbor_score) in enumerate(neighbor_scores_list):
        print(f"近傍 {i+1}: スコア = {neighbor_score}")

print(f"\n合計 {len(neighbor_scores_list)} 個の近傍が見つかりました。")

# スコア順に昇順でソート
neighbor_scores_list = sorted(neighbor_scores_list, key=lambda x: x[1])

print(neighbor_scores_list)


# In[ ]:


"template"
"""
c = generate_H_S_CNOT_T_circuit(8, 200, p_t=0.08, seed=1000)
g = c.to_graph()
pyzx.simplify.spider_simp(g)
pyzx.simplify.to_gh(g) #red node -> green
pyzx.simplify.id_simp(g)
pyzx.draw(g)

# グラフの描画（初期状態）
pyzx.draw(g)
print("初期スコア:", score(g))

# 適用可能なマッチを取得
a, b = get_actions(g)
print(f"適用可能マッチ数: lcomp={len(a)}, pivot={len(b)}")

# 空でない方を適用
if a:
    pyzx.rules.apply_rule(g, pyzx.rules.lcomp, [a[0]])
    print("→ lcomp を適用")
    pyzx.simplify.id_simp(g)
    pyzx.simplify.spider_simp(g)

elif b:
    pyzx.rules.apply_rule(g, pyzx.rules.pivot, [b[0]])
    print("→ pivot を適用")
    pyzx.simplify.id_simp(g)
    pyzx.simplify.spider_simp(g)

else:
    print("適用可能なルールがありません")

# スコアと描画を表示
print("適用後スコア:", score(g))
pyzx.draw(g)

"""

