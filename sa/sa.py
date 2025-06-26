import random
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import pyzx
from pyzx.graph.base import BaseGraph, VT, ET


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

    def _acceptance_probability(self, old_score, new_score, temp):
        """
        悪い解への遷移を受理する確率を計算する。
        exp(-delta_E / T)
        """
        if new_score < old_score:
            return 1.0
        # new_score >= old_score の場合（スコアが悪化した場合）
        delta_score = new_score - old_score
        return math.exp(-delta_score / temp)

    def solve(self, initial_state, evaluate_func, get_neighbor_func):
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


c = generate_H_S_CNOT_T_circuit(20, 5000, p_t=0.08, seed=1000)
g = c.to_graph()
pyzx.simplify.spider_simp(g)
pyzx.simplify.id_simp(g)
print(g.stats())


print("pivot: ",pyzx.rules.match_pivot_parallel(g))
print("lcomp: ",pyzx.rules.match_lcomp_parallel(g))

print("sa.py")

