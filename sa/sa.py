#!/usr/bin/env python
# coding: utf-8

import random
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import pyzx
from typing import List, Tuple, Optional
from pyzx.graph.base import BaseGraph, VT, ET

class SimulatedAnnealer:
    """
    焼きなまし法を実行するクラス

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

    def solve(self, initial_state: BaseGraph, evaluate_func, get_neighbor_func):
        """
        焼きなまし法を実行して最適解を探す。

        Args:
            initial_state: 初期状態
            evaluate_func: 状態を評価する関数
            get_neighbor_func: 近傍の状態を生成する関数

        Returns:
            (best_state, best_score, history): 見つかった最も良い状態、その評価値、スコアの履歴
        """
        current_temp = self.initial_temp
        current_state = copy.deepcopy(initial_state)
        current_score = evaluate_func(current_state)

        best_state = current_state
        best_score = current_score

        history = {'temp': [], 'score': []}
        print(f"初期スコア: {best_score}")

        while current_temp > self.final_temp:
            for i in range(self.max_iterations):
                # 1. 指定された方法で近傍の状態を生成
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
            print(f"温度: {current_temp:.2f}, 現在スコア: {current_score}, 最良スコア: {best_score}")


            # 5. 温度を更新（冷却）
            current_temp *= self.cooling_rate

        return best_state, best_score, history

def generate_H_S_CNOT_T_circuit(qubits, gates, p_t=0, seed=1000):
    """ランダムな量子回路を生成する"""
    random.seed(seed)
    p_s = 0.333 * (1.0 - p_t)
    p_had = 0.333 * (1.0 - p_t)

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
            q = list(range(qubits))
            ctrl = random.choice(q)
            q.remove(ctrl)
            tgt = random.choice(q)
            c.add_gate("CNOT", ctrl, tgt)
    return c

# 評価関数
def score(g: BaseGraph) -> int:
    """グラフから抽出した回路のゲート数に基づいてスコアを計算する"""
    g_tmp = g.copy()
    try:
        # 回路抽出に失敗することがあるため、例外処理を追加
        c = pyzx.extract.extract_circuit(g_tmp, up_to_perm=True)
        c = pyzx.optimize.basic_optimization(c)
        stats = c.stats_dict()
        # 2量子ビットゲートを10、1量子ビットゲートを1として重み付け
        return 1 * stats.get("twoqubit", 0) + (stats.get("gates", 0) - stats.get("twoqubit", 0))
    except Exception:
        # 抽出に失敗した場合は非常に大きなスコアを返し、その状態を避ける
        return 999999

# 適用可能なルール（アクション）を取得
def get_actions(g: BaseGraph) -> Tuple[List, List]:
    lcomp_matches = pyzx.rules.match_lcomp_parallel(g)
    pivot_matches = pyzx.rules.match_pivot_parallel(g)
    return lcomp_matches, pivot_matches

# 全ての近傍とそのスコアを計算
def get_neighbor_scores(g: BaseGraph) -> List[Tuple[BaseGraph, int]]:
    """
    現在のグラフから遷移可能な全ての近傍を生成し、スコアを計算してリストで返す。
    """
    g_base = g.copy() # ルール適用の元となるグラフ
    lcomp_matches, pivot_matches = get_actions(g_base)
    
    neighbor_list = []

    # lcomp操作を適用した近傍を生成・評価
    for match in lcomp_matches:
        g_tmp = g_base.copy()
        pyzx.rules.apply_rule(g_tmp, pyzx.rules.lcomp, [match], check_isolated_vertices=True)
        pyzx.simplify.id_simp(g_tmp, quiet=True)
        pyzx.simplify.spider_simp(g_tmp, quiet=True)
        neighbor_list.append((g_tmp, score(g_tmp)))

    # pivot操作を適用した近傍を生成・評価
    for match in pivot_matches:
        g_tmp = g_base.copy()
        pyzx.rules.apply_rule(g_tmp, pyzx.rules.pivot, [match], check_isolated_vertices=True)
        pyzx.simplify.id_simp(g_tmp, quiet=True)
        pyzx.simplify.spider_simp(g_tmp, quiet=True)
        neighbor_list.append((g_tmp, score(g_tmp)))
    
    return neighbor_list

# 最も良いスコアの近傍を選択する関数
def get_best_neighbor_func(g: BaseGraph) -> BaseGraph:
    """
    全ての近傍を評価し、最もスコアの低い（良い）近傍を返す。
    """
    neighbor_scores = get_neighbor_scores(g)
    
    if not neighbor_scores:
        # 遷移先がなければ現在の状態を返す
        return g

    # スコアでソートし、最も良いものを選択
    neighbor_scores.sort(key=lambda x: x[1])
    best_neighbor_state = neighbor_scores[0][0]
    
    return best_neighbor_state

# --- メイン実行ブロック ---
if __name__ == '__main__':
    # 1. 初期グラフの準備
    print("--- 初期グラフ生成中 ---")
    c = generate_H_S_CNOT_T_circuit(8, 200, p_t=0.08, seed=1000)
    initial_graph = c.to_graph()
    #pyzx.simplify.full_reduce(initial_graph, quiet=True) # 初期グラフを簡略化
    pyzx.simplify.spider_simp(initial_graph)
    pyzx.simplify.to_gh(initial_graph) #red node -> green
    pyzx.simplify.id_simp(initial_graph)
    
    print("初期グラフが生成されました。")
    initial_score = score(initial_graph)
    
    # 2. 焼きなまし法の設定と実行
    print("\n--- 焼きなまし法を開始 ---")
    sa = SimulatedAnnealer(
        initial_temp=50.0,      # 初期温度
        final_temp=0.1,         # 最終温度
        cooling_rate=0.95,      # 冷却率
        max_iterations=5        # 各温度での試行回数
    )
    
    # 焼きなまし法を実行
    best_graph, best_score_result, score_history = sa.solve(
        initial_state=initial_graph,
        evaluate_func=score,
        get_neighbor_func=get_best_neighbor_func # ここで「最も良い近傍を選ぶ」関数を指定
    )
    
    # 3. 結果の表示
    print("\n--- 結果 ---")
    print(f"初期スコア: {initial_score}")
    print(f"最終的な最良スコア: {best_score_result}")
    
    print("\n初期グラフ:")
    pyzx.draw(initial_graph, labels=True)
    
    print("\n最適化後のグラフ:")
    pyzx.draw(best_graph, labels=True)
    
    # 4. スコア履歴のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(score_history['temp'], score_history['score'], marker='o')
    plt.xlabel("Temperature")
    plt.ylabel("Best Score")
    plt.title("Score Progression during Simulated Annealing")
    plt.gca().invert_xaxis() # 温度が高い方からプロット
    plt.grid(True)
    plt.show()