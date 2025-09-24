import random
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import pyzx
from typing import List,Tuple,TypeVar
from pyzx.graph.base import BaseGraph, VT, ET
import math
import random
import time
import matplotlib.pyplot as plt
import pyzx
from pyzx.graph.base import BaseGraph
from pyzx.graph import Graph
from pyzx.utils import VertexType, EdgeType
from fractions import Fraction
from pyzx.rules import apply_rule,lcomp
import logging
import copy
import json
import datetime
import os

# (ここにご提示の plot_graphs2, neighbor_unfusion, is_lc_vertex, ... , score_t, display_results までの全ての関数を貼り付け)
def plot_graphs2(history):
    """
    実行履歴から2種類のグラフを描画する。
    1. 横軸: 実行時間, 縦軸: イテレーション回数
    2. 横軸: イテレーション回数, 縦軸: スコア
    """
    if not history or len(history['iteration_count']) <= 1:
        print("グラフを描画するのに十分な履歴がありません。")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # グラフ①: 横軸 実行時間, 縦軸 イテレーション回数
    ax1.plot(history['time'], history['iteration_count'], marker='o', linestyle='-', markersize=3)
    ax1.set_xlabel("Execution Time (seconds)")
    ax1.set_ylabel("Iteration Count")
    ax1.set_title("Iterations per Time")
    ax1.grid(True)

    # グラフ②: 横軸 イテレーション回数, 縦軸 スコア
    ax2.plot(history['iteration_count'], history['score'], marker='o', linestyle='-', markersize=3)
    ax2.set_xlabel("Iteration Count")
    ax2.set_ylabel("Score")
    ax2.set_title("Score per Iteration")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

VT = TypeVar('VT', bound=int) # The type that is used for representing vertices (e.g. an integer)
ET = TypeVar('ET') # The type used for representing edges (e.g. a pair of integers)

def neighbor_unfusion(g:BaseGraph,v1,v2,sign_flag=True):

    new_g = copy.deepcopy(g)

    if (sign_flag):
        sign = Fraction(1,2)
    else:
        sign = Fraction(3,2)
        
    #v1 
    new_v1 = new_g.add_vertex(ty=VertexType.Z)
    new_v2 = new_g.add_vertex(ty=VertexType.Z)
    new_g.remove_edge(new_g.edge(v1,v2))

    new_g.set_phase(new_v2,phase = new_g.phase(v1) - sign)
    new_g.set_phase(v1,phase = sign)
    
    e1 = new_g.add_edge(new_g.edge(v1,new_v1),edgetype=EdgeType.HADAMARD)
    e2 = new_g.add_edge(new_g.edge(new_v1,new_v2),edgetype=EdgeType.HADAMARD)
    e3 = new_g.add_edge(new_g.edge(new_v2,v2),edgetype=EdgeType.HADAMARD)
    pyzx.rules.apply_rule(new_g,lcomp,[[v1,list(new_g.neighbors(v1))]])

    return new_g

#from localserach.congruences
def is_lc_vertex(g, v):
    """Checks if a spider in a ZX-diagram is a valid subject for local complementation."""
    # don't want to apply LC to a single-degree spider because it will only result in a growing chain
    if g.vertex_degree(v) < 2 or g.type(v) != VertexType.Z:
        return False

    # TODO: no I/O vertices for now. note that they are included in degree count
    for n in g.neighbors(v):
        if g.type(n) == VertexType.BOUNDARY:
            return False

    return True

def get_candidate_of_neighbor_unfusion(g:BaseGraph):

    lc_vs = [v for v in g.vertices() if is_lc_vertex(g, v)]
    if not lc_vs:
        raise ValueError("No valid candidate for local complementation")
    lc_v = np.random.choice(lc_vs)   # int が返る
    
    tmp_list =[]
    for v in g.neighbors(lc_v):
        if g.edge_type(g.edge(lc_v,v)) ==EdgeType.HADAMARD:
            tmp_list.append(v)

    if not tmp_list:
        raise ValueError("No valid candidate for neighbor unfusion")
    return lc_v , np.random.choice(tmp_list)

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

def get_all_actions(g:BaseGraph):
    lc = pyzx.rules.match_lcomp_parallel(g)
    pv = pyzx.rules.match_pivot_parallel(g)
    labeled_lc = [("lc", a) for a in lc]
    labeled_pv = [("pv", a) for a in pv]
    return labeled_lc + labeled_pv

def get_neibor_with_rand_lc(g:BaseGraph,evaluate_func):
    next_state = copy.deepcopy(g)
    
    if random.uniform(0,1) <=0.8:
        actions = get_all_actions(next_state)
        if not actions: # 有効なアクションがない場合
            return g, evaluate_func(g)
        label, action = random.choice(actions)
        if label == "lc":
            pyzx.rules.apply_rule(
                next_state,
                pyzx.rules.lcomp,
                [action],
                check_isolated_vertices=True
            )
        elif label == "pv":
            pyzx.rules.apply_rule(
                next_state,
                pyzx.rules.pivot,
                [action],
                check_isolated_vertices=True
            )
        else:
            raise TypeError(f"Unknown action label: {label}")
        pyzx.simplify.id_simp(next_state, quiet=True)
        pyzx.simplify.spider_simp(next_state, quiet=True)
    else:
        try:
            a,b = get_candidate_of_neighbor_unfusion(g)
            tmp_g = neighbor_unfusion(g,a,b)
            if pyzx.gflow.gflow(tmp_g):
                next_state = tmp_g
        except ValueError:
            # neighbor_unfusion の候補が見つからなかった場合は何もしない
            pass


    next_score = evaluate_func(next_state)

    return next_state, next_score

def get_gate_num(g:BaseGraph):
    g_tmp = g.copy()
    c = pyzx.extract.extract_circuit(g_tmp,up_to_perm=True)
    c = pyzx.optimize.phase_block_optimize(c)
    
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

def score_t(g:BaseGraph) -> int:
    g_tmp = g.copy()
    c = pyzx.extract.extract_circuit(g_tmp,up_to_perm=True)
    c = pyzx.optimize.phase_block_optimize(c)
    
    return c.tcount()

def display_results(initial_score, best_graph, best_score, score_history):
    """
    最適化の結果を表示し、パフォーマンスグラフを描画する。
    """
    print("\n--- 結果 ---")
    print(f"初期スコア: {initial_score}")
    print(f"最終的な最良スコア: {best_score}")

    print("\n最適化後のグラフ:")
    # pyzx.draw(best_graph, labels=True) # 環境によっては描画エラーになるためコメントアウト

    print("\n--- パフォーマンスグラフ ---")
    plot_graphs2(score_history)

    final_t_score = score_t(best_graph)
    gate_stats = get_gate_num(best_graph)

    print(f"最終的なTスコア: {final_t_score}")
    print(f"最終的なゲート数: {gate_stats}")

class SimulatedAnnealer_T:
    """
    Attributes:
        initial_temp (float): 初期温度
        final_temp (float): 最終温度
        cooling_rate (float): 冷却率 (0 < alpha < 1)
    """
    def __init__(self, initial_temp, final_temp, cooling_rate):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate

    def _acceptance_probability(self, old_score, new_score, temp):
        if new_score < old_score:
            return 1.0
        if temp == 0:
            return 0.0
        delta_score = new_score - old_score
        return math.exp(-delta_score / temp)
    
    def solve(self, initial_state, evaluate_func):
        total_start_time = time.time()

        timings = {
            'evaluate_func': 0.0,
            'get_neighbor': 0.0,
            'acceptance_probability': 0.0,
            'state_update': 0.0,
            'history_update': 0.0,
            'print': 0.0,
            'cooling': 0.0,
            'loop_total': 0.0,
        }

        current_temp = self.initial_temp

        t0 = time.time()
        current_score = evaluate_func(initial_state)
        timings['evaluate_func'] += time.time() - t0

        current_state = initial_state
        best_state = current_state
        best_score = current_score

        history = {
            'score': [current_score],
            'time': [0.0],
            'transition_count': [0],
            'iteration_count': [0]
        }

        transition_count = 0
        iteration_count = 0

        print(f"初期スコア: {best_score}")

        while current_temp > self.final_temp:
            loop_start = time.time()
            iteration_count += 1

            t0 = time.time()
            neighbor_state, neighbor_score = get_neibor_with_rand_lc(current_state,evaluate_func)
            timings['get_neighbor'] += time.time() - t0

            t0 = time.time()
            prob = self._acceptance_probability(current_score, neighbor_score, current_temp)
            timings['acceptance_probability'] += time.time() - t0

            if random.random() < prob:
                # 状態更新
                t0 = time.time()
                current_state = neighbor_state
                current_score = neighbor_score
                transition_count += 1
                timings['state_update'] += time.time() - t0

                # 履歴更新
                t0 = time.time()
                elapsed_time = time.time() - total_start_time
                history['score'].append(current_score)
                history['time'].append(elapsed_time)
                history['transition_count'].append(transition_count)
                history['iteration_count'].append(iteration_count)
                timings['history_update'] += time.time() - t0

                # 新しい最良スコアが見つかった場合のログ
                if current_score < best_score:
                    best_state = current_state
                    best_score = current_score
                    t0 = time.time()
                    print(f"  T={current_temp:.4f}, iter={iteration_count}, 遷移回数={transition_count} -> 新しい最良スコア: {best_score}")
                    timings['print'] += time.time() - t0

            # 冷却
            t0 = time.time()
            current_temp *= self.cooling_rate
            timings['cooling'] += time.time() - t0

            # 100イテレーション毎に進捗表示
            if iteration_count % 100 == 0:
                t0 = time.time()
                print(f"Iter: {iteration_count}, Temp: {current_temp:.4f}, Current Score: {current_score}, Best Score: {best_score}")
                timings['print'] += time.time() - t0

            timings['loop_total'] += time.time() - loop_start

        total_time = time.time() - total_start_time

        # 時間計測の結果を表示
        print("\n--- 処理時間レポート ---")
        for key, val in timings.items():
            print(f"{key:>25}: {val:.4f} 秒")
        print(f"{'total_time':>25}: {total_time:.4f} 秒")

        return best_state, best_score, history, timings, total_time

# ★★★ ここからが追加・変更部分 ★★★

def export_results_to_json(
    circuit_name: str,
    sa_params: dict,
    total_time: float,
    timings: dict,
    history: dict,
    initial_stats: dict,
    final_stats: dict,
    best_graph: BaseGraph
):
    """
    指定された全ての項目を一つのJSONファイルにまとめて出力する。

    Args:
        circuit_name (str): 最適化対象の回路名や説明。
        sa_params (dict): 焼きなまし法で使われたパラメータ。
        total_time (float): 総実行時間。
        timings (dict): 実行時間の内訳。
        history (dict): スコアや時間の履歴データ。
        initial_stats (dict): 最適化前の回路のゲート構成。
        final_stats (dict): 最適化後の回路のゲート構成。
        best_graph (BaseGraph): 最良解のグラフオブジェクト。
    """
    # 出力するデータを一つの辞書にまとめる
    output_data = {
        'circuit_name': circuit_name,
        'sa_parameters': sa_params,
        'performance': {
            'total_execution_time_seconds': total_time,
            'timing_breakdown_seconds': timings
        },
        'initial_circuit_stats': initial_stats,
        'optimized_circuit_stats': final_stats,
        'optimization_history': history,
        # pyzxのグラフオブジェクトをJSON互換の辞書形式に変換して埋め込む
        'best_graph_object': json.loads(best_graph.to_json())
    }

    # ファイル名をタイムスタンプで生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sa_result_{timestamp}.json"

    # JSONファイルに書き出す
    with open(filename, 'w', encoding='utf-8') as f:
        # indent=4で見やすくフォーマット、ensure_ascii=Falseで日本語の文字化けを防ぐ
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"\n結果がJSONファイルとして保存されました: {filename}")


