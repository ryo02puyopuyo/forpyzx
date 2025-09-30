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
    pyzx.rules.apply_rule(new_g,lcomp,[[v1,list(new_g.neighbors(v1))]],check_isolated_vertices=True)

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

def get_neibor_with_rand_lc(g:BaseGraph):
    next_state = copy.deepcopy(g)
    
    if random.uniform(0,1) <=0.8:
        actions = get_all_actions(next_state)
        if not actions: # 有効なアクションがない場合
            print("no action")
            return g
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
            tmp_g  =neighbor_unfusion(g,a,b)
            if pyzx.gflow.gflow(tmp_g):
                next_state = tmp_g
        except ValueError:
            # neighbor_unfusion の候補が見つからなかった場合は何もしない
            pass
    return next_state

def get_gate_num(g:BaseGraph):
    g_tmp = g.copy()
    c = pyzx.extract.extract_circuit(g_tmp,up_to_perm=True)
    c = pyzx.optimize.phase_block_optimize(c)
    
    stats = c.stats_dict()
    dic = {}
    dic["all"] = stats["gates"]
    dic["two"] = stats["twoqubit"]
    dic["one"] = stats["gates"] - stats["twoqubit"]
    dic["t"] = stats["tcount"]
    return dic

def get_node_and_edge_num(g:BaseGraph) -> Tuple[int, int]:
    a= g.num_vertices()
    b = g.num_edges()
    return a,b

def display_results(initial_stats, best_graph, best_stats, score_history):
    print("\n--- 結果 ---")
    ### <--- 変更点: 初期状態と最良状態の全ゲート数を表示 ---
    print(f"初期ゲート数: T={initial_stats['t']}, 2-qubit={initial_stats['two']}, Total={initial_stats['all']}")
    print(f"最終的な最良ゲート数: T={best_stats['t']}, 2-qubit={best_stats['two']}, Total={best_stats['all']}")
    print("\n--- パフォーマンスグラフ ---")
    plot_graphs2(score_history)


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
    
    def solve(self, initial_state):
        total_start_time = time.time()

        timings = {k: 0.0 for k in ['get_gate_num', 
                                    'get_neighbor', 
                                    'acceptance_probability', 
                                    'state_update', 'history_update', 
                                    'cooling', 
                                    'loop_total']}

        current_temp = self.initial_temp

        t0 = time.time()
        initial_stats = get_gate_num(initial_state)
        current_stats = initial_stats
        current_state = initial_state
        timings['get_gate_num'] += time.time() - t0

        best_state = current_state
        best_score = current_stats['t'] # スコアはTゲート数のみ
        best_stats = current_stats # 最良状態の全ゲート情報も保持

        history = {
            'score': [current_stats['t']], # グラフ描画用のスコア
            't_gates': [current_stats['t']],
            'total_gates': [current_stats['all']],
            'two_qubit_gates': [current_stats['two']],
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

            ### <--- 変更点: スコア計算を分離したため、グラフのみ受け取る ---
            t0 = time.time()
            neighbor_state = get_neibor_with_rand_lc(current_state)
            timings['get_neighbor'] += time.time() - t0

            t0 = time.time()
            neighbor_stats = get_gate_num(neighbor_state)
            timings['get_gate_num'] += time.time() - t0
            
            t0 = time.time()
            prob = self._acceptance_probability(current_stats['t'], neighbor_stats['t'], current_temp)
            timings['acceptance_probability'] += time.time() - t0

            if random.random() < prob:
                # 状態更新
                t0 = time.time()
                current_state = neighbor_state
                current_stats = neighbor_stats
                transition_count += 1
                timings['state_update'] += time.time() - t0

                ### <--- 変更点: history に全ゲート数情報を追加して記録 ---
                t0 = time.time()
                elapsed_time = time.time() - total_start_time
                history['score'].append(current_stats['t'])
                history['t_gates'].append(current_stats['t'])
                history['total_gates'].append(current_stats['all'])
                history['two_qubit_gates'].append(current_stats['two'])
                history['time'].append(elapsed_time)
                history['transition_count'].append(transition_count)
                history['iteration_count'].append(iteration_count)
                timings['history_update'] += time.time() - t0

                # 新しい最良スコアが見つかった場合のログ
                if current_stats['t'] < best_score:
                    best_state = current_state
                    best_score = current_stats['t']
                    best_stats = current_stats
                    t0 = time.time()
                    print(f"  T={current_temp:.4f}, iter={iteration_count}, 遷移回数={transition_count} -> 新しい最良スコア: {best_score}")

            # 冷却
            t0 = time.time()
            current_temp *= self.cooling_rate
            timings['cooling'] += time.time() - t0

            print(f"Iter: {iteration_count}, Temp: {current_temp:.4f}, Current Score: {current_stats['t']}, Best Score: {best_score}")
            t0 = time.time()
            timings['loop_total'] += time.time() - loop_start

        total_time = time.time() - total_start_time

        # 時間計測の結果を表示
        print("\n--- 処理時間レポート ---")
        for key, val in timings.items():
            print(f"{key:>25}: {val:.4f} 秒")
        print(f"{'total_time':>25}: {total_time:.4f} 秒")

        return best_state, best_stats, history, timings, total_time, initial_stats

# ★★★ ここからが変更された関数 ★★★

def export_results_to_json(
    circuit_name: str,
    sa_params: dict,
    total_time: float,
    timings: dict,
    history: dict,
    initial_stats: dict,
    best_stats: dict,
    best_graph: BaseGraph,
    results_dir: str = "results",
    graphs_dir: str = "graphs"
):
    """
    指定された項目をJSONファイルに出力する。
    結果のサマリーと最適化されたグラフは別々のファイル・ディレクトリに保存される。

    Args:
        circuit_name (str): 最適化対象の回路名。
        sa_params (dict): 焼きなまし法で使われたパラメータ。
        total_time (float): 総実行時間。
        timings (dict): 実行時間の内訳。
        history (dict): スコアや時間の履歴データ。
        initial_stats (dict): 最適化前の回路のゲート構成。
        final_stats (dict): 最適化後の回路のゲート構成。
        best_graph (BaseGraph): 最良解のグラフオブジェクト。
        results_dir (str): 結果サマリーを保存するディレクトリ名。
        graphs_dir (str): グラフオブジェクトを保存するディレクトリ名。
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    # タイムスタンプと回路名を含むベースファイル名を作成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{circuit_name}_{timestamp}"

    # --- 1. 結果サマリーのJSONファイルを出力 ---
    output_data = {
        'circuit_name': circuit_name,
        'sa_parameters': sa_params,
        'performance': {
            'total_execution_time_seconds': total_time,
            'timing_breakdown_seconds': timings
        },
        'initial_circuit_stats': initial_stats,
        'optimized_circuit_stats': best_stats,
        'optimization_history': history,
    }

    # 結果ファイルのパスを構築
    result_filename = f"sa_result_{base_filename}.json"
    result_filepath = os.path.join(results_dir, result_filename)

    # JSONファイルに書き出す
    with open(result_filepath, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n結果サマリーがJSONファイルとして保存されました: {result_filepath}")


    # --- 2. 最良グラフのJSONファイルを出力 ---
    # グラフファイルのパスを構築
    graph_filename = f"best_graph_{base_filename}.json"
    graph_filepath = os.path.join(graphs_dir, graph_filename)
    
    # グラフオブジェクトをJSON形式で書き出す
    with open(graph_filepath, 'w', encoding='utf-8') as f:
        f.write(best_graph.to_json())

    print(f"最良グラフがJSONファイルとして保存されました: {graph_filepath}")