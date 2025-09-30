import matplotlib.pyplot as plt
import numpy as np
import pyzx
from typing import List,Tuple
from pyzx.graph.base import BaseGraph, VT, ET
import matplotlib.pyplot as plt
import pyzx
from pyzx.graph.base import BaseGraph
from pyzx.graph import Graph
from typing import List, Tuple
import pennylane as qml
from tqdm import tqdm
import tempfile


#回路セット読み込み
#一分ぐらいかかる
[ds] = qml.data.load("op-t-mize")

pyzx_circuits = []

for i, qscript in enumerate(ds.circuits):
    try:
        # QuantumScript を OpenQASM 2.0 に変換
        qasm_str = qscript.to_openqasm(
            wires=None,
            rotations=False,
            measure_all=False,
            precision=8
        )

        with tempfile.NamedTemporaryFile(suffix=".qasm", mode='w+', delete=False) as tmpfile:
            tmpfile.write(qasm_str)
            tmpfile.flush()
            zx_circuit = pyzx.Circuit.load(tmpfile.name)

        # 元の回路名を設定（PyZX Circuit に名前属性を追加）
        zx_circuit.name = ds.circuit_names[i]
        pyzx_circuits.append(zx_circuit)
        
    except Exception as e:
        print(f"回路 {i}（{ds.circuit_names[i]}）の変換に失敗: {e}")

from sa_util_new import *
import sys

# コマンドライン引数から回路名を取得
if len(sys.argv) > 1:
    target_circuit_name = sys.argv[1]
else:
    print("エラー: 実行する回路名をコマンドライン引数で指定してください。")
    print("例: python run_benchmark.py rc_adder_6")
    sys.exit() # エラーでスクリプトを終了

print(f"指定された回路名: {target_circuit_name}")

circuit = None
for zx_circuit in pyzx_circuits:
    # 引数で受け取った回路名と一致するものを探す
    if hasattr(zx_circuit, 'name') and zx_circuit.name == target_circuit_name:
        circuit = zx_circuit
        break

if circuit is None:
    print(f"エラー: 回路 '{target_circuit_name}' はデータセット内に見つかりませんでした。")
    sys.exit()

g = circuit.to_graph()
pyzx.draw(g,labels=True)
#phase_reduceすると、phase_index周りのエラーがでる
g = pyzx.simplify.teleport_reduce(circuit.to_graph())
#一度回路に戻さないとphase_index周りのエラーがでる
c2 = pyzx.Circuit.from_graph(g)
g2 = c2.to_graph()
pyzx.draw(g,labels=True)
initial_graph = g2
pyzx.simplify.spider_simp(initial_graph)
pyzx.simplify.to_gh(initial_graph)
pyzx.simplify.id_simp(initial_graph)


# ステップ2: 焼きなまし法の設定と実行
print("\n--- 焼きなまし法を開始 ---")
INITIAL_TEMP = 100.0
FINAL_TEMP = 0.1
COOLING_RATE = 0.98

sa = SimulatedAnnealer_T(
    initial_temp=INITIAL_TEMP,
    final_temp=FINAL_TEMP,
    cooling_rate=COOLING_RATE,
)

best_graph, best_stats, history, timings, total_time, initial_stats = sa.solve(
    initial_state=initial_graph,
)

display_results(initial_stats, best_graph, best_stats, history)


print("\n--- JSONファイルへの出力を開始 ---")
sa_params = {
    "initial_temp": INITIAL_TEMP,
    "final_temp": FINAL_TEMP,
    "cooling_rate": COOLING_RATE
}

export_results_to_json(
    circuit_name=circuit.name, # 回路名を動的に設定
    sa_params=sa_params,
    total_time=total_time,
    timings=timings,
    history=history,
    initial_stats=initial_stats, 
    best_stats=best_stats,       
    best_graph=best_graph
)