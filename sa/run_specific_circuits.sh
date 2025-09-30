#!/bin/bash

#SBATCH -J specific_test      # ジョブ名を実験内容に合わせて変更
#SBATCH -o slurm_logs/%x-%j.out # ログファイル名 (%j: jobID)
#SBATCH -e slurm_logs/%x-%j.err # エラーファイル名
#SBATCH -N 1                  # ノード数は1
#SBATCH -t 01:00:00           # 実行時間 (実験に応じて調整)

# --- 実行環境の情報を表示 ---
echo "Runnning on host $(hostname)"
echo "Starting at $(date)"
echo "Current working directory is $(pwd)"
echo "------------------------------------------------"

# ログ保存用のディレクトリを事前に作成
mkdir -p slurm_logs

# ▼▼▼【ここを編集】実行したい回路をスペース区切りで指定 ▼▼▼
#CIRCUIT_LIST="rc_adder_6 tof_10 gf2^4_mult"
CIRCUIT_LIST="gf2^4_mult"


# --- 指定された回路リストを順番に実行 ---
echo "Target circuits: ${CIRCUIT_LIST}"

for circuit_name in ${CIRCUIT_LIST}; do
    echo ""
    echo "===== Processing: ${circuit_name} ====="

    # Pythonスクリプトを実行
    # 必要に応じて仮想環境を有効化 (activate) してください
    source /path/to/your/venv/bin/activate
    python run_benchmark.py "${circuit_name}"

    echo "===== Finished: ${circuit_name} ====="
done

echo "------------------------------------------------"
echo "All specified circuits have been processed."
echo "Ending at $(date)"