#!/bin/bash

#SBATCH -J zx_benchmark_run      # Job name (test.sh の設定を参考に変更)
#SBATCH -o slurm_logs/%x-%A_%a.out # Standard output file (%x: job name, %A: jobID, %a: arrayID)
#SBATCH -e slurm_logs/%x-%A_%a.err # Standard error file
#SBATCH -N 1                     # Total number of nodes requested 
#SBATCH -t 24:00:00              # Run time (hh:mm:ss) 
#SBATCH --array=1-31             # Job array (実行する回路の総数)

# --- 実行環境の情報を表示 (test.sh より) ---
echo "Runnning on host $(hostname)"
echo "Starting at $(date)"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Current working directory is $(pwd)"
echo "------------------------------------------------"

# --- ベンチマーク実行 ---
# ログ保存用のディレクトリを事前に作成
mkdir -p slurm_logs

# 回路名が記述されたファイル
CIRCUIT_LIST_FILE="circuit_list.txt"

# ジョブ配列のIDを使い、ファイルから対応する回路名を取得
CIRCUIT_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${CIRCUIT_LIST_FILE})

echo "Now processing circuit: ${CIRCUIT_NAME}"

# Pythonスクリプトを実行
# 必要に応じて、仮想環境 (venv) を有効化 (activate) してください
# source /path/to/your/venv/bin/activate
python run_benchmark.py "${CIRCUIT_NAME}"

echo "------------------------------------------------"
echo "Finished processing ${CIRCUIT_NAME}"
echo "Ending at $(date)"