#!/bin/bash
#SBATCH --job-name=aml-mem
#SBATCH --account=stf
#SBATCH --partition=gpu-l40
#SBATCH --qos=stf-gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-6%2
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONFIG="${CONFIG:-configs/hyak_sweep.tsv}"
CACHE_ROOT="${CACHE_ROOT:-/gscratch/stf/$USER/.cache/aml_mem}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs
mkdir -p "$CACHE_ROOT"/{hf,pip}
export HF_HOME="$CACHE_ROOT/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export XDG_CACHE_HOME="$CACHE_ROOT"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"

module purge
module load cesg/python/3.8.10
python3 --version
which python3

VENV_DIR="${SLURM_TMPDIR:-/tmp}/aml_mem_venv_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-hyak.txt

LINE="$(awk -v task_id="$SLURM_ARRAY_TASK_ID" 'NR == task_id + 1 {print}' "$CONFIG")"
if [ -z "$LINE" ]; then
  echo "No config row for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

IFS=$'\t' read -r EXPERIMENT MODEL MEM_COUNT BG_COUNT STR_LEN MEM_EPOCHS FORGET_EPOCHS MEM_LR FORGET_LR BATCH_SIZE GRAD_ACCUM MAX_LENGTH <<< "$LINE"

python scripts/run_pipeline.py \
  --experiment-name "$EXPERIMENT" \
  --model-name-or-path "$MODEL" \
  --memorize-count "$MEM_COUNT" \
  --background-count "$BG_COUNT" \
  --string-length "$STR_LEN" \
  --memorize-epochs "$MEM_EPOCHS" \
  --forget-epochs "$FORGET_EPOCHS" \
  --memorize-learning-rate "$MEM_LR" \
  --forget-learning-rate "$FORGET_LR" \
  --memorize-batch-size "$BATCH_SIZE" \
  --forget-batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --max-length "$MAX_LENGTH" \
  --device auto
