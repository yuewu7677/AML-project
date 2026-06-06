#!/bin/bash
#SBATCH --job-name=aml-forget-ckpt
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt-g2
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=1-1%1
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONFIG="${CONFIG:-configs/h3_capacity_forget_from_checkpoint_20260605.tsv}"
CACHE_ROOT="${CACHE_ROOT:-${SLURM_TMPDIR:-/tmp}/aml_mem_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"

cd "$PROJECT_DIR"
mkdir -p slurm/logs
mkdir -p "$CACHE_ROOT"/{hf,pip}
export HF_HOME="$CACHE_ROOT/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf/hub"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export XDG_CACHE_HOME="$CACHE_ROOT"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

source .venv/bin/activate
python --version
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

LINE="$(awk -v task_id="$SLURM_ARRAY_TASK_ID" 'NR == task_id + 1 {print}' "$CONFIG")"
if [ -z "$LINE" ]; then
  echo "No config row for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

IFS=$'\t' read -r EXPERIMENT BASE_EXPERIMENT MODEL_LABEL BG_COUNT FORGET_LR BATCH_SIZE GRAD_ACCUM MAX_LENGTH FORGET_EPOCHS <<< "$LINE"

python scripts/run_forget_from_checkpoint.py \
  --experiment-name "$EXPERIMENT" \
  --base-experiment "$BASE_EXPERIMENT" \
  --model-label "$MODEL_LABEL" \
  --background-count "$BG_COUNT" \
  --forget-learning-rate "$FORGET_LR" \
  --forget-epochs "$FORGET_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --max-length "$MAX_LENGTH" \
  --device auto
