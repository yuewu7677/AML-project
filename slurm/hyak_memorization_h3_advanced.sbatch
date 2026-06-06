#!/bin/bash
#SBATCH --job-name=aml-h3-adv
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt-g2
#SBATCH --qos=ckpt-gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --array=1-4%2
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
CONFIG="${CONFIG:-configs/hyak_sweep_h3_advanced_20260602.tsv}"
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
