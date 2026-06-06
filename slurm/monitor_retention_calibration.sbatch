#!/bin/bash
#SBATCH --job-name=aml-ret-mon
#SBATCH --account=stf-ckpt
#SBATCH --partition=ckpt-g2
#SBATCH --qos=ckpt-gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:30:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
ARRAY_JOB_ID="${ARRAY_JOB_ID:-35835727}"
cd "$PROJECT_DIR"

summarize_retention() {
  .venv/bin/python - <<'PY'
import json
import pathlib
import time

roots = sorted(pathlib.Path("results/hyak").glob("gpt2_medium_retention_*_20260602"))
if not roots:
    print("no_retention_results_yet")
for d in roots:
    print("")
    print(d.name)
    for p in sorted(d.glob("eval_*.json"), key=lambda p: p.stat().st_mtime):
        obj = json.loads(p.read_text())
        print(
            time.strftime("%H:%M:%S", time.localtime(p.stat().st_mtime)),
            p.name,
            f"{obj.get('exact_matches')}/{obj.get('num_examples')}",
            f"recall={obj.get('exact_recall')}",
            f"lp={obj.get('average_target_token_log_probability')}",
        )
    if (d / "summary.json").exists():
        print("summary=complete")
PY
}

echo "monitor_start $(date)"
while true; do
  echo
  echo "===== $(date) ====="
  squeue -j "$ARRAY_JOB_ID" -o '%.18i %.9P %.24j %.8u %.2t %.10M %.6D %.5C %.10m %.30R' || true
  summarize_retention || true

  if [ -z "$(squeue -h -j "$ARRAY_JOB_ID" 2>/dev/null || true)" ]; then
    echo "array_not_in_queue"
    break
  fi
  sleep 120
done

echo
echo "final_check $(date)"
summarize_retention || true
.venv/bin/python scripts/aggregate_results.py --results-dir results/hyak --output-csv results/hyak/aggregate.csv || true
echo "aggregate_refreshed results/hyak/aggregate.csv"
echo "monitor_end $(date)"
