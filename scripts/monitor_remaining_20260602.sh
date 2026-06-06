#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/gscratch/stf/yuew29/AML_Project}"
JOB_ID="${JOB_ID:-35834978_2}"
EXP="${EXP:-gpt2_medium_keys512_len12_rerun_20260602}"
LOG="${LOG:-$PROJECT_DIR/slurm/logs/monitor_${JOB_ID}.log}"

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$LOG")"

summarize_results() {
  .venv/bin/python - <<'PY'
import json
import pathlib
import time

exp = "gpt2_medium_keys512_len12_rerun_20260602"
d = pathlib.Path("results/hyak") / exp
for p in sorted(d.glob("eval_*.json"), key=lambda p: p.stat().st_mtime):
    obj = json.loads(p.read_text())
    print(
        time.strftime("%H:%M:%S", time.localtime(p.stat().st_mtime)),
        p.name,
        f"{obj.get('exact_matches')}/{obj.get('num_examples')}",
        f"recall={obj.get('exact_recall')}",
        f"lp={obj.get('average_target_token_log_probability')}",
    )
PY
}

{
  echo "monitor_start $(date)"
  while true; do
    echo
    echo "===== $(date) ====="
    squeue -j "$JOB_ID" -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %.5C %.10m %.20R' || true
    summarize_results || true

    if [ -f "results/hyak/$EXP/summary.json" ]; then
      echo "summary_exists results/hyak/$EXP/summary.json"
      break
    fi

    if ! squeue -h -j "$JOB_ID" >/dev/null 2>&1 || [ -z "$(squeue -h -j "$JOB_ID" 2>/dev/null || true)" ]; then
      echo "job_not_in_queue"
      break
    fi

    sleep 120
  done

  echo
  echo "final_check $(date)"
  summarize_results || true
  if [ -f "results/hyak/$EXP/summary.json" ]; then
    .venv/bin/python scripts/aggregate_results.py --results-dir results/hyak --output-csv results/hyak/aggregate.csv || true
    echo "aggregate_refreshed results/hyak/aggregate.csv"
  fi
  echo "monitor_end $(date)"
} >> "$LOG" 2>&1
