from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate memorized-string recall.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--eval-file", default="data/keyed_memorize.jsonl")
    parser.add_argument("--output-path", default="results/eval.json")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--generation-margin", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from aml_memorization.evaluation import evaluate_recall

    metrics = evaluate_recall(
        model_name_or_path=args.model_name_or_path,
        eval_file=args.eval_file,
        output_path=args.output_path,
        max_length=args.max_length,
        generation_margin=args.generation_margin,
        device_name=args.device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
