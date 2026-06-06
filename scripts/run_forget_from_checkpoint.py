from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def run_command(args: list[str]) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_background_subset(source: Path, target: Path, count: int) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line in src:
            if written >= count:
                break
            dst.write(line)
            written += 1
    if written != count:
        raise ValueError(f"Requested {count} background examples from {source}, wrote {written}")


def has_model_artifacts(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "model.safetensors").exists()
        or any(path.glob("model-*.safetensors"))
        or (path / "pytorch_model.bin").exists()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run forgetting-only checkpoints from an existing memorized run.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--background-count", type=int, required=True)
    parser.add_argument("--forget-learning-rate", type=float, required=True)
    parser.add_argument("--forget-epochs", default="1,2,4,8")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable

    base_results = Path("results/hyak") / args.base_experiment
    base_data = Path("data/hyak") / args.base_experiment
    base_memorize_dir = base_results / "memorize"
    base_memorize_file = base_data / "memorize.jsonl"
    base_background_file = base_data / "background.jsonl"
    base_pre_eval = base_results / "eval_memorize.json"
    base_summary = load_json(base_results / "summary.json")

    if not has_model_artifacts(base_memorize_dir):
        raise FileNotFoundError(f"Missing base memorized checkpoint: {base_memorize_dir}")
    for path in [base_memorize_file, base_background_file, base_pre_eval]:
        if not path.exists():
            raise FileNotFoundError(f"Missing base artifact: {path}")

    experiment_dir = Path("results/hyak") / args.experiment_name
    data_dir = Path("data/hyak") / args.experiment_name
    memorize_file = data_dir / "memorize.jsonl"
    background_file = data_dir / "background.jsonl"
    pre_eval_path = experiment_dir / "eval_memorize.json"
    summary_path = experiment_dir / "summary.json"

    if summary_path.exists():
        print(f"Skipping {args.experiment_name}: found complete {summary_path}", flush=True)
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return

    experiment_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    if not memorize_file.exists():
        shutil.copy2(base_memorize_file, memorize_file)
    write_background_subset(base_background_file, background_file, args.background_count)
    if not pre_eval_path.exists():
        shutil.copy2(base_pre_eval, pre_eval_path)

    summary: dict[str, Any] = {
        "experiment_name": args.experiment_name,
        "model_name_or_path": args.model_label,
        "memorize_count": base_summary["memorize_count"],
        "background_count": args.background_count,
        "string_length": base_summary["string_length"],
        "seed": base_summary["seed"],
        "memorize_epochs": base_summary["memorize_epochs"],
        "forget_epochs": {},
        "pre_forget": load_json(pre_eval_path),
        "retention_calibration": {
            "base_experiment": args.base_experiment,
            "base_memorized_checkpoint": str(base_memorize_dir),
            "forget_learning_rate": args.forget_learning_rate,
        },
    }

    for forget_epochs in parse_int_list(args.forget_epochs):
        forget_dir = experiment_dir / f"forget_e{forget_epochs}"
        forget_eval_path = experiment_dir / f"eval_forget_e{forget_epochs}.json"
        if has_model_artifacts(forget_dir):
            print(f"Reusing existing forgetting checkpoint {forget_dir}", flush=True)
        else:
            run_command(
                [
                    python,
                    "scripts/train_forget.py",
                    "--model-name-or-path",
                    str(base_memorize_dir),
                    "--train-file",
                    str(background_file),
                    "--output-dir",
                    str(forget_dir),
                    "--epochs",
                    str(forget_epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--gradient-accumulation-steps",
                    str(args.gradient_accumulation_steps),
                    "--learning-rate",
                    str(args.forget_learning_rate),
                    "--max-length",
                    str(args.max_length),
                    "--device",
                    args.device,
                ]
            )
        if forget_eval_path.exists():
            print(f"Reusing existing forgetting eval {forget_eval_path}", flush=True)
        else:
            run_command(
                [
                    python,
                    "scripts/eval_recall.py",
                    "--model-name-or-path",
                    str(forget_dir),
                    "--eval-file",
                    str(memorize_file),
                    "--output-path",
                    str(forget_eval_path),
                    "--max-length",
                    str(args.max_length),
                    "--device",
                    args.device,
                ]
            )
        summary["forget_epochs"][str(forget_epochs)] = load_json(forget_eval_path)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
