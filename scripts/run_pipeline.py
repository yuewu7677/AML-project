from __future__ import annotations

import argparse
import json
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


def has_model_artifacts(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "model.safetensors").exists()
        or any(path.glob("model-*.safetensors"))
        or (path / "pytorch_model.bin").exists()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run keyed memorization and forgetting pipeline.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--memorize-count", type=int, required=True)
    parser.add_argument("--background-count", type=int, required=True)
    parser.add_argument("--string-length", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memorize-epochs", type=int, required=True)
    parser.add_argument("--forget-epochs", default="1,2,4,8")
    parser.add_argument("--memorize-learning-rate", type=float, default=5e-5)
    parser.add_argument("--forget-learning-rate", type=float, default=5e-5)
    parser.add_argument("--memorize-batch-size", type=int, default=4)
    parser.add_argument("--forget-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    experiment_dir = Path("results") / "hyak" / args.experiment_name
    data_dir = Path("data") / "hyak" / args.experiment_name
    memorize_file = data_dir / "memorize.jsonl"
    background_file = data_dir / "background.jsonl"
    memorize_dir = experiment_dir / "memorize"
    pre_eval_path = experiment_dir / "eval_memorize.json"
    summary_path = experiment_dir / "summary.json"

    if summary_path.exists():
        print(f"Skipping {args.experiment_name}: found complete {summary_path}", flush=True)
        print(summary_path.read_text(encoding="utf-8"), flush=True)
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if memorize_file.exists() and background_file.exists():
        print(f"Reusing existing data files in {data_dir}", flush=True)
    else:
        run_command(
            [
                python,
                "scripts/generate_datasets.py",
                "--memorize-path",
                str(memorize_file),
                "--background-path",
                str(background_file),
                "--memorize-count",
                str(args.memorize_count),
                "--background-count",
                str(args.background_count),
                "--string-length",
                str(args.string_length),
                "--seed",
                str(args.seed),
            ]
        )

    if has_model_artifacts(memorize_dir):
        print(f"Reusing existing memorization checkpoint {memorize_dir}", flush=True)
    else:
        run_command(
            [
                python,
                "scripts/train_memorize.py",
                "--model-name-or-path",
                args.model_name_or_path,
                "--train-file",
                str(memorize_file),
                "--output-dir",
                str(memorize_dir),
                "--epochs",
                str(args.memorize_epochs),
                "--batch-size",
                str(args.memorize_batch_size),
                "--gradient-accumulation-steps",
                str(args.gradient_accumulation_steps),
                "--learning-rate",
                str(args.memorize_learning_rate),
                "--max-length",
                str(args.max_length),
                "--device",
                args.device,
            ]
        )

    if pre_eval_path.exists():
        print(f"Reusing existing pre-forgetting eval {pre_eval_path}", flush=True)
    else:
        run_command(
            [
                python,
                "scripts/eval_recall.py",
                "--model-name-or-path",
                str(memorize_dir),
                "--eval-file",
                str(memorize_file),
                "--output-path",
                str(pre_eval_path),
                "--max-length",
                str(args.max_length),
                "--device",
                args.device,
            ]
        )

    summary: dict[str, Any] = {
        "experiment_name": args.experiment_name,
        "model_name_or_path": args.model_name_or_path,
        "memorize_count": args.memorize_count,
        "background_count": args.background_count,
        "string_length": args.string_length,
        "seed": args.seed,
        "memorize_epochs": args.memorize_epochs,
        "forget_epochs": {},
        "pre_forget": load_json(pre_eval_path),
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
                    str(memorize_dir),
                    "--train-file",
                    str(background_file),
                    "--output-dir",
                    str(forget_dir),
                    "--epochs",
                    str(forget_epochs),
                    "--batch-size",
                    str(args.forget_batch_size),
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
