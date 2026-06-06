from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def build_parser(
    *,
    description: str,
    model_name_or_path: str,
    train_file: str,
    output_dir: str,
    epochs: int,
    learning_rate: float,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-name-or-path", default=model_name_or_path)
    parser.add_argument("--train-file", default=train_file)
    parser.add_argument("--output-dir", default=output_dir)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=learning_rate)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def run_with_defaults(
    *,
    description: str,
    model_name_or_path: str,
    train_file: str,
    output_dir: str,
    epochs: int,
    learning_rate: float,
) -> None:
    args = build_parser(
        description=description,
        model_name_or_path=model_name_or_path,
        train_file=train_file,
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
    ).parse_args()

    from aml_memorization.training import TrainingConfig, run_training

    config = TrainingConfig(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        max_steps=args.max_steps,
        log_every=args.log_every,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(run_training(config), indent=2))
