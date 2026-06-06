from __future__ import annotations

from _train_common import run_with_defaults


if __name__ == "__main__":
    run_with_defaults(
        description="Stage 1: train on memorization strings.",
        model_name_or_path="sshleifer/tiny-gpt2",
        train_file="data/keyed_memorize.jsonl",
        output_dir="results/memorize",
        epochs=8,
        learning_rate=5e-4,
    )
