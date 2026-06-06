from __future__ import annotations

from _train_common import run_with_defaults


if __name__ == "__main__":
    run_with_defaults(
        description="Stage 2: continue SFT on unrelated data.",
        model_name_or_path="results/memorize",
        train_file="data/keyed_background.jsonl",
        output_dir="results/forget",
        epochs=4,
        learning_rate=1e-4,
    )
