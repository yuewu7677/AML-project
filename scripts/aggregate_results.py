from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate memorization pipeline summaries.")
    parser.add_argument("--results-dir", default="results/hyak")
    parser.add_argument("--output-csv", default="results/hyak/aggregate.csv")
    return parser.parse_args()


def metric(payload: dict[str, Any], name: str) -> Any:
    return payload.get(name)


def avg_target_log_loss(payload: dict[str, Any]) -> Any:
    if "average_target_token_log_loss" in payload:
        return payload["average_target_token_log_loss"]
    log_prob = payload.get("average_target_token_log_probability")
    return -log_prob if log_prob is not None else None


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows: list[dict[str, Any]] = []

    for summary_path in sorted(results_dir.glob("*/summary.json")):
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

        pre = summary["pre_forget"]
        rows.append(
            {
                "experiment": summary["experiment_name"],
                "model": summary["model_name_or_path"],
                "memorize_count": summary["memorize_count"],
                "background_count": summary["background_count"],
                "string_length": summary["string_length"],
                "stage": "memorize",
                "forget_epochs": 0,
                "exact_recall": metric(pre, "exact_recall"),
                "exact_matches": metric(pre, "exact_matches"),
                "num_examples": metric(pre, "num_examples"),
                "avg_target_log_prob": metric(pre, "average_target_token_log_probability"),
                "avg_target_log_loss": avg_target_log_loss(pre),
            }
        )

        for forget_epochs, payload in sorted(
            summary["forget_epochs"].items(),
            key=lambda item: int(item[0]),
        ):
            rows.append(
                {
                    "experiment": summary["experiment_name"],
                    "model": summary["model_name_or_path"],
                    "memorize_count": summary["memorize_count"],
                    "background_count": summary["background_count"],
                    "string_length": summary["string_length"],
                    "stage": "forget",
                    "forget_epochs": int(forget_epochs),
                    "exact_recall": metric(payload, "exact_recall"),
                    "exact_matches": metric(payload, "exact_matches"),
                    "num_examples": metric(payload, "num_examples"),
                    "avg_target_log_prob": metric(payload, "average_target_token_log_probability"),
                    "avg_target_log_loss": avg_target_log_loss(payload),
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "model",
        "memorize_count",
        "background_count",
        "string_length",
        "stage",
        "forget_epochs",
        "exact_recall",
        "exact_matches",
        "num_examples",
        "avg_target_log_prob",
        "avg_target_log_loss",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
