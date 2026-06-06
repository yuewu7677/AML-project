import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "hyak"
FIGURE_DIR = PROJECT_ROOT / "project_report" / "figures"
EPOCHS = [0, 1, 2, 4, 8, 16]


def load_summary(experiment):
    with (RESULTS_DIR / experiment / "summary.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_at(summary, epoch, metric):
    payload = summary["pre_forget"] if epoch == 0 else summary["forget_epochs"][str(epoch)]
    return float(payload[metric])


def series(experiment, metric, epochs=EPOCHS):
    summary = load_summary(experiment)
    return [metric_at(summary, epoch, metric) for epoch in epochs]


def setup_axes(ax, title, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Background fine-tuning epochs")
    ax.set_ylabel(ylabel)
    ax.set_xticks(EPOCHS)
    ax.grid(True, axis="y", alpha=0.3)


def plot_scaled_curves():
    runs = [
        ("distilgpt2 N=256 L=8", "distilgpt2_keys256_len8_rerun_20260602"),
        ("gpt2 N=256 L=8", "gpt2_keys256_len8_rerun_20260602"),
        ("gpt2 N=256 L=12", "gpt2_keys256_len12_rerun_20260602"),
        ("gpt2 N=512 L=12", "gpt2_keys512_len12_rerun_20260602"),
        ("gpt2-medium N=512 L=12", "gpt2_medium_keys512_len12_rerun_20260602"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for label, experiment in runs:
        axes[0].plot(EPOCHS, series(experiment, "exact_recall"), marker="o", linewidth=1.6, label=label)
        axes[1].plot(
            EPOCHS,
            series(experiment, "average_target_token_log_loss"),
            marker="o",
            linewidth=1.6,
            label=label,
        )
    setup_axes(axes[0], "Exact Recall During Background SFT", "Exact recall")
    setup_axes(axes[1], "Target-Token Log Loss During Background SFT", "Average log loss")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "scaled_recall_nll_curves.png", dpi=220)
    plt.close(fig)


def plot_h2_target_length():
    groups = [
        ("N=128", "gpt2_keys128_len8", "gpt2_keys128_len12"),
        ("N=256", "gpt2_keys256_len8_rerun_20260602", "gpt2_keys256_len12_rerun_20260602"),
    ]
    x = list(range(len(groups)))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, (_, len8, len12) in enumerate(groups):
        s8 = load_summary(len8)
        s12 = load_summary(len12)
        axes[0].bar(i - width / 2, metric_at(s8, 1, "exact_recall"), width, color="#4c78a8")
        axes[0].bar(i + width / 2, metric_at(s12, 1, "exact_recall"), width, color="#f58518")
        axes[1].bar(i - width / 2, metric_at(s8, 1, "average_target_token_log_loss"), width, color="#4c78a8")
        axes[1].bar(i + width / 2, metric_at(s12, 1, "average_target_token_log_loss"), width, color="#f58518")
    for ax, ylabel, title in [
        (axes[0], "Exact recall @1", "Longer Targets Reduce One-Epoch Recall"),
        (axes[1], "Log loss @1", "Longer Targets Increase One-Epoch Loss"),
    ]:
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(["Len. 8", "Len. 12"], fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "h2_target_length.png", dpi=220)
    plt.close(fig)


def plot_h3_calibration():
    epochs = [0, 1, 2, 4, 8]
    runs = [
        ("BG4096 LR 2e-6", "gpt2_medium_retention_bg4096_lr2e6_20260602"),
        ("BG4096 LR 5e-7", "gpt2_medium_retention_bg4096_lr5e7_20260602"),
        ("BG1024 LR 2e-5", "gpt2_medium_retention_bg1024_lr2e5_20260602"),
        ("BG1024 LR 2e-6", "gpt2_medium_retention_bg1024_lr2e6_20260602"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for label, experiment in runs:
        axes[0].plot(epochs, series(experiment, "exact_recall", epochs), marker="o", linewidth=1.6, label=label)
        axes[1].plot(
            epochs,
            series(experiment, "average_target_token_log_loss", epochs),
            marker="o",
            linewidth=1.6,
            label=label,
        )
    for ax in axes:
        ax.set_xlabel("Background fine-tuning epochs")
        ax.set_xticks(epochs)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_title("H3: Exact Recall Depends on Interference Strength", fontsize=10)
    axes[0].set_ylabel("Exact recall")
    axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_title("H3: Log Loss Separates Weak and Strong Interference", fontsize=10)
    axes[1].set_ylabel("Average log loss")
    axes[1].legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "h3_calibration_curves.png", dpi=220)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_scaled_curves()
    plot_h2_target_length()
    plot_h3_calibration()
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
