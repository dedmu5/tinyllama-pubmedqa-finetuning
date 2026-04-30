from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
OVERLAP_METRIC_LABELS = {
    "mean_bleu": "BLEU",
    "mean_rouge1": "ROUGE-1",
    "mean_rougeL": "ROUGE-L",
    "mean_rouge3": "ROUGE-3",
    "mean_rouge4": "ROUGE-4",
}


def load_metrics() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "metrics_tidy.csv")


def plot_task_metrics(df: pd.DataFrame, task: str, output_name: str) -> None:
    task_df = df[(df["task"] == task) & (df["split"] == "test")]
    metrics = [m for m in OVERLAP_METRIC_LABELS if m in set(task_df["metric"])]
    labels = task_df[task_df["metric"] == metrics[0]]["label"].tolist()
    x = np.arange(len(labels))
    width = 0.13
    offsets = np.linspace(-2 * width, 2 * width, len(metrics))
    colors = ["#3f51b5", "#009688", "#ff9800", "#8e24aa", "#607d8b"]

    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=180)
    for metric, offset, color in zip(metrics, offsets, colors):
        values = task_df[task_df["metric"] == metric]["score"].to_numpy()
        ax.bar(x + offset, values, width=width, label=OVERLAP_METRIC_LABELS[metric], color=color)

    title = "Answer Generation Overlap Metrics" if task == "answer" else "Context Continuation Overlap Metrics"
    ax.set_title(f"{title} on PubMedQA", fontsize=15, fontweight="bold")
    ax.set_ylabel("Test score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(ASSETS / output_name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_bertscore(df: pd.DataFrame) -> None:
    metric_df = df[(df["split"] == "test") & (df["metric"] == "mean_bertscore_f1")]
    labels = pd.read_csv(RESULTS / "model_registry.csv")["label"].tolist()
    x = np.arange(len(labels))
    width = 0.28

    fig, ax = plt.subplots(figsize=(13, 5.8), dpi=180)
    for i, task in enumerate(["answer", "context"]):
        values = metric_df[metric_df["task"] == task]["score"].to_numpy()
        ax.bar(x + (i - 0.5) * width, values, width=width, label=task)

    ax.set_title("BERTScore F1 on PubMedQA", fontsize=15, fontweight="bold")
    ax.set_ylabel("Test BERTScore F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.7, 0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "bertscore_f1.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_experiment_matrix() -> None:
    registry = pd.read_csv(RESULTS / "model_registry.csv")
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=180)
    ax.axis("off")
    ax.set_title("Fine-Tuning Strategy Matrix", fontsize=16, fontweight="bold", pad=18)

    table_data = registry[["model_id", "label", "starting_model", "evaluation_focus"]].values.tolist()
    table = ax.table(
        cellText=table_data,
        colLabels=["ID", "Configuration", "Starting model", "Eval focus"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.09, 0.38, 0.28, 0.15],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f6f8fa")
    fig.savefig(ASSETS / "fine_tuning_strategy_matrix.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    df = load_metrics()
    plot_experiment_matrix()
    plot_task_metrics(df, "answer", "answer_metrics.png")
    plot_task_metrics(df, "context", "context_metrics.png")
    plot_bertscore(df)


if __name__ == "__main__":
    main()
