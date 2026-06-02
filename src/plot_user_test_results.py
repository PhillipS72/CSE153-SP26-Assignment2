from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["pretrained", "unconditional", "conditional"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot average user ratings for pretrained, unconditional, and conditional melodies."
    )
    parser.add_argument(
        "--forms-csv",
        type=Path,
        default=Path("evaluation/user_test/forms2.csv"),
        help="CSV containing the 1-5 ratings from the user test.",
    )
    parser.add_argument(
        "--references-csv",
        type=Path,
        default=Path("evaluation/user_test_references.csv"),
        help="CSV mapping melody1/melody2/melody3 to the model type.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("evaluation/user_test/average_quality_by_model.png"),
        help="Path to save the bar chart PNG.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("evaluation/user_test/average_quality_by_model.csv"),
        help="Path to save the computed averages.",
    )
    return parser


def load_long_form(forms_csv: Path, references_csv: Path) -> pd.DataFrame:
    forms = pd.read_csv(forms_csv)
    references = pd.read_csv(references_csv)

    merged = forms.merge(references, on="sample", how="inner", validate="one_to_one")

    rows = []
    for _, row in merged.iterrows():
        rows.append({"sample": row["sample"], "model": row["melody1"], "quality": row["melody1_quality"]})
        rows.append({"sample": row["sample"], "model": row["melody2"], "quality": row["melody2_quality"]})
        rows.append({"sample": row["sample"], "model": row["melody3"], "quality": row["melody3_quality"]})

    long_form = pd.DataFrame(rows)
    return long_form


def compute_summary(long_form: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long_form.groupby("model", as_index=False)
        .agg(mean_quality=("quality", "mean"), ratings=("quality", "size"))
        .set_index("model")
        .reindex(MODEL_ORDER)
        .reset_index()
    )
    return summary


def plot_summary(summary: pd.DataFrame, output_png: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="pastel")
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)

    sns.barplot(
        data=summary,
        x="model",
        y="mean_quality",
        order=MODEL_ORDER,
        ax=ax,
        edgecolor="black",
    )

    ax.set_title("Average Melody Rating by Model Type", pad=14, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Average user rating (1-5)")
    ax.set_ylim(0, 5.5)
    ax.set_xticklabels(["Pretrained", "Unconditional", "Conditional"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for index, (_, row) in enumerate(summary.iterrows()):
        ax.text(
            index,
            row["mean_quality"] + 0.08,
            f"{row['mean_quality']:.2f}\n(n={int(row['ratings'])})",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    long_form = load_long_form(args.forms_csv, args.references_csv)
    summary = compute_summary(long_form)
    summary.to_csv(args.output_csv, index=False)
    plot_summary(summary, args.output_png)

    print(summary.to_string(index=False))
    print(f"Saved chart to {args.output_png}")
    print(f"Saved summary to {args.output_csv}")


if __name__ == "__main__":
    main()