from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from src.utils import get_logger, load_config

def plot_dataset_stats(df, config, output_path=None):
    logger = get_logger(__name__, config)
    if output_path is None:
        output_path = Path(config["data"]["processed_dir"]) / "dataset_stats.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    PALETTE = {"positive": "#4CAF50", "neutral": "#FFC107", "negative": "#F44336"}
    CAT_COLOR = "#5C6BC0"

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Customer Feedback Dataset - Stage 1 Statistics", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    if "sentiment" in df.columns:
        counts = df["sentiment"].value_counts()
        labels = counts.index.tolist()
        colors = [PALETTE.get(l, "#9E9E9E") for l in labels]
        ax.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
               textprops={"fontsize": 11})
        ax.set_title("Sentiment Distribution", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No sentiment column", ha="center", va="center")

    ax = axes[0, 1]
    if "category" in df.columns:
        cat_counts = df["category"].value_counts().sort_values()
        bars = ax.barh(cat_counts.index, cat_counts.values, color=CAT_COLOR, edgecolor="white")
        ax.bar_label(bars, fmt="%d", padding=4, fontsize=9)
        ax.set_xlabel("Number of Reviews")
        ax.set_title("Reviews per Category", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.text(0.5, 0.5, "No category column", ha="center", va="center")

    ax = axes[0, 2]
    lengths = df["text"].str.len()
    ax.hist(lengths, bins=50, color=CAT_COLOR, edgecolor="white", alpha=0.85)
    ax.axvline(lengths.median(), color="#E53935", linestyle="--", linewidth=1.5,
               label=f"Median = {lengths.median():.0f}")
    ax.set_xlabel("Review length (characters)")
    ax.set_ylabel("Count")
    ax.set_title("Text Length Distribution", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1, 0]
    if "rating" in df.columns:
        rating_counts = df["rating"].value_counts().sort_index()
        bar_colors = [PALETTE["negative"], PALETTE["negative"], PALETTE["neutral"],
                      PALETTE["positive"], PALETTE["positive"]]
        bars = ax.bar(rating_counts.index.astype(str), rating_counts.values,
                      color=bar_colors[:len(rating_counts)], edgecolor="white")
        ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
        ax.set_xlabel("Star Rating")
        ax.set_ylabel("Count")
        ax.set_title("Rating Distribution", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.text(0.5, 0.5, "No rating column", ha="center", va="center")

    ax = axes[1, 1]
    if "date" in df.columns:
        monthly = df.set_index("date").resample("ME").size()
        monthly = monthly[monthly > 0]
        if not monthly.empty:
            ax.plot(monthly.index, monthly.values, color=CAT_COLOR, linewidth=2)
            ax.fill_between(monthly.index, monthly.values, alpha=0.2, color=CAT_COLOR)
            ax.set_xlabel("Month")
            ax.set_ylabel("Reviews")
            ax.set_title("Reviews Over Time", fontweight="bold")
            ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
            fig.autofmt_xdate(rotation=30, ha="right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        else:
            ax.text(0.5, 0.5, "Insufficient date data", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No date column", ha="center", va="center")

    ax = axes[1, 2]
    if "category" in df.columns and "sentiment" in df.columns:
        pivot = df.groupby(["category", "sentiment"]).size().unstack(fill_value=0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        im = ax.imshow(pivot_pct.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(len(pivot_pct.columns)))
        ax.set_xticklabels(pivot_pct.columns, fontsize=9)
        ax.set_yticks(range(len(pivot_pct.index)))
        ax.set_yticklabels([c.replace("_", "\n") for c in pivot_pct.index], fontsize=8)
        fig.colorbar(im, ax=ax, label="% of category")
        ax.set_title("Sentiment % by Category", fontweight="bold")
        for i in range(len(pivot_pct.index)):
            for j in range(len(pivot_pct.columns)):
                ax.text(j, i, f"{pivot_pct.values[i, j]:.0f}%",
                        ha="center", va="center", fontsize=9, color="black")
    else:
        ax.text(0.5, 0.5, "Need category + sentiment", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Dataset stats saved to %s", output_path)
    return output_path
