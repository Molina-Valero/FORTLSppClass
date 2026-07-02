"""
analyze_low_point_trees.py

Counts points in each .las file listed in the metadata CSV and produces
three bar charts showing how many trees fall below a point-count threshold,
broken down by species, data_type, and tree height range (z-range proxy).

Usage:
    python analyze_low_point_trees.py <las_dir> <csv_path> [--threshold N] [--density_threshold N]

Arguments:
    las_dir      Root directory that contains the .las files.
                 Paths in the CSV (e.g. /train/00070.las) are joined to this root.
    csv_path     Path to tree_metadata_dev.csv (tab-separated).
    --threshold  Point count below which a tree is considered "low-point" (default: 1000).
    --density_threshold Point density threshold (points per metre of z-range) for an additional reference line in the scatter plot (default: same as --threshold).
"""

import argparse
import logging
import sys
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Point counting
# ---------------------------------------------------------------------------

def count_points(las_path: Path) -> int | None:
    """Return the number of points in a .las file, or None on failure."""
    try:
        with laspy.open(las_path) as f:
            return f.header.point_count
    except Exception as e:
        logging.warning(f"Could not read {las_path.name}: {e}")
        return None


def build_counts_df(df: pd.DataFrame, las_dir: Path) -> pd.DataFrame:
    """Attach a 'point_count' column to the metadata dataframe."""
    counts = []
    total = len(df)
    for i, row in enumerate(df.itertuples(), 1):
        # Strip leading slash so Path.joinpath works correctly
        rel_path = row.filename.lstrip("/").replace(".las", ".laz")
        full_path = las_dir / rel_path
        count = count_points(full_path)
        counts.append(count)
        if i % 100 == 0 or i == total:
            logging.info(f"  Processed {i}/{total} files...")
    df = df.copy()
    df["point_count"] = counts
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

PALETTE_LOW = "#E07B6A"   # warm red for low-point bars

def _bar_chart(ax, categories, low_counts, title, xlabel, note=None):
    """Draw a bar chart of low-point tree counts on *ax*."""
    x = np.arange(len(categories))

    bars = ax.bar(x, low_counts, color=PALETTE_LOW, edgecolor="white", linewidth=0.6)

    # Absolute count labels on top of each bar
    for bar, count in zip(bars, low_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(low_counts) * 0.01,
                    str(count), ha="center", va="bottom",
                    fontsize=7.5, color="#333333")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Number of trees", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=40, ha="right", fontsize=8)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.35, linestyle="--")

    if note:
        ax.annotate(note, xy=(0.01, 0.99), xycoords="axes fraction",
                    va="top", fontsize=7, color="#777777",
                    style="italic")


DATA_TYPE_COLORS = {
    "TLS": "#4C9BE8",
    "ULS": "#E8A84C",
    "MLS": "#6EC87A",
}
DEFAULT_COLOR = "#AAAAAA"  # fallback for unexpected data_type values


def plot_points_vs_height(df: pd.DataFrame, threshold: int, density_threshold: int, out_dir: Path):
    """Scatter plot of point_count vs tree_H, coloured by data_type."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for dtype, group in df.groupby("data_type"):
        color = DATA_TYPE_COLORS.get(dtype, DEFAULT_COLOR)
        ax.scatter(group["tree_H"], group["point_count"],
                   label=dtype, color=color,
                   alpha=0.55, s=12, linewidths=0)

    # Horizontal reference line: absolute point count threshold
    ax.axhline(threshold, color="#E07B6A", linewidth=1.2,
               linestyle="--", label=f"Absolute threshold ({threshold:,} pts)")

    # Diagonal reference line: point density threshold (pts per metre of z-range)
    x_range = np.linspace(df["tree_H"].min(), df["tree_H"].max(), 200)
    ax.plot(x_range, density_threshold * x_range, color="#A45EE0", linewidth=1.2,
            linestyle="--", label=f"Density threshold ({density_threshold:,} pts/m)")

    ax.set_yscale("log")
    ax.set_title("Point count vs z-range by acquisition type", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("tree_H — z-range proxy (m), not true tree height", fontsize=10)
    ax.set_ylabel("Point count (log scale)", fontsize=10)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--")
    ax.annotate("tree_H = max_z − min_z per tree; used here only as a rough size proxy.",
                xy=(0.01, 0.01), xycoords="axes fraction",
                va="bottom", fontsize=7, color="#777777", style="italic")

    fig.tight_layout()
    path = out_dir / "points_vs_height_scatter.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    logging.info(f"Saved → {path}")
    plt.show()


def plot_by_species(df_low: pd.DataFrame, threshold: int, out_dir: Path):
    species_low = df_low["species"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(max(10, len(species_low) * 0.55), 5.5))
    _bar_chart(ax,
               categories=species_low.index.tolist(),
               low_counts=species_low.values,
               title=f"Low-point trees by species  (threshold < {threshold:,} pts)",
               xlabel="Species")
    fig.tight_layout()
    path = out_dir / "low_point_by_species.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    logging.info(f"Saved → {path}")
    plt.show()


def plot_by_data_type(df_low: pd.DataFrame, threshold: int, out_dir: Path):
    dtype_low = df_low["data_type"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(5, 5))
    _bar_chart(ax,
               categories=dtype_low.index.tolist(),
               low_counts=dtype_low.values,
               title=f"Low-point trees by acquisition type  (threshold < {threshold:,} pts)",
               xlabel="Data type (TLS / ULS / MLS)")
    fig.tight_layout()
    path = out_dir / "low_point_by_data_type.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    logging.info(f"Saved → {path}")
    plt.show()


def plot_by_tree_h(df_low: pd.DataFrame, threshold: int, out_dir: Path):
    bins   = [0, 5, 10, 15, 20, 25, 30, np.inf]
    labels = ["0–5 m", "5–10 m", "10–15 m", "15–20 m", "20–25 m", "25–30 m", ">30 m"]

    df_low = df_low.copy()
    df_low["h_bin"] = pd.cut(df_low["tree_H"], bins=bins, labels=labels, right=False)
    h_low = df_low["h_bin"].value_counts().reindex(labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    _bar_chart(ax,
               categories=labels,
               low_counts=h_low.values,
               title=f"Low-point trees by z-range bin  (threshold < {threshold:,} pts)",
               xlabel="Z-range bin (top − bottom, not true tree height)",
               note="tree_H = max_z − min_z per tree; used here only as a rough size proxy.")
    fig.tight_layout()
    path = out_dir / "low_point_by_tree_h.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    logging.info(f"Saved → {path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse low-point .las trees.")
    parser.add_argument("las_dir",  type=Path, help="Root directory containing .las files.")
    parser.add_argument("csv_path", type=Path, help="Path to tree_metadata_dev.csv.")
    parser.add_argument("--threshold", type=int, default=1000,
                        help="Point count threshold for 'low-point' (default: 1000).")
    parser.add_argument("--density_threshold", type=int, default=1000,
                        help="Point density threshold (#of points per meter) for additional reference line (default: same as --threshold).")

    args = parser.parse_args()

    if not args.las_dir.is_dir():
        logging.error(f"las_dir not found: {args.las_dir}")
        sys.exit(1)
    if not args.csv_path.is_file():
        logging.error(f"CSV not found: {args.csv_path}")
        sys.exit(1)

    # Load metadata
    df = pd.read_csv(args.csv_path, sep=None, engine="python")
    required = {"treeID", "species", "data_type", "tree_H", "filename"}
    missing = required - set(df.columns)
    if missing:
        logging.error(f"CSV is missing columns: {missing}")
        sys.exit(1)

    logging.info(f"Loaded {len(df)} trees from {args.csv_path.name}.")
    logging.info("Counting points in .las files...")
    df = build_counts_df(df, args.las_dir)

    # Drop rows where reading failed
    failed = df["point_count"].isna().sum()
    if failed:
        logging.warning(f"{failed} files could not be read and will be excluded.")
    df = df.dropna(subset=["point_count"]).copy()
    df["point_count"] = df["point_count"].astype(int)

    # Summary
    df_low = df[df["point_count"] < args.threshold]
    logging.info(
        f"\n{'='*50}\n"
        f"  Total trees processed : {len(df):,}\n"
        f"  Low-point trees (<{args.threshold:,}): {len(df_low):,} "
        f"({100*len(df_low)/len(df):.1f}%)\n"
        f"  Min points : {df['point_count'].min():,}\n"
        f"  Max points : {df['point_count'].max():,}\n"
        f"{'='*50}"
    )

    if df_low.empty:
        logging.info("No trees below threshold — nothing to plot.")
        return

    out_dir = Path("./outputs/point_histograms")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_by_species(df_low, args.threshold, out_dir)
    plot_by_data_type(df_low, args.threshold, out_dir)
    plot_by_tree_h(df_low, args.threshold, out_dir)
    plot_points_vs_height(df, args.threshold, args.density_threshold, out_dir)


if __name__ == "__main__":
    main()