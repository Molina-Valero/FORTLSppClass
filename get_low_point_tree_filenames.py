"""
Evaluates point cloud files against absolute and density-based point thresholds.
Reads tree metadata, efficiently counts points in the corresponding .las/.laz files,
and exports the filenames of trees that fail the quality checks.
Useful for generating clean deletion lists to filter out low-quality LiDAR captures.

Outputs:
    Generates two plain-text files containing the relative filenames of the trees
    that failed the checks (one filename per line).

    1. failed_absolute_threshold.txt
    2. failed_density_threshold.txt

    Saved to: ./outputs/point_histograms/

Usage:
    python get_low_point_tree_filenames.py <las_dir> <csv_path> [--threshold N] [--density_threshold N]

Arguments:
    las_dir             Root directory containing the .las/.laz files.
    csv_path            Path to the tree metadata CSV (must contain 'tree_H' and 'filename').
    --threshold         Absolute point count minimum (default: 1000). Trees below this fail.
    --density_threshold Density minimum in points per meter of z-range (default: 1000).
                        Trees below (tree_H * density_threshold) fail.

Example:
    get_low_point_tree_filenames.py ../data/trees_laz ../data/trees_laz/tree_metadata_dev.csv --threshold 1000 --density_threshold 100
"""

import argparse
import logging
import sys
from pathlib import Path

import laspy
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Point counting
# ---------------------------------------------------------------------------

def count_points(las_path: Path) -> int | None:
    """Return the number of points in a .las/.laz file, or None on failure."""
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
        # Strip leading slash and handle extension matching
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
# Main Extraction Logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export filenames of low-quality point clouds.")
    parser.add_argument("las_dir", type=Path, help="Root directory containing .las/.laz files.")
    parser.add_argument("csv_path", type=Path, help="Path to tree metadata CSV.")
    parser.add_argument("--threshold", type=int, default=1000,
                        help="Absolute point count threshold (default: 1000).")
    parser.add_argument("--density_threshold", type=int, default=1000,
                        help="Linear density threshold in points per meter (default: 1000).")

    args = parser.parse_args()

    if not args.las_dir.is_dir():
        logging.error(f"las_dir not found: {args.las_dir}")
        sys.exit(1)
    if not args.csv_path.is_file():
        logging.error(f"CSV not found: {args.csv_path}")
        sys.exit(1)

    # Load metadata
    df = pd.read_csv(args.csv_path, sep=None, engine="python")
    required = {"tree_H", "filename"}
    missing = required - set(df.columns)
    if missing:
        logging.error(f"CSV is missing columns: {missing}")
        sys.exit(1)

    logging.info(f"Loaded {len(df)} trees from {args.csv_path.name}.")
    logging.info("Scanning files to count points...")
    df = build_counts_df(df, args.las_dir)

    # Clean up unreadable files
    failed = df["point_count"].isna().sum()
    if failed:
        logging.warning(f"{failed} files could not be read and are excluded from the export.")
    df = df.dropna(subset=["point_count"]).copy()
    df["point_count"] = df["point_count"].astype(int)

    # Apply conditions
    df_fails_absolute = df[df["point_count"] < args.threshold]
    df_fails_density = df[df["point_count"] < (df["tree_H"] * args.density_threshold)]

    # Prepare output directory
    out_dir = Path("./outputs/point_histograms")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_absolute = out_dir / "failed_absolute_threshold.txt"
    out_density = out_dir / "failed_density_threshold.txt"

    # Save to disk
    # We output just the clean filename list so it's easy to pass into bash scripts or deletion tools
    with open(out_absolute, "w") as f:
        for fname in df_fails_absolute["filename"]:
            f.write(f"{fname}\n")

    with open(out_density, "w") as f:
        for fname in df_fails_density["filename"]:
            f.write(f"{fname}\n")

    logging.info(f"\n{'='*50}")
    logging.info(f"Export Complete.")
    logging.info(f"  Trees failing absolute threshold (< {args.threshold} pts) : {len(df_fails_absolute):,}")
    logging.info(f"  -> Saved to: {out_absolute}")
    logging.info(f"  Trees failing density threshold (< {args.density_threshold} pts/m) : {len(df_fails_density):,}")
    logging.info(f"  -> Saved to: {out_density}")
    logging.info(f"{'='*50}")

if __name__ == "__main__":
    main()