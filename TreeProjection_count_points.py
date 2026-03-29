# This script quickly counts the total number of points in all .las and .laz
# files within a specified directory by reading only their file headers.
# It then generates and displays (or saves) a histogram showing the
# distribution of point counts across all the processed point cloud files.

import laspy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def get_point_counts(input_path):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path.absolute()}")

    las_files = list(input_path.glob('*.las')) + list(input_path.glob('*.laz'))
    logging.info(f"Found {len(las_files)} files. Extracting point counts...")

    point_counts = []

    for file in las_files:
        try:
            # Using laspy.open() instead of laspy.read() only reads the header
            # This is significantly faster and uses almost no memory
            with laspy.open(file) as f:
                point_counts.append(f.header.point_count)
        except Exception as e:
            logging.error(f"Error reading header of {file}: {e}")

    return point_counts


def plot_histogram(point_counts, output_image_path=None):
    if not point_counts:
        logging.warning("No data to plot.")
        return

    plt.figure(figsize=(10, 6))

    # We need a non-zero minimum for log scale calculations
    min_val = max(min(point_counts), 1)
    max_val = max(point_counts)

    # Generate 50 logarithmically spaced bins between the min and max values
    log_bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)

    plt.hist(point_counts, bins=log_bins, color='skyblue', edgecolor='black', alpha=0.7)

    # Set the x-axis to logarithmic
    plt.xscale('log')

    plt.title('Distribution of Point Cloud Sizes')
    plt.xlabel('Number of Points (Log Scale)')
    plt.ylabel('Frequency (Number of Files)')

    plt.grid(axis='y', alpha=0.75)

    if output_image_path:
        plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
        logging.info(f"Histogram saved to {output_image_path}")

    plt.show()


def main(input_path, output_image_path=None):
    counts = get_point_counts(input_path)

    if counts:
        logging.info(f"Successfully extracted counts from {len(counts)} files.")
        logging.info(f"Min points: {min(counts):,} | Max points: {max(counts):,}")
        plot_histogram(counts, output_image_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python count_points.py <input_folder_path> [output_histogram_path.png]")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    main(in_path, out_path)