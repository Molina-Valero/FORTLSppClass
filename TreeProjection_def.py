import argparse
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

DEFAULT_ANGLES = (0, 45, 90, 135)
DEFAULT_OUTPUT_SIZE = 640


def las2numpy(las):
    return np.vstack([las.x, las.y, las.z]).T


def find_highest_point(points):
    max_idx = np.argmax(points[:, 2])
    return points[max_idx]


def offset(points):
    highest = find_highest_point(points)
    return np.array([highest[0], highest[1], 0])


def get_plane(angle_deg):
    angle = np.radians(float(angle_deg))
    u = np.array([np.cos(angle), np.sin(angle), 0])
    v = np.array([0.0, 0.0, 1.0])
    n = np.array([-np.sin(angle), np.cos(angle), 0])
    return u, v, n


def project_to_plane(points, u, v):
    return np.column_stack([
        np.dot(points, u),
        np.dot(points, v),
    ])


def hist2d(points, output_size):
    xext = (points[:, 0].min(), points[:, 0].max())
    yext = (points[:, 1].min(), points[:, 1].max())

    width = xext[1] - xext[0]
    height = yext[1] - yext[0]

    if width <= 0 or height <= 0:
        raise ValueError(
            "Projected point cloud must have non-zero width and height"
        )

    scale = min(output_size / width, output_size / height)

    xbins = min(output_size, max(1, round(width * scale)))
    ybins = min(output_size, max(1, round(height * scale)))

    histogram, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=[xbins, ybins],
        range=[[xext[0], xext[1]], [yext[0], yext[1]]],
    )

    return histogram.T


def pad_to_square(image, output_size):
    height, width = image.shape

    padded = np.zeros(
        (output_size, output_size),
        dtype=image.dtype,
    )

    y_offset = (output_size - height) // 2
    x_offset = (output_size - width) // 2

    padded[
        y_offset:y_offset + height,
        x_offset:x_offset + width,
    ] = image

    return padded


def process_tree(
    input_file,
    output_folder,
    angles=DEFAULT_ANGLES,
    output_size=DEFAULT_OUTPUT_SIZE,
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    las = laspy.read(input_file)
    points = las2numpy(las)

    shifted_points = points - offset(points)

    for angle in angles:
        filename = Path(input_file).stem
        filepath = output_folder / f"{filename}_{angle}.png"

        if filepath.exists():
            logging.info("Skipping existing file: %s", filepath.name)
            continue

        u, v, _ = get_plane(angle)
        projected = project_to_plane(shifted_points, u, v)

        histogram = hist2d(projected, output_size)

        log_image = np.log1p(histogram)
        maximum = log_image.max()

        if maximum > 0:
            log_image = log_image / maximum

        padded = pad_to_square(log_image, output_size)

        # Direct array saving guarantees exact pixel dimensions.
        # binary: empty pixels are white and dense pixels are black.
        plt.imsave(
            filepath,
            padded,
            cmap="binary",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
        )

    logging.info("Processed: %s", input_file)


def process_file_wrapper(args):
    file, output_path, angles, output_size = args

    try:
        process_tree(
            file,
            output_path,
            angles=angles,
            output_size=output_size,
        )
        return True
    except Exception as error:
        logging.error("Error processing %s: %s", file, error)
        return False


def main(
    input_path,
    output_path,
    n_workers=None,
    angles=DEFAULT_ANGLES,
    output_size=DEFAULT_OUTPUT_SIZE,
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {input_path.absolute()