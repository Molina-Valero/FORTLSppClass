import numpy as np
import laspy
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
import logging
import argparse
from jakteristics import compute_features, FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DEFAULT_ANGLES = (0, 45, 90, 135)
DEFAULT_SEARCH_RADIUS = 0.2  # metres; adjust to match point cloud density
DEFAULT_FEATURE = "verticality"


def las2numpy(las):
    """Convert LAS points to numpy array of [x, y, z] coordinates."""
    points = np.vstack([las.x, las.y, las.z]).T
    return points


def find_highest_point(points):
    """Find the point with the maximum z coordinate."""
    max_idx = np.argmax(points[:, 2])
    return points[max_idx]


def offset(points):
    """Calculate offset based on highest point (x, y, 0)."""
    highest = find_highest_point(points)
    return np.array([highest[0], highest[1], 0])


def get_plane(angle_deg):
    """
    Get plane basis vectors for projection at given angle.

    Returns:
        u: first basis vector (in xy plane, rotated by angle)
        v: second basis vector (z direction)
        n: normal vector to the plane
    """
    angle = np.radians(float(angle_deg))
    u = np.array([np.cos(angle), np.sin(angle), 0])
    v = np.array([0., 0., 1.])
    n = np.array([-np.sin(angle), np.cos(angle), 0])
    return u, v, n


def project_to_plane(points, u, v):
    """Project 3D points onto plane defined by basis vectors u and v."""
    projected = np.column_stack([
        np.dot(points, u),
        np.dot(points, v)
    ])
    return projected


def feature_mean_2d(projected_points, feature_values, xbins, ybins):
    """
    Create a 2D grid where each cell holds the mean of feature_values for all
    points that fall into it.  Empty cells are set to NaN.

    Returns a 2-D array shaped (len(ybins)-1, len(xbins)-1).
    """
    # Digitise each point into its bin (1-based, 0 = out-of-range)
    xi = np.digitize(projected_points[:, 0], xbins) - 1
    yi = np.digitize(projected_points[:, 1], ybins) - 1

    nx = len(xbins) - 1
    ny = len(ybins) - 1

    # Keep only points that fall strictly inside the grid
    mask = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    xi, yi, vals = xi[mask], yi[mask], feature_values[mask]

    # Accumulate sum and count per cell
    flat_idx = yi * nx + xi
    sum_grid = np.zeros(ny * nx, dtype=np.float64)
    cnt_grid = np.zeros(ny * nx, dtype=np.int64)
    np.add.at(sum_grid, flat_idx, vals)
    np.add.at(cnt_grid, flat_idx, 1)

    mean_grid = np.full(ny * nx, np.nan)
    filled = cnt_grid > 0
    mean_grid[filled] = sum_grid[filled] / cnt_grid[filled]

    return mean_grid.reshape(ny, nx)


def process_tree(input_file, output_folder, angles=DEFAULT_ANGLES,
                 search_radius=DEFAULT_SEARCH_RADIUS, feature=DEFAULT_FEATURE):
    """
    Process a single LAS file and generate verticality-coloured projections
    at multiple angles.

    Args:
        input_file:     Path to input .las/.laz file.
        output_folder:  Path to output folder.
        angles:         Tuple of angles (degrees) for projections.
        search_radius:  Neighbourhood radius (metres) for jakteristics.
        feature:        Name of the jakteristics feature used for greyscale.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ load
    las = laspy.read(input_file)
    points = las2numpy(las)

    # ----------------------------------------- compute jakteristics feature
    # jakteristics expects float64 (x, y, z) array
    xyz = points.astype(np.float64)
    feature_names_list = list(FEATURE_NAMES)
    if feature not in feature_names_list:
        raise ValueError(
            f"Unknown feature '{feature}'. "
            f"Valid options are: {feature_names_list}"
        )
    features = compute_features(
        xyz,
        search_radius=search_radius,
        feature_names=[feature],
    )
    feature_values = features[:, 0]

    # Replace NaN values (isolated points with no neighbours) with 0
    feature_values = np.nan_to_num(feature_values, nan=0.0)

    # ------------------------------------------- centre on highest point
    o = offset(points)
    shifted_points = points - o

    # ------------------------------------------------- per-angle projection
    for angle in angles:
        filename = Path(input_file).stem
        filepath = output_folder / f"{filename}_{angle}.png"

        u, v, _ = get_plane(angle)

        projected_points = project_to_plane(shifted_points, u, v)

        xext = (projected_points[:, 0].min(), projected_points[:, 0].max())
        yext = (projected_points[:, 1].min(), projected_points[:, 1].max())

        xl = xext[1] - xext[0]
        yl = yext[1] - yext[0]

        # Use point-density spacing for bins (same logic as original)
        pd = np.sqrt((xl * yl) / len(projected_points))
        xbins = np.arange(xext[0], xext[1] + pd, pd)
        ybins = np.arange(yext[0], yext[1] + pd, pd)

        # 2-D mean feature grid
        arr = feature_mean_2d(projected_points, feature_values, xbins, ybins)

        # Fill NaN cells (no points) with 0 for display
        arr_display = np.nan_to_num(arr, nan=0.0)

        # ----------------------------------------------- figure layout
        aspect_ratio = xl / yl
        if aspect_ratio >= 1:
            img_w = 640
            img_h = int(640 / aspect_ratio)
        else:
            img_h = 640
            img_w = int(640 * aspect_ratio)

        x_offset = (640 - img_w) / 2
        y_offset = (640 - img_h) / 2

        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        ax.set_position([0, 0, 1, 1])
        ax.set_facecolor('white')

        # Greyscale: low feature value → white, high feature value → black.
        ax.imshow(
            arr_display,
            cmap='binary',        # low = white, high = black
            aspect='auto',
            vmin=0,
            vmax=1,
            origin='lower',
            extent=[x_offset, x_offset + img_w, y_offset, y_offset + img_h],
        )

        ax.set_xlim(0, 640)
        ax.set_ylim(0, 640)
        ax.axis('off')

        plt.savefig(filepath, dpi=100)
        plt.close(fig)

    logging.info(f"Processed: {input_file}")


def process_file_wrapper(args):
    file, output_path, angles, search_radius, feature = args
    try:
        process_tree(file, output_path, angles, search_radius, feature)
        return True
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")
        return False


def main(input_path, output_path, n_workers=None,
         angles=DEFAULT_ANGLES, search_radius=DEFAULT_SEARCH_RADIUS,
         feature=DEFAULT_FEATURE):
    """
    Process all LAS/LAZ files in input directory structure.

    Args:
        input_path:    Directory containing per-species sub-folders.
        output_path:   Output directory (mirrors input structure).
        n_workers:     Parallel workers (default: CPU count).
        angles:        Projection angles in degrees.
        search_radius: Neighbourhood radius for feature computation.
        feature:       jakteristics feature name used for greyscale.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if n_workers is None:
        n_workers = cpu_count()

    all_tasks = []
    for species_folder in [d for d in input_path.iterdir() if d.is_dir()]:
        output_folder_path = output_path / species_folder.name
        las_files = (list(species_folder.glob('*.las'))
                     + list(species_folder.glob('*.laz')))
        for f in las_files:
            all_tasks.append(
                (str(f), str(output_folder_path), list(angles), search_radius, feature)
            )

    logging.info(f"Processing {len(all_tasks)} files using {n_workers} workers...")
    logging.info(f"Projection angles: {angles}  |  search_radius: {search_radius} m  |  feature: {feature}")

    with Pool(n_workers) as pool:
        results = pool.map(process_file_wrapper, all_tasks)

    successful = sum(results)
    logging.info(
        f"Completed: {successful}/{len(all_tasks)} files processed successfully"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate tree projections coloured by a jakteristics feature "
            "from LAS/LAZ point cloud files."
        )
    )
    parser.add_argument(
        "input_path",
        help="Path to input directory containing species folders",
    )
    parser.add_argument("output_path", help="Path to output directory")
    parser.add_argument(
        "--n_workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--angles", type=int, nargs="+", default=None,
        metavar="ANGLE",
        help=f"Projection angles in degrees (e.g. --angles 0 90). "
             f"Default: {list(DEFAULT_ANGLES)}",
    )
    parser.add_argument(
        "--search_radius", type=float, default=DEFAULT_SEARCH_RADIUS,
        metavar="RADIUS",
        help=(
            f"Neighbourhood radius in metres used by jakteristics "
            f"(default: {DEFAULT_SEARCH_RADIUS}). "
            "Increase for sparser clouds, decrease for very dense ones."
        ),
    )

    parser.add_argument(
        "--feature", type=str, default=DEFAULT_FEATURE,
        metavar="FEATURE",
        help=(
            f"jakteristics feature used for greyscale colouring "
            f"(default: '{DEFAULT_FEATURE}'). "
            f"Valid choices: {list(FEATURE_NAMES)}"
        ),
    )

    args = parser.parse_args()
    angles = list(args.angles) if args.angles is not None else list(DEFAULT_ANGLES)

    main(args.input_path, args.output_path,
         args.n_workers, angles, args.search_radius, args.feature)
