import json
import numpy as np
import laspy
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
from scipy.ndimage import grey_dilation
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DEFAULT_PX_PER_METER = 50


def las2numpy(las):
    return np.vstack([las.x, las.y, las.z]).T


def offset(points):
    """
    Centre the cloud horizontally (mean x, mean y) and align the base
    of the tree to z=0 (minimum z).  This ensures all trees start from
    the same vertical reference in the projected image regardless of
    crown shape or asymmetry.
    """
    x_center = points[:, 0].mean()
    y_center = points[:, 1].mean()
    z_base   = points[:, 2].min()
    return np.array([x_center, y_center, z_base])


def get_plane(angle_deg):
    angle = np.radians(angle_deg)
    u = np.array([np.cos(angle), np.sin(angle), 0])
    v = np.array([0., 0., 1.])
    n = np.array([-np.sin(angle), np.cos(angle), 0])
    return u, v, n


def project_to_plane(points, u, v):
    return np.column_stack([
        np.dot(points, u),
        np.dot(points, v)
    ])


def hist2d(points, px_per_meter):
    """
    Build a 2D histogram where each bin represents 1/px_per_meter metres.
    Canvas size is determined by the real-world extent of the tree, so all
    trees share the same physical scale (metres per pixel).
    """
    xext = (points[:, 0].min(), points[:, 0].max())
    yext = (points[:, 1].min(), points[:, 1].max())

    width  = xext[1] - xext[0]
    height = yext[1] - yext[0]

    xbins = max(int(round(width  * px_per_meter)), 1)
    ybins = max(int(round(height * px_per_meter)), 1)

    H, _, _ = np.histogram2d(
        points[:, 0], points[:, 1],
        bins=[xbins, ybins],
        range=[[xext[0], xext[1]], [yext[0], yext[1]]]
    )
    return H.T, xbins, ybins


def pad_to_square(image, canvas_size):
    """
    Place the image centred horizontally and anchored to the bottom of the
    canvas vertically, with the tree base at the bottom and crown at the top.

    numpy arrays are row-major with row 0 at the top, so:
    - We flip the image vertically so z=0 (base) ends up at the last row.
    - We place it at the bottom of the canvas (last rows).
    - Remaining rows at the top are left as zeros (white background).
    """
    h, w = image.shape
    # Flip so that z=0 (base, first row after offset shift) is at the bottom
    image = np.flipud(image)
    padded = np.zeros((canvas_size, canvas_size))
    x_offset = (canvas_size - w) // 2   # centred horizontally
    h_fit = min(h, canvas_size)
    # Place at the bottom rows of the canvas
    padded[canvas_size - h_fit:canvas_size, x_offset:x_offset + w] = image[:h_fit, :]
    return padded


def process_tree(input_file, output_folder, angles=(0, 45, 90, 135),
                 canvas_size=1024, dpi=300, px_per_meter=DEFAULT_PX_PER_METER,
                 kernel_size=None):
    """
    Process a single LAS/LAZ file and generate projections at multiple angles.

    Args:
        input_file:   Path to input .las/.laz file.
        output_folder: Path to output folder.
        angles:       Projection angles in degrees.
        canvas_size:  Output image size in pixels (square). Trees larger than
                      canvas_size / px_per_meter metres will be clipped.
                      Increase canvas_size or reduce px_per_meter if needed.
        dpi:          Output image resolution.
        px_per_meter: Spatial resolution of the projection in pixels per metre.
                      All trees are rendered at this fixed scale so that their
                      relative sizes are preserved across the dataset.
        kernel_size:  If set (e.g. 3), applies a morphological grey dilation of
                      that size (in pixels) to thicken sparse point clouds.
                      Useful for low-density trees. None = no dilation.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    las = laspy.read(input_file)
    points = las2numpy(las)
    o = offset(points)
    shifted_points = points - o

    for angle in angles:
        filename = Path(input_file).stem
        output_name = f"{filename}_{angle:03}_{canvas_size}_{dpi}.png"
        filepath = output_folder / output_name

        if filepath.exists():
            logging.info(f"Skipping existing file: {filepath.name}")
            continue

        u, v, _ = get_plane(angle)
        projected = project_to_plane(shifted_points, u, v)

        hist, w_px, h_px = hist2d(projected, px_per_meter=px_per_meter)

        # Warn if the tree is larger than the canvas at the chosen scale
        if w_px > canvas_size or h_px > canvas_size:
            logging.warning(
                f"{Path(input_file).name} angle {angle}: tree extent "
                f"({w_px}x{h_px} px) exceeds canvas ({canvas_size} px). "
                f"Consider increasing canvas_size or reducing px_per_meter."
            )

        log_img  = np.log1p(hist)
        norm_img = log_img / log_img.max() if log_img.max() > 0 else log_img

        # Optional morphological dilation to fill gaps in sparse clouds.
        # grey_dilation expands each occupied pixel to a square of kernel_size,
        # propagating the local maximum — it thickens structure without
        # inventing new geometry.
        if kernel_size is not None and kernel_size > 1:
            norm_img = grey_dilation(norm_img, size=(kernel_size, kernel_size))

        padded = pad_to_square(norm_img, canvas_size)

        fig, ax = plt.subplots(figsize=(canvas_size / dpi, canvas_size / dpi))
        ax.set_facecolor('white')
        # binary: 0 → white (background), 1 → black (points)
        # vmax * 0.5 increases contrast by saturating the densest bins earlier
        ax.imshow(padded, cmap='binary', origin='upper',  # row 0 = top, base at bottom
                  vmin=0, vmax=padded.max() * 0.5)
        ax.axis('off')
        plt.savefig(filepath, bbox_inches='tight', dpi=dpi)
        plt.close(fig)

    logging.info(f"Processed: {input_file}")


def process_file_wrapper(args):
    file, output_path, canvas_size, dpi, px_per_meter, kernel_size, angles = args
    try:
        process_tree(file, output_path, angles=angles, canvas_size=canvas_size,
                     dpi=dpi, px_per_meter=px_per_meter, kernel_size=kernel_size)
        return True
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")
        return False


def main(input_path, output_path, n_workers=None, canvas_size=1024, dpi=300,
         px_per_meter=DEFAULT_PX_PER_METER, kernel_size=None):
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path.absolute()}")

    if n_workers is None:
        n_workers = cpu_count()

    # Load per-file angle assignments if balance_dataset.py was used
    config_path = input_path / "angles_config.json"
    if config_path.exists():
        with open(config_path) as f:
            angles_config = json.load(f)
        logging.info(f"Loaded angles_config.json ({len(angles_config)} entries)")
    else:
        angles_config = {}

    all_tasks = []
    las_files = list(input_path.rglob('*.las')) + list(input_path.rglob('*.laz'))
    for file in las_files:
        # Mirror subfolder structure (e.g. species folders) in output
        relative = file.parent.relative_to(input_path)
        output_folder = output_path / relative
        # Use per-file angles from config if available, else default
        rel_key = str(relative / file.name)
        file_angles = angles_config.get(rel_key, [0, 45, 90, 135])
        all_tasks.append((str(file), str(output_folder), canvas_size, dpi,
                          px_per_meter, kernel_size, file_angles))

    logging.info(
        f"Found {len(all_tasks)} files to process using {n_workers} workers "
        f"[px_per_meter={px_per_meter}, canvas={canvas_size}px, dpi={dpi}, "
        f"kernel_size={kernel_size}]"
    )
    with Pool(n_workers) as pool:
        results = pool.map(process_file_wrapper, all_tasks)

    successful = sum(results)
    logging.info(f"Completed: {successful}/{len(all_tasks)} files processed successfully")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(
            "Usage: python TreeProjection.py <input_path> <output_path> "
            "[n_workers] [canvas_size] [dpi] [px_per_meter] [kernel_size]"
        )
        sys.exit(1)
    input_path   = sys.argv[1]
    output_path  = sys.argv[2]
    n_workers    = int(sys.argv[3])   if len(sys.argv) > 3 else None
    canvas_size  = int(sys.argv[4])   if len(sys.argv) > 4 else 1024
    dpi          = int(sys.argv[5])   if len(sys.argv) > 5 else 300
    px_per_meter = float(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_PX_PER_METER
    kernel_size  = int(sys.argv[7])   if len(sys.argv) > 7 else None
    main(input_path, output_path, n_workers, canvas_size, dpi, px_per_meter, kernel_size)
