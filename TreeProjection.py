import numpy as np
import laspy
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def las2numpy(las):
    return np.vstack([las.x, las.y, las.z]).T


def find_highest_point(points):
    max_idx = np.argmax(points[:, 2])
    return points[max_idx]


def offset(points):
    highest = find_highest_point(points)
    return np.array([highest[0], highest[1], 0])


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


def hist2d(points, canvas_size):
    xext = (points[:, 0].min(), points[:, 0].max())
    yext = (points[:, 1].min(), points[:, 1].max())

    width = xext[1] - xext[0]
    height = yext[1] - yext[0]
    scale = min(canvas_size / width, canvas_size / height)

    xbins = int(width * scale)
    ybins = int(height * scale)

    H, xedges, yedges = np.histogram2d(
        points[:, 0], points[:, 1],
        bins=[xbins, ybins],
        range=[[xext[0], xext[1]], [yext[0], yext[1]]]
    )
    return H.T, xbins, ybins


def pad_to_square(image, canvas_size):
    h, w = image.shape
    padded = np.zeros((canvas_size, canvas_size))
    y_offset = (canvas_size - h) // 2
    x_offset = (canvas_size - w) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = image
    return padded


def process_tree(input_file, output_folder, angles=(0, 45, 90, 135), canvas_size=1024, dpi=300):
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
        hist, w, h = hist2d(projected, canvas_size=canvas_size)
        log_img = np.log1p(hist)
        norm_img = log_img / log_img.max()
        padded = pad_to_square(norm_img, canvas_size)

        fig, ax = plt.subplots(figsize=(canvas_size / dpi, canvas_size / dpi))
        ax.imshow(padded, cmap='gray', origin='lower')
        ax.axis('off')
        plt.savefig(filepath, bbox_inches='tight', dpi=dpi)
        plt.close(fig)

    logging.info(f"Processed: {input_file}")


def process_file_wrapper(args):
    file, output_path, canvas_size, dpi = args
    try:
        process_tree(file, output_path, canvas_size=canvas_size, dpi=dpi)
        return True
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")
        return False


def main(input_path, output_path, n_workers=None, canvas_size=1024, dpi=300):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path.absolute()}")

    if n_workers is None:
        n_workers = cpu_count()

    all_tasks = []
    las_files = list(input_path.glob('*.las')) + list(input_path.glob('*.laz'))
    for file in las_files:
        all_tasks.append((str(file), str(output_path), canvas_size, dpi))

    logging.info(f"Found {len(all_tasks)} files to process using {n_workers} workers...")
    with Pool(n_workers) as pool:
        results = pool.map(process_file_wrapper, all_tasks)

    successful = sum(results)
    logging.info(f"Completed: {successful}/{len(all_tasks)} files processed successfully")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python TreeProjection.py <input_path> <output_path> [n_workers] [canvas_size] [dpi]")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    canvas_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
    dpi = int(sys.argv[5]) if len(sys.argv) > 5 else 300
    main(input_path, output_path, n_workers, canvas_size, dpi)