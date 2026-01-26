import numpy as np
import laspy
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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
    angle = np.radians(angle_deg)
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

def hist2d(points, xbins, ybins):
    """Create 2D histogram of projected points."""
    H, _, _ = np.histogram2d(
        points[:, 0], 
        points[:, 1], 
        bins=[xbins, ybins]
    )
    return H.T  # Transpose to match Julia's indexing

def process_tree(input_file, output_folder, angles=(0, 45, 90, 135)):
    """
    Process a single LAS file and generate projections at multiple angles.
    
    Args:
        input_file: Path to input .las file
        output_folder: Path to output folder
        angles: Tuple of angles (in degrees) for projections
    """
    # Ensure output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load LAS file
    las = laspy.read(input_file)
    points = las2numpy(las)
    
    # Shift points based on highest point
    o = offset(points)
    shifted_points = points - o
    
    # Process each angle
    for angle in angles:
        filename = Path(input_file).stem
        filepath = output_folder / f"{filename}_{angle}.png"
        
        # Get plane basis vectors
        u, v, n = get_plane(angle)
        
        # Project points to plane
        projected_points = project_to_plane(shifted_points, u, v)
        
        # Calculate extent and bins
        xext = (projected_points[:, 0].min(), projected_points[:, 0].max())
        yext = (projected_points[:, 1].min(), projected_points[:, 1].max())
        
        xl = xext[1] - xext[0]
        yl = yext[1] - yext[0]
        
        ypx = 1000
        xpx = int(ypx / (yl / xl))
        
        # Calculate point density
        pd = np.sqrt((xl * yl) / len(projected_points))
        
        xbins = np.arange(xext[0], xext[1] + pd, pd)
        ybins = np.arange(yext[0], yext[1] + pd, pd)
        
        # Create 2D histogram
        arr = hist2d(projected_points, xbins, ybins)
        
        # Log transform (add 1 to avoid log(0))
        larr = np.log(arr + 1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 8))
        
        # Plot heatmap
        im = ax.imshow(
            larr, 
            cmap='binary',
            aspect=xpx / ypx,
            vmin=-1,
            vmax=larr.max() * 0.5,
            origin='lower'
        )
        
        # Remove decorations
        ax.axis('off')
        
        # Save figure
        plt.savefig(filepath, bbox_inches='tight', dpi=100)
        plt.close(fig)
        
    logging.info(f"Processed: {input_file}")

def process_file_wrapper(args):
    """Wrapper for multiprocessing."""
    file, output_path = args
    try:
        process_tree(file, output_path)
        return True
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")
        return False

def main(input_path, output_path, n_workers=None):
    """
    Process all LAS files in input directory structure.
    
    Args:
        input_path: Path to input directory containing species folders
        output_path: Path to output directory
        n_workers: Number of parallel workers (default: CPU count)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Validate input path exists
    if not input_path.exists():
        raise FileNotFoundError(
            f"\nInput path does not exist: {input_path.absolute()}\n\n"
            f"Please provide a valid Windows path. Examples:\n"
            f"  Full path:     python TreeProjection.py \"G:\\My Drive\\FORTLSppCLass\\data\\pruebas\" \"data\\projections\" 4\n"
            f"  Relative path: python TreeProjection.py \"data\\pruebas\" \"data\\projections\" 4\n"
        )
    
    if not input_path.is_dir():
        raise NotADirectoryError(
            f"\nInput path is not a directory: {input_path.absolute()}\n"
            f"Please provide a path to a directory containing species folders with LAS files."
        )
    
    if n_workers is None:
        n_workers = cpu_count()
    
    # Collect all files to process
    all_tasks = []
    species_folders = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not species_folders:
        logging.warning(f"No subdirectories found in {input_path.absolute()}")
        logging.info("Looking for LAS files directly in input directory...")
        las_files = list(input_path.glob('*.las')) + list(input_path.glob('*.laz'))
        if las_files:
            for file in las_files:
                all_tasks.append((str(file), str(output_path)))
        else:
            raise FileNotFoundError(
                f"\nNo LAS/LAZ files found in {input_path.absolute()}\n"
                f"Expected either:\n"
                f"  - Subdirectories containing LAS files, or\n"
                f"  - LAS files directly in the input directory"
            )
    else:
        for species_folder in species_folders:
            output_folder_name = species_folder.name
            output_folder_path = output_path / output_folder_name
            
            las_files = list(species_folder.glob('*.las')) + list(species_folder.glob('*.laz'))
            
            for file in las_files:
                all_tasks.append((str(file), str(output_folder_path)))
    
    if not all_tasks:
        raise FileNotFoundError(
            f"\nNo LAS/LAZ files found to process in {input_path.absolute()}\n"
            f"Please check that your input directory contains LAS or LAZ files."
        )
    
    logging.info(f"Found {len(all_tasks)} files to process using {n_workers} workers...")
    
    # Parallel processing
    with Pool(n_workers) as pool:
        results = pool.map(process_file_wrapper, all_tasks)
    
    successful = sum(results)
    logging.info(f"Completed: {successful}/{len(all_tasks)} files processed successfully")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("\nUsage: python TreeProjection.py <input_path> <output_path> [n_workers]")
        print("\nExamples:")
        print('  python TreeProjection.py "G:\\My Drive\\data\\pruebas" "G:\\My Drive\\data\\projections" 4')
        print('  python TreeProjection.py "data\\pruebas" "data\\projections" 4')
        print("\nArguments:")
        print("  input_path  : Directory containing species folders with LAS files")
        print("  output_path : Directory where projection images will be saved")
        print("  n_workers   : Number of parallel workers (optional, default: CPU count)")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    try:
        main(input_path, output_path, n_workers)
    except Exception as e:
        logging.error(f"\n{type(e).__name__}: {e}")
        sys.exit(1)
