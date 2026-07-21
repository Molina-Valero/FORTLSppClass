"""
balance_dataset.py
==================
Balances a LAS/LAZ dataset so that each species has equal representation
across its own height range (relative stratified sampling).

For each species:
  1. Measure the height (z_max - z_min) of every tree.
  2. Divide the species height range [min, max] into N equal bins.
  3. Sample up to `n_per_bin` trees from each bin (all available if fewer).
  4. Copy the selected LAZ/LAS files to the output directory, preserving
     the species subfolder structure.
  5. Assign projection angles based on bin fullness:
       - bin full (>= n_per_bin)      → 4 angles:  0, 45, 90, 135
       - bin at 50-99% of n_per_bin   → 6 angles:  0, 30, 60, 90, 120, 150  (+ 2 intermediate)
       - bin below 50% of n_per_bin   → 12 angles: every 30° (maximum diversity)
     The angle assignments are saved to angles_config.json in the output
     directory so TreeProjection.py can read them.

Usage
-----
python balance_dataset.py <input_path> <output_path> [--n_bins 5] [--n_per_bin 10] [--seed 42]

Example
-------
python balance_dataset.py train_laz/ train_laz_balanced/ --n_bins 5 --n_per_bin 10
"""

import argparse
import json
import logging
import shutil
import numpy as np
import laspy
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Angle sets ordered by increasing diversity
ANGLES_FULL      = [0, 45, 90, 135]           # bin full
ANGLES_HALF      = [0, 30, 60, 90, 120, 150]  # bin 50–99 %
ANGLES_SPARSE    = list(range(0, 180, 15))     # bin < 50 % → every 15°


def angles_for_bin(n_selected, n_per_bin, fixed_angles=False):
    """Return the angle list based on how full the bin is.
    If fixed_angles=True, always return the 4 base angles regardless of bin
    fullness (recommended for validation sets).
    """
    if fixed_angles:
        return ANGLES_FULL
    ratio = n_selected / n_per_bin
    if ratio >= 1.0:
        return ANGLES_FULL
    elif ratio >= 0.5:
        return ANGLES_HALF
    else:
        return ANGLES_SPARSE


def get_tree_height(laz_path):
    """Read only the header to get z range — fast, no full point load."""
    try:
        with laspy.open(laz_path) as f:
            z_min = f.header.mins[2]
            z_max = f.header.maxs[2]
            return z_max - z_min
    except Exception:
        try:
            las = laspy.read(laz_path)
            return float(las.z.max() - las.z.min())
        except Exception as e:
            logging.warning(f"Could not read {laz_path}: {e}")
            return None


def balance_species(files_and_heights, n_bins, n_per_bin, rng, fixed_angles=False):
    """
    Stratified sampling across a species height range.

    Returns
    -------
    selected : list of (Path, list[int])  — (file, angles)
    stats    : dict  — per-bin label → "n_selected/n_available (n_angles angles)"
    """
    if not files_and_heights:
        return [], {}

    heights = np.array([h for _, h in files_and_heights])
    paths   = [p for p, _ in files_and_heights]

    h_min, h_max = heights.min(), heights.max()

    if h_max == h_min:
        chosen = rng.choice(len(paths), size=min(n_per_bin, len(paths)),
                            replace=False).tolist()
        angles = angles_for_bin(len(chosen), n_per_bin)
        return [(paths[i], angles) for i in chosen], {"single_bin": len(chosen)}

    bin_edges = np.linspace(h_min, h_max, n_bins + 1)
    bin_edges[-1] += 1e-6

    selected = []
    stats = {}
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        in_bin = [p for p, h in zip(paths, heights) if lo <= h < hi]
        n_take = min(n_per_bin, len(in_bin))
        angles = angles_for_bin(n_take, n_per_bin, fixed_angles=fixed_angles)

        if n_take > 0:
            chosen = rng.choice(len(in_bin), size=n_take, replace=False)
            selected.extend([(in_bin[i], angles) for i in chosen])

        label = f"{lo:.1f}-{hi:.1f}m"
        stats[label] = (
            f"{n_take}/{len(in_bin)} trees  →  {len(angles)} angles {angles}"
        )

    return selected, stats


def main(input_path, output_path, n_bins=5, n_per_bin=10, seed=42, fixed_angles=False):
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path.absolute()}")

    rng = np.random.RandomState(seed)

    # ── 1. Collect all LAZ/LAS files grouped by species folder ──────────────
    species_files = defaultdict(list)
    for laz in sorted(input_path.rglob('*.laz')) + sorted(input_path.rglob('*.las')):
        species = laz.parent.relative_to(input_path)
        species_files[str(species)].append(laz)

    logging.info(f"Found {len(species_files)} species folders in {input_path}")

    total_in   = 0
    total_out  = 0
    total_imgs = 0
    angles_config = {}   # relative_path → angles list

    # ── 2. Per-species stratified sampling ──────────────────────────────────
    for species, files in sorted(species_files.items()):
        logging.info(f"\n{'─'*60}")
        logging.info(f"Species: {species}  ({len(files)} trees)")

        files_and_heights = []
        for f in files:
            h = get_tree_height(f)
            if h is not None:
                files_and_heights.append((f, h))

        heights = [h for _, h in files_and_heights]
        if heights:
            logging.info(
                f"  Height range: {min(heights):.1f}m – {max(heights):.1f}m  "
                f"(mean {np.mean(heights):.1f}m)"
            )

        selected, stats = balance_species(
            files_and_heights, n_bins, n_per_bin, rng, fixed_angles=fixed_angles
        )

        for bin_label, info in stats.items():
            logging.info(f"  bin {bin_label}: {info}")

        # ── 3. Copy files and build angles_config ───────────────────────────
        out_species = output_path / species
        out_species.mkdir(parents=True, exist_ok=True)

        for src, angles in selected:
            dst = out_species / src.name
            shutil.copy2(src, dst)
            # Key: path relative to output_path (forward slashes for portability)
            rel_key = str(Path(species) / src.name)
            angles_config[rel_key] = angles
            total_imgs += len(angles)

        total_in  += len(files)
        total_out += len(selected)
        logging.info(f"  → {len(selected)} trees selected")

    # ── 4. Save angles_config.json ──────────────────────────────────────────
    config_path = output_path / "angles_config.json"
    output_path.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(angles_config, f, indent=2)

    # ── 5. Summary ──────────────────────────────────────────────────────────
    logging.info(f"\n{'='*60}")
    logging.info(f"Input trees    : {total_in}")
    logging.info(f"Output trees   : {total_out}")
    logging.info(f"Output images  : ~{total_imgs}  (before projection)")
    logging.info(f"Angles config  : {config_path}")
    logging.info(f"Output path    : {output_path.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Balance a LAS/LAZ dataset by sampling trees uniformly across "
            "each species' own height range, assigning more projection angles "
            "to under-represented bins."
        )
    )
    parser.add_argument("input_path",  help="Input directory with species subfolders")
    parser.add_argument("output_path", help="Output directory for balanced dataset")
    parser.add_argument(
        "--n_bins", type=int, default=5,
        help="Number of height strata per species (default: 5)"
    )
    parser.add_argument(
        "--n_per_bin", type=int, default=10,
        help="Target number of trees per stratum (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        '--fixed_angles', action='store_true',
        help='Always use the 4 base angles (0,45,90,135) regardless of bin '
             'fullness. Recommended for validation sets.'
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.n_bins, args.n_per_bin,
         args.seed, args.fixed_angles)
