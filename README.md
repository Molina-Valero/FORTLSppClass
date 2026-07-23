# FORTLSppClass
## Tree species classification from gound-based LiDAR
This workflow classifies tree species from cross-section images of individual tree point clouds using a YOLOv8 image classification model. The script TreeProjection.py generates four 640×640 px cross-section images for each input LAS/LAZ file, rendered from four viewing angles: 0°, 45°, 90°, and 135°.
## Installation

1. Clone the repository:
```bash
git clone https://github.com/Molina-Valero/FORTLSppClass.git
cd FORTLSppClass
```

2. Create a virtual environment (optional):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install PyTorch separately BEFORE running `pip install -r requirements.txt`.

   Choose ONE of the commands below depending on your setup and CUDA version:
    - CPU-only: pip install torch --index-url https://download.pytorch.org/whl/cpu
    - CUDA 12.x (replace 12.8 with your CUDA version): pip install torch --index-url https://download.pytorch.org/whl/cu128

4. Install the rest of the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

TODO


# Tree projections
```bash
python TreeProjection.py <input_path> <output_path> [n_workers] [canvas_size] [dpi] # Alessia
python TreeProjection_JAMV.py <input_path> <output_path> [n_workers] [angles] # Juan
python TreeProjection_features.py <input_path> <output_path> [n_workers] [angles] [search_radius] [feature] # Juan
```

### Arguments:
- `input_path`: Directory containing `.las` or `.laz` files (flat or nested structure)
- `output_path`: Directory where projected images will be saved
- `n_workers`: Number of parallel processes (optional)
- `canvas_size`: Size (in pixels) of output square image (default: 1024)
- `dpi`: Resolution in dots per inch (default: 300)
- `angles`: Generates projections at multiple angles (default: 0°, 45°, 90°, 135°)
- `search_radius`: Implemented radius to calculate geometric features (default: 0.2)
- `feature`: Geometric feature (default: "verticality")


**Examples:**
```bash
python TreeProjection.py "data/input" "data/output" 4 1024 300
python TreeProjection_JAMV.py "data/input" "data/output" 4 (0, 45, 90, 135)
python TreeProjection_features.py "data/input" "data/output" 4 (0, 45, 90, 135) 0.2 "verticality"
```
Each processed LAS/LAZ file produces four grayscale PNG images, named `<filename>_<angle>.png`, placed in the specified output directory.

## 📁 Output Format
- PNG images (square canvas)
- Size controlled by `canvas_size`
- Centered and padded to avoid aspect-ratio distortion
- Normalized point intensity for consistent brightness

## 🤖 Downstream Use
The images are formatted for training or inference with YOLO-based classification models. Images retain key structural traits of trees (crown shape, trunk taper) thanks to aspect-aware projection and padding.

## Features

- Processes LAS/LAZ point cloud files
- Generates projections at multiple angles (0°, 45°, 90°, 135°)
- Parallel processing support
- Automatic normalization based on highest point

# YOLO Image Classification 
## Training

Train a YOLO classification model on your own dataset.

## Usage

**Option 1 — single folder (auto-split or pre-split into `train/`/`val/`)**
```bash
python train_classifier.py --data "path/to/dataset"
```

**Option 2 — separate train/val folders** (no need to move or copy files — the script links them automatically)
```bash
python train_classifier.py --train "path/to/train" --val "path/to/val"
```

**Common options**
```bash
python train_classifier.py --train "path/to/train" --val "path/to/val" \
  --model yolov8s-cls.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --name my_run \
  --device 0
```

## Options

| Flag | Description | Default |
|---|---|---|
| `--data` | Dataset folder (with `train/`+`val/` subfolders) | — |
| `--train`, `--val` | Separate train/val folders, used together instead of `--data` | — |
| `--model` | Base YOLO classification model | `yolov8n-cls.pt` |
| `--epochs` | Training epochs | `50` |
| `--imgsz` | Image size | `640` |
| `--batch` | Batch size | `16` |
| `--name` | Run name (output folder) | `my_classifier` |
| `--device` | `0` for GPU, `cpu`, or `mps` | auto-detected |
| `--test` | Optional image path to run a prediction on after training | — |

Results are saved to `runs/classify/<name>/`, with best weights at `runs/classify/<name>/weights/best.pt`.

# Prediction

Predicts tree species from cropped tree images (produced downstream of TLS
point-cloud tree detection) using a trained YOLO classifier.

## Pipeline position

```
Point cloud → tree detection (FORTLS) → per-tree image crops
    → predict_classifier.py → predictions_tree.csv
    → merge on treeID with tree-attribute table
```

Images per tree must share a `treeID` prefix before the first underscore
(`00069_1.jpg`, `00069_2.jpg`, ... → `treeID = 00069`), matching the tree
IDs from the detection step.

## Usage

```bash
pip install ultralytics torch

python predict_classifier.py \
  --source "path/to/tree_images_folder" \
  --model best.pt \
  --tree_output predictions_tree.csv
```

## Output

`predictions_tree.csv` — one row per tree:

```
"treeID","predicted_species"
"00069","Eucalyptus_miniata"
```

The species per tree is chosen by **majority vote** across that tree's
images, with ties broken by summed confidence. A per-image CSV
(`predictions.csv`) is also written for inspection.

Join `predictions_tree.csv` on `treeID` with the tree-attribute table from
FORTLS (DBH, height, coordinates) for stand-level analysis.

*Note: couldn't fetch the actual repo folder structure (GitHub blocks
automated access), so adjust paths above if they differ.*


## 📊 Classification Performance by Feature

| Feature       | Precision (Top-1) | Precision (Top-5) |
|---------------|-------------------|-------------------|
| Verticality   |             0.7357|             0.9530|
| Sphericity    |             0.7174|             0.9530|
| Linearity     |             0.7377|             0.9526|
| Planarity     | -                 | -                 |

> Metrics computed on the test set. **Bold** values indicate best performance per column.


