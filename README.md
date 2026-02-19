# FORTLSppClass
## Tree species classification from gound-based LiDAR
This workflow classifies tree species from cross-section images of individual tree point clouds using a YOLOv5 image classification model. The script TreeProjection.py generates four 600×800 px cross-section images for each input LAS/LAZ file, rendered from four viewing angles: 0°, 45°, 90°, and 135°.
## Installation

1. Clone the repository:
```bash
git clone https://github.com/Molina-Valero/FORTLSppClass.git
cd FORTLSppClass
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
```bash
python TreeProjection.py <input_path> <output_path> [n_workers] [canvas_size] [dpi]
```

### Arguments:
- `input_path`: Directory containing `.las` or `.laz` files (flat or nested structure)
- `output_path`: Directory where projected images will be saved
- `n_workers`: Number of parallel processes (optional)
- `canvas_size`: Size (in pixels) of output square image (default: 1024)
- `dpi`: Resolution in dots per inch (default: 300)

**Examples:**
```bash
python TreeProjection.py "data/input" "data/output" 4
python TreeProjection.py "G:\My Drive\data\pruebas" "G:\My Drive\data\projections"
python TreeProjection.py data/test data/output 4 1024 300
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
