# FORTLSppClass
## Tree species classification from gound-based LiDAR

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
python TreeProjection.py <input_path> <output_path> [n_workers]
```

**Examples:**
```bash
python TreeProjection.py "data/input" "data/output" 4
python TreeProjection.py "G:\My Drive\data\pruebas" "G:\My Drive\data\projections"
```

## Features

- Processes LAS/LAZ point cloud files
- Generates projections at multiple angles (0°, 45°, 90°, 135°)
- Parallel processing support
- Automatic normalization based on highest point

## License

MIT License
