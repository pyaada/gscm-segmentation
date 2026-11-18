# Global Surface Coal Mine Segmentation with Multi-Spectral Sentinel-2 Imagery

Official code repository for:  
**Global Surface Coal Mine Segmentation with Multi-Spectral Sentinel-2 Imagery: Overcoming Spectral Dilution using Geospatial Foundation Models**  
Piyush Yadav, Chengxing Zhai, Hui Su

> Peer-reviewed and accepted for presentation at the *34th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (SIGSPATIAL 2026)*.

## Overview

This repository provides a globally scalable pipeline for segmenting surface coal mine boundaries from multi-spectral Sentinel-2 imagery. We introduce the **Global Surface Coal Mine Polygons (GSCM)** dataset and benchmark CNNs, Vision Transformers, and the Prithvi geospatial foundation model, with a focus on overcoming spectral dilution in multi-spectral inputs.

Key contributions:
- **GSCM Dataset**: Global surface coal mine polygon labels created by linking the [Global Coal Mine Tracker (GCMT)](https://globalenergymonitor.org/projects/global-coal-mine-tracker/) with the [Global Mining Footprint (GMF)](https://www.fineprint.global/resources/) dataset
- **Cluster-level spatial splits**: Training/validation/test splits that prevent spatial leakage and test geographic generalization
- **Multi-model benchmark**: CNNs (UNet, DeepLabV3+) and Vision Transformers (SegFormer) with various encoders (ResNet50, EfficientNet-B3, MiT-B1), plus the Prithvi geospatial foundation model
- **Multi-spectral analysis**: Evaluation of 3-band (RGB), 4-band (RGB+NIR), 6-band (RGB+IRS), and 15-band (all Sentinel-2) configurations

## Setup

This project uses [Pixi](https://pixi.sh/) for dependency management.

### Prerequisites
- Linux (tested on Ubuntu 22.04)
- CUDA 12.2
- [Pixi](https://pixi.sh/latest/#installation)

### Installation

```bash
git clone https://github.com/pyaada/gscm-segmentation.git
cd gscm-segmentation

# Install dependencies with Pixi
pixi install
```

### GSCM Polygons (Zenodo)
The generated surface coal mine polygon labels are available on Zenodo. [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20759680-3C5A99?logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.20759680)    

Download to `gscm-segmentation/data/`

| File | Description |
|------|-------------|
| `aoi_rectangles.geojson` | Area-of-interest rectangles with tile IDs and split assignments |
| `surface_mine_polygons.gpkg` | Full GSCM polygon dataset |
| `surface_mine_polygons_train.gpkg` | Training split polygons |
| `surface_mine_polygons_val.gpkg` | Validation split polygons |
| `surface_mine_polygons_test.gpkg` | Test split polygons |

## Usage

### 1. Training Data Preparation

Download Sentinel-2 imagery from Microsoft Planetary Computer and create training tiles:

```bash
# Step 1: Download individual bands from STAC
python src/create_training_data.py download

# Step 2: Merge bands into spectral combinations
python src/create_training_data.py merge --combos rgb rgb_nir rgb_irs

# Step 3: Export GeoTIFF tiles
python src/create_training_data.py tile --combos rgb --splits test --dims 1024

# Or run the full pipeline
python src/create_training_data.py all --combos rgb rgb_nir --splits train val test --dims 1024
```
**Data paths convention:** The pipeline expects external data under a `BASE_DIR` (default: `data/sentinal_datastore`) and writes run artifacts to a `RUNS_DIR` (default: `BASE_DIR/runs`). Both are read from the environment; set them in `pixi.toml` under `[activation.env]`.

### 2. Model Training

#### CNN / Vision Transformer Models (geoai)

```bash
python src/geoai_segment.py
```

Edit the script to configure:
- `data_category`: spectral band combinations and batch sizes
- `architectures`: `["segformer", "deeplabv3plus", "unet"]`
- `encoders`: `["mit_b1", "efficientnet-b3", "resnet50"]`

#### Prithvi Foundation Model (terratorch)

```bash
# Train the three Prithvi experiments (frozen backbone / backbone lr 1e-5 / backbone unfrozen at epoch 10) for the RGB tiles
python src/terratorch_segment.py --category rgb+ve_tiles_1024

# Train a single experiment
python src/terratorch_segment.py --category rgb_nir+ve_tiles_1024 --experiments frozen
```

The model architecture lives in `src/terratorch_configs/prithvi_300.yaml`; the
experiment presets (learning rates, freezing schedule, loss) are defined in the
`EXPERIMENTS` table of `src/terratorch_segment.py`.

### 3. Benchmarking

Evaluate every trained model on the test split of each tile category:

```bash
python src/benchmark.py
```

Reports per-class IoU, F1, precision, recall, and COCO mAP, and writes a summary
CSV to `RUNS_DIR/rgb+ve_benchmark_results.csv`.

Useful flags: `--categories` (limit to specific tile categories), `--max-items`
(evaluate on the first N test tiles), `--no-coco` (skip the COCO metrics),
`--no-discover` (only use the checkpoints listed in `CATEGORY_SPECS`).

### 4. Dataset Auditing

Audit model predictions and dataset quality:

```bash
python src/audit.py
```

Audit visualizations and labels are provided in `audit_partial/` and `audit_test/`.

### 5. Evaluation & Visualization

Interactive Jupyter notebooks for analysis:
- `src/evaluate-ms.ipynb` — Multi-spectral evaluation and metric analysis
- `src/evaluate-ms-viz.ipynb` — Visualization of segmentation results

## License

Code is released under the [MIT License](LICENSE). The GSCM dataset is available on Zenodo under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgments

This work uses:
- [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) for Sentinel-2 data access
- [Global Coal Mine Tracker](https://globalenergymonitor.org/projects/global-coal-mine-tracker/) (Global Energy Monitor)
- [Global Mining Footprint](https://www.nature.com/articles/s43247-023-00805-6#data-availability) (Tang & Werner 2023)
- [GeoAI](https://opengeoai.org/) (Wu 2026)
- [Terratorch](https://github.com/IBM/terratorch) (IBM)