# GaussDF: Precise & Memory-Efficient Gaussian Distance Fields

GaussDF is a high-performance C++ pipeline designed to generate **precise offline maps** from large-scale 3D point clouds. By representing environments as a compact, block-sparse Gaussian Mixture Model (GMM), it provides a continuous **Distance Field (DF)** with centimeter-level accuracy (MAE < 0.03m).

The resulting maps are extremely lightweight and scalable to any environment size, making them ideal for **fast, real-time localization** and high-fidelity reconstruction in robotics applications.

## Overview

GaussDF partitions 3D space into 1m³ voxels (cubes) and approximates local geometry using an adaptive set of 3D Gaussians. It achieves high compression ratios while enabling fast, continuous distance queries.

### Key Features
*   **Adaptive Training**: Automatically scales Gaussian count based on local geometric complexity.
*   **Dual Distance Modes**:
    *   **Pure (Unsigned)**: Direct Euclidean distance approximation (UDF).
    *   **Signed**: Surface-aware field using Felzenszwalb-based propagation (SDF).
*   **Massive Scale Support**: Optimized for clouds with 50M+ points using multi-level KdTree structures.
*   **Parallel Processing**: Fully multi-threaded execution via OpenMP.
*   **External Configuration**: All parameters configurable via YAML (no recompilation needed).

## Dependencies

*   **C++17**
*   **PCL** (Point Cloud Library)
*   **Ceres Solver** (Non-linear optimization)
*   **OpenMP**
*   **yaml-cpp**

## Installation

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

### 1. Training (`gaussian_trainer`)
Converts a point cloud into a Gaussian-based Distance Field model.

**Configure** `config/config.yaml`:
```yaml
io:
  input_file: "/path/to/pointcloud.ply"
  output_base: "/path/to/output"
```

**Run**:
```bash
./build/gaussian_trainer config/config.yaml
```

Output: `<output_base>.csv`

### 2. Reconstruction (`gaussian_to_ply`)
Generates a dense point cloud (isosurface) from a trained model.

**Configure** `config/reconstruction.yaml`:
```yaml
io:
  input_csv: "/path/to/model.csv"
  output_ply: "/path/to/output.ply"

reconstruction:
  threshold: 0.05    # Surface distance threshold (m)
  resolution: 0.02   # Sampling resolution (m)
```

**Run**:
```bash
./build/gaussian_to_ply config/reconstruction.yaml
```

> **Note**: Modifying YAML files does **NOT** require recompilation.

---

## Configuration Reference

### Training (`config/config.yaml`)

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| **io** | `input_file` | `""` | Path to input pointcloud (.ply or .pcd) |
| | `output_base` | `""` | Base name for output files |
| **processing** | `num_threads` | `0` | OpenMP threads (0 = auto) |
| | `cube_size` | `1.0` | Size of each cube in meters |
| **downsampling** | `step` | `10` | Take every Nth point for coarse KdTree |
| **trainer** | `sample_points` | `1000` | Sample points per cube |
| | `mae_threshold_good` | `0.03` | Target MAE for "good" fit (m) |
| | `mae_threshold_max` | `0.30` | Max MAE before discarding (m) |
| | `edt_mode` | `"pure"` | `"pure"` (unsigned) or `"signed"` |
| **solver** | `max_iterations` | `250` | Max optimization iterations |
| | `max_time_seconds` | `2.0` | Time budget per cube (s) |
| **adaptive** | `populated_steps` | `[8,16,32]` | Gaussian counts for populated cubes |
| | `empty_steps` | `[2,4,8,16]` | Gaussian counts for empty cubes |
| | `mae_threshold` | `0.03` | Early stopping threshold (m) |
| | `empty_distance_threshold` | `3.0` | Max distance to train empty cubes |
| **edt** | `voxel_size` | `0.025` | SDF grid resolution (m) |
| | `margin_pure` | `0.0` | Grid margin for cube overlap (m) |
| | `margin_signed` | `0.3` | Grid margin for signed mode (m) |
| | `empty_search_margin` | `0.5` | Extra search radius for empty cubes (m) |

### Reconstruction (`config/reconstruction.yaml`)

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| **io** | `input_csv` | `""` | Path to Gaussian model (.csv) |
| | `output_ply` | `""` | Path to output PLY |
| **reconstruction** | `threshold` | `0.05` | Surface detection threshold (m) |
| | `resolution` | `0.02` | Voxel sampling resolution (m) |
| **region** | `enabled` | `false` | Enable region filtering |
| | `x_min/max, y_min/max, z_min/max` | `0.0` | Region bounds |

---

## Technical Architecture

### Spatial Partitioning
The environment is divided into a grid of cubes (default 1m³). Cubes are classified:
*   **Populated**: Contains raw points. Uses **8, 16, or 32** Gaussians.
*   **Empty**: No points, but within threshold of surface. Uses **2, 4, 8, or 16** Gaussians.

### EDT Margin & Overlap
Setting `margin_pure: 0.1` creates **20cm overlap** between adjacent cubes, improving boundary continuity for smooth interpolation.

### Training Pipeline
For each cube:
1.  **Context Gathering**: Fetches points within extended radius (proportional to cube + margin).
2.  **EDT Generation**: Computes local Euclidean Distance Transform grid.
3.  **Optimization**: Ceres Solver fits Gaussian parameters (μ, Σ, w) to the EDT.
4.  **Convergence**: If MAE > threshold, Gaussian count is increased.

---

## Data Format (CSV)

Each line represents a single Gaussian component:
```
CubeX,CubeY,CubeZ,MAE,StdDev,G_ID,MeanX,MeanY,MeanZ,SigmaX,SigmaY,SigmaZ,Weight
```

| Field | Description |
|-------|-------------|
| `CubeX,Y,Z` | Origin of the cube |
| `MAE, StdDev` | Error metrics for the cube |
| `G_ID` | Gaussian index within cube |
| `MeanX,Y,Z` | Gaussian center (absolute) |
| `SigmaX,Y,Z` | Standard deviations |
| `Weight` | Gaussian weight |
