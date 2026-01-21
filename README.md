# Gaussian SDF Trainer & Reconstructor

This project implements a pipeline to convert 3D pointclouds into a compact Gaussian Mixture Model (GMM) representation of the Signed Distance Function (SDF), and reconstruct them back to meshes/pointclouds.

## Overview

The system divides the pointcloud into 1m³ cubes and trains a set of Gaussians (default 16) for each cube to approximate the local geometry. It supports two modes:
1.  **Pure (Unsigned)**: Approximates the Euclidean distance to the nearest point.
2.  **Signed**: Uses the Felzenszwalb algorithm to estimate interior/exterior regions (Signed Distance Field).

## Dependencies

*   **C++17** compiler
*   **CMake** 3.14+
*   **PCL** (Point Cloud Library) - `libpcl-dev`
*   **Ceres Solver** - `libceres-dev`
*   **OpenMP** - For parallel processing

## Compilation

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

This will generate two executables:
*   `gaussian_trainer`: The main training application.
*   `gaussian_to_ply`: Tool to visualize the trained models.

## Usage

### 1. Training (`gaussian_trainer`)

Reads a pointcloud and trains Gaussians for each occupied cube.

```bash
./gaussian_trainer <input_cloud> <output_base> [threads] [samples] [mode]
```

*   **input_cloud**: Path to `.ply` or `.pcd` file.
*   **output_base**: Prefix for output files (e.g., `outputs/model` will create `outputs/model.csv` and `outputs/model.gsdf`).
*   **threads**: Number of OMP threads (default: max available).
*   **samples**: Number of points to sample per cube for training (default: 1000).
*   **mode**: `pure` (default) or `signed`.

**Example:**
```bash
# Train with 8 threads, 2000 samples per cube, in signed mode
./gaussian_trainer input.ply my_model 8 2000 pure
```

**Outputs:**
*   `*.csv`: Human-readable text format.
*   `*.gsdf`: Binary format (Gaussian SDF), more compact and faster to load.

### 2. Reconstruction (`gaussian_to_ply`)

Converts the trained Gaussian parameters back to a dense pointcloud (isosurface extraction) for visualization.

```bash
./gaussian_to_ply <input_model> <output_ply> [threshold] [resolution] [bounds...]
```

*   **input_model**: Path to `.csv` or `.gsdf` file from the trainer.
*   **output_ply**: Path to save the reconstructed PLY.
*   **threshold**: Iso-value threshold to define the surface (default: 0.05). Lower = thinner surface.
*   **resolution**: Step size for reconstruction grid in meters (default: 0.02). Smaller = higher quality but slower.
*   **bounds** (Optional): `xmin xmax ymin ymax zmin zmax` to reconstruction only a specific region.

**Example:**
```bash
# Reconstruct with 2cm resolution
./gaussian_to_ply my_model.gsdf reconstructed.ply 0.05 0.02
```

## File Formats

### CSV Format
Each line represents a single Gaussian, with the cube metadata repeated.
```
CubeX,CubeY,CubeZ,MAE,StdDev,G_ID,MeanX,MeanY,MeanZ,SigmaX,SigmaY,SigmaZ,Weight
```
*   **CubeX, CubeY, CubeZ**: Origin of the 1m³ cube.
*   **MAE, StdDev**: Error metrics for the entire cube.
*   **G_ID**: Index of the Gaussian within the cube (0-15).
*   **MeanX, MeanY, MeanZ**: Absolute position of the Gaussian center.
*   **SigmaX, SigmaY, SigmaZ**: Standard deviations (widths) of the Gaussian.
*   **Weight**: Weight coefficient.

### GSDF (Binary)
A custom binary format designed for efficient storage of the block-sparse Gaussian field.

---

## Algorithm Details

### Cube Classification

Each 1m³ cube is classified as either **Populated** (contains points) or **Empty** (no points but within distance threshold).

### Populated Cubes

1.  **Point Collection**: Points strictly inside the cube bounds are stored in `cube.point_indices`.
2.  **Context Search (1.5x)**: A radius search (`sqrt(3) * 1.5m ≈ 2.6m`) from the cube center gathers neighboring points *outside* the cube to provide boundary context.
3.  **Linear Densification**: Points inside the cube are optionally densified by interpolating between neighbors.
4.  **EDT Cloud**: The final cloud for EDT generation is `densified_cloud` (cube points + interpolated) + `context_indices` (raw neighbor points).
5.  **Training**: Adaptive loop tries `{8, 16, 32}` Gaussians until MAE ≤ 0.03m.

### Empty Cubes

1.  **Coarse Check**: A downsampled KdTree quickly checks if the nearest point is within `empty_distance_threshold` (3.0m).
2.  **Exact Distance**: If the coarse check passes, the global KdTree calculates the precise distance to the nearest point.
3.  **Dynamic Radius**: The search radius for EDT is set to `exact_dist + 0.5m`, minimizing the number of points gathered.
4.  **EDT Cloud**: All points within this dynamic radius are used to build a local KdTree.
5.  **Training**: Adaptive loop tries `{2, 4, 8, 16}` Gaussians with `positive_only=true` (no negative weights) and `use_importance_weighting=false` (to properly fit large distances).

### Solver Configuration

| Parameter | Value | Notes |
|---|---|---|
| `max_iterations` | 250 | Increased for better convergence |
| `max_time_seconds` | 2.0 | Time budget per training attempt |
| `mae_threshold_good` | 0.03 | Target error for early stopping |
| `mae_threshold_max` | 0.30 | Cubes above this are discarded |

### Performance Optimizations

*   **Coarse KdTree**: A downsampled version of the input cloud (1 point per 10-50) is used for fast distance checks on millions of empty cubes.
*   **Local KdTree**: For populated and nearby-empty cubes, a small local KdTree is built from relevant points, avoiding 64,000 searches against the full 66M-point cloud.
*   **Dynamic Search Radius**: For empty cubes, the search radius adapts to the actual distance, preventing unnecessary point gathering.
*   **Importance Weighting**: Disabled for empty cubes so the solver properly minimizes error for large distance values (which would otherwise be ignored).

