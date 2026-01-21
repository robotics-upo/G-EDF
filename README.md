# GaussDF: Precise & Memory-Efficient Gaussian Distance Fields

GaussDF is a high-performance C++ pipeline designed to generate **precise offline maps** from large-scale 3D point clouds. By representing environments as a compact, block-sparse Gaussian Mixture Model (GMM), it provides a continuous **Distance Field (DF)** with centimeter-level accuracy (MAE < 0.03m).

The resulting maps are extremely lightweight and scalable to any environment size, making them ideal for **fast, real-time localization** and high-fidelity reconstruction in robotics applications.

## Overview

GaussDF partitions 3D space into millions of 1m³ voxels (cubes) and approximates local geometry using an adaptive set of 3D Gaussians. It achieves high compression ratios (storing complex surfaces in a lightweight CSV format) while enabling fast, continuous distance queries.

### Key Features
*   **Adaptive Training**: Automatically scales Gaussian count based on local geometric complexity.
*   **Dual Distance Modes**:
    *   **Pure (Unsigned)**: Direct Euclidean distance approximation (UDF).
    *   **Signed**: Surface-aware field using Felzenszwalb-based propagation (SDF).
*   **Massive Scale Support**: Optimized for clouds with 50M+ points using multi-level KdTree structures.
*   **Parallel Processing**: Fully multi-threaded execution via OpenMP.

## Dependencies

*   **C++17**
*   **PCL** (Point Cloud Library)
*   **Ceres Solver** (Non-linear optimization)
*   **OpenMP**

## Installation

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

### 1. Training (`gaussian_trainer`)
Converts a point cloud into a Gaussian-based Distance Field model.
```bash
./gaussian_trainer <input> <output_base> [threads] [samples] [mode]
```

### 2. Reconstruction (`gaussian_to_ply`)
Generates a dense point cloud (isosurface) from a trained model for visualization.
```bash
./gaussian_to_ply <model.csv> <output.ply> [threshold] [resolution] [bounds...]
```
*   `threshold`: Distance value defining the surface (default: 0.05m).
*   `resolution`: Voxel size for reconstruction (default: 0.02m).

---

## Configuration

GaussDF is highly tunable. Parameters can be adjusted via command-line arguments or directly in the source code.

### 1. Internal Logic (`src/main.cpp`)
These variables control the adaptive training and spatial search behavior:

| Variable | Default | Description |
|---|---|---|
| `populated_steps` | `{8, 16, 32}` | Gaussian counts tried for populated cubes. |
| `empty_steps` | `{2, 4, 8, 16}` | Gaussian counts tried for empty cubes. |
| `mae_threshold` | `0.03m` | Target Mean Absolute Error for early stopping. |
| `empty_dist_threshold`| `3.0m` | Max distance from surface to train empty cubes. |
| `search_radius_pop` | `1.5m` | Radius for gathering neighbor context in populated cubes. |
| `search_radius_empty`| `dist + 0.5m`| Dynamic radius for gathering points in empty cubes. |
| `edt_resolution` | `0.025m` | Resolution of the local EDT grid used for training. |

### 2. Solver Parameters (`include/solver/solver.hpp`)
Fine-tune the Ceres Solver optimization:

| Parameter | Default | Description |
|---|---|---|
| `max_iterations` | `250` | Maximum iterations per training attempt. |
| `max_time_seconds` | `2.0s` | Time limit for a single optimization attempt. |
| `function_tolerance` | `1e-4` | Convergence criteria for the objective function. |
| `use_importance_weighting`| `true/false`| Prioritizes accuracy near the surface (disabled for empty cubes). |

---

## Technical Architecture

### 1. Spatial Partitioning
The environment is divided into a 1m³ grid. Cubes are processed in parallel and classified:
*   **Populated**: Contains raw points. Uses an adaptive set of **8, 16, or 32** Gaussians.
*   **Empty**: No points, but within 3.0m of the surface. Uses **2, 4, 8, or 16** Gaussians.

### 2. Adaptive Training Pipeline
For each cube, the system executes an iterative optimization loop:
1.  **Context Gathering**: Fetches points within a 1.5x radius (2.6m) to ensure boundary continuity.
2.  **EDT Generation**: Computes a local Euclidean Distance Transform grid.
3.  **Optimization**: Uses Ceres Solver to fit Gaussian parameters $(\mu, \Sigma, w)$ to the EDT.
4.  **Convergence**: If Mean Absolute Error (MAE) > 0.03m, the Gaussian count is increased and the fit is refined.

### 3. Performance Optimizations
*   **Coarse KdTree**: A downsampled global index for fast distance checks on millions of empty cubes.
*   **Local KdTree**: Dynamic search radius based on exact distance to minimize local tree construction overhead.
*   **Importance Weighting**: Prioritizes surface accuracy in populated zones while ensuring global field consistency in empty regions.

## Data Formats

### CSV (Exchange)
A human-readable format for analysis. Each line represents a single Gaussian component:
`CubeX,CubeY,CubeZ,MAE,StdDev,G_ID,MeanX,MeanY,MeanZ,SigmaX,SigmaY,SigmaZ,Weight`

*   **CubeX, CubeY, CubeZ**: Origin of the 1m³ cube.
*   **MAE, StdDev**: Error metrics for the entire cube.
*   **G_ID**: Index of the Gaussian within the cube.
*   **MeanX, MeanY, MeanZ**: Absolute position of the Gaussian center.
*   **SigmaX, SigmaY, SigmaZ**: Standard deviations (widths) of the Gaussian.
*   **Weight**: Weight coefficient.
