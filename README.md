# Gaussian SDF: High-Performance Surface Modeling

A high-performance C++ pipeline for representing large-scale 3D point clouds as a compact, block-sparse Gaussian Mixture Model (GMM) of the Signed Distance Function (SDF).

## Overview

This system partitions 3D space into 1m³ voxels (cubes) and approximates the local geometry using an adaptive number of 3D Gaussians. It provides a significant compression ratio over raw point clouds while enabling fast, continuous SDF queries for robotics and reconstruction tasks.

### Key Features
*   **Adaptive Training**: Automatically scales Gaussian count based on local geometric complexity.
*   **Dual SDF Modes**:
    *   **Pure (Unsigned)**: Direct Euclidean distance approximation.
    *   **Signed**: Surface-aware field using Felzenszwalb-based propagation.
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
Converts a point cloud into a Gaussian-based SDF model.

```bash
./gaussian_trainer <input.ply/pcd> <output_prefix> [threads] [samples] [mode]
```
*   `mode`: `pure` (default) or `signed`.
*   `samples`: Points sampled per cube for optimization (default: 1000).

### 2. Reconstruction (`gaussian_to_ply`)
Generates a dense point cloud (isosurface) from a trained model.

```bash
./gaussian_to_ply <model.csv> <output.ply> [threshold] [resolution] [bounds...]
```
*   `threshold`: Distance value defining the surface (default: 0.05m).
*   `resolution`: Voxel size for reconstruction (default: 0.02m).
*   `bounds`: Optional `xmin xmax ymin ymax zmin zmax` to reconstruct a specific region.

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
