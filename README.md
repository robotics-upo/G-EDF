# G-EDF: Block-Sparse Gaussian Distance Fields

**G-EDF** is a high-performance C++ framework for modeling large-scale 3D environments as **continuous, memory-efficient Euclidean Distance Fields (EDF)**.

By leveraging a **Block-Sparse Gaussian Mixture Model**, G-EDF overcomes the limitations of discrete voxel grids, offering:

*   **Continuous Representation**: Infinite resolution with analytical gradients ($\|\nabla \hat{d}\| \approx 1$).
*   **High Compression**: Represents complex geometry with fewer parameters.
*   **CPU-Optimized**: Efficient parallel training and querying on standard CPUs.
*   **Precision**: Centimeter-level accuracy (MAE < 0.03m) for reliable navigation and physics.

## How It Works

The core idea is to use a Gaussian Mixture Model (GMM) not as a probability density, but as a function approximator. The distance field is represented as:

$$
\hat{d}(\mathbf{x}) = \sum_{k=1}^{K} w_k \cdot \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) \right)
$$


Where each Gaussian has a weight $w_k$, center $\boldsymbol{\mu}_k$, and diagonal covariance $\boldsymbol{\Sigma}_k$. Weights can be negative to carve sharp valleys near surfaces.

### Block-Sparse Architecture

To handle unbounded environments, the space is divided into a grid of cubes (default 1m³). Each cube trains an independent local GMM against its Euclidean Distance Transform (EDT). Adjacent cubes share an overlap margin and are blended using a Smoothstep function to guarantee global C¹ continuity, eliminating boundary artifacts.

### Key Features
*   **Adaptive Complexity**: Starts with few Gaussians; adds more only if error exceeds threshold.
*   **Analytical Gradients**: Closed-form gradient $\nabla \hat{d}(\mathbf{x})$ satisfies the Eikonal property ($\|\nabla \hat{d}\| \approx 1$).
*   **Euclidean Distance Field**: Models the unsigned distance to nearest surface.
*   **Scalable**: Processes 50M+ point clouds using multi-level KdTree and OpenMP parallelization.
*   **YAML Configuration**: All parameters tunable without recompilation.

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
  export_csv: true    # Debug-friendly CSV format
  export_bin: true    # Memory-efficient binary format
```

**Run**:
```bash
./build/gaussian_trainer config/config.yaml
```

Output: `<output_base>.csv` and/or `<output_base>.bin`

### 2. Reconstruction (`gaussian_to_ply`)
Generates a dense point cloud (isosurface) from a trained model.

**Configure** `config/reconstruction.yaml`:
```yaml
io:
  input_csv: "/path/to/model.csv"  # Supports .csv or .bin
  output_ply: "/path/to/output.ply"

reconstruction:
  threshold: 0.05    # Surface distance threshold (m)
  resolution: 0.02   # Sampling resolution (m)
  max_mae: 0.2       # Filter cubes with high error

blending:
  enabled: true      # Smooth interpolation between cubes
```

**Run**:
```bash
./build/gaussian_to_ply config/reconstruction.yaml
```

> **Note**: Modifying YAML files does **NOT** require recompilation.

---

## Configuration Reference

### Training (`config/config.yaml`)

| Section          | Parameter                  | Default      | Description                             |
| ---------------- | -------------------------- | ------------ | --------------------------------------- |
| **io**           | `input_file`               | `""`         | Path to input pointcloud (.ply or .pcd) |
|                  | `output_base`              | `""`         | Base name for output files              |
|                  | `export_csv`               | `true`       | Export human-readable CSV format        |
|                  | `export_bin`               | `true`       | Export memory-efficient binary format   |
| **processing**   | `num_threads`              | `0`          | OpenMP threads (0 = auto)               |
|                  | `cube_size`                | `1.0`        | Size of each cube in meters             |
| **downsampling** | `step`                     | `10`         | Take every Nth point for coarse KdTree  |
| **trainer**      | `sample_points`            | `1000`       | Sample points per cube                  |
|                  | `mae_threshold_good`       | `0.03`       | Target MAE for "good" fit (m)           |
|                  | `mae_threshold_max`        | `0.30`       | Max MAE before discarding (m)           |
| **solver**       | `max_iterations`           | `200`        | Max optimization iterations             |
|                  | `max_time_seconds`         | `0.3`        | Time budget per cube (s)                |
| **adaptive**     | `populated_steps`          | `[8,16,32]`  | Gaussian counts for populated cubes     |
|                  | `empty_steps`              | `[2,4,8,16]` | Gaussian counts for empty cubes         |
|                  | `mae_threshold`            | `0.03`       | Early stopping threshold (m)            |
|                  | `empty_distance_threshold` | `2.0`        | Max distance to train empty cubes       |
| **edt**          | `voxel_size`               | `0.05`       | EDT grid resolution (m)                 |
|                  | `margin`                   | `0.25`       | Grid margin for cube overlap (m)        |
|                  | `edt_extension`            | `0.25`       | Extra grid extension beyond margin (m)  |
|                  | `empty_search_margin`      | `0.25`       | Extra search radius for empty cubes (m) |
|                  | `empty_nearby_count`       | `100`        | Nearest neighbors for empty cube EDT    |

### Reconstruction (`config/reconstruction.yaml`)

| Section            | Parameter                         | Default | Description                               |
| ------------------ | --------------------------------- | ------- | ----------------------------------------- |
| **io**             | `input_csv`                       | `""`    | Path to Gaussian model (.csv or .bin)     |
|                    | `output_ply`                      | `""`    | Path to output PLY                        |
| **reconstruction** | `threshold`                       | `0.05`  | Surface detection threshold (m)           |
|                    | `resolution`                      | `0.02`  | Voxel sampling resolution (m)             |
|                    | `max_mae`                         | `100.0` | Max MAE allowed for reconstruction (m)    |
| **region**         | `enabled`                         | `false` | Enable region filtering                   |
|                    | `x_min/max, y_min/max, z_min/max` | `0.0`   | Region bounds                             |
| **blending**       | `enabled`                         | `true`  | Enable smooth interpolation between cubes |

---

## Technical Details

### Training Pipeline
For each cube:
1.  **EDT Generation**: Computes a local high-resolution Euclidean Distance Transform (EDT) grid.
2.  **Initialization**: Non-Maximum Suppression (NMS) identifies local extrema for Gaussian placement.
3.  **Optimization**: Levenberg-Marquardt (Ceres Solver) fits parameters $\{w_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}$ to minimize squared error.
4.  **Adaptive Refinement**: If MAE exceeds threshold, Gaussian count increases and optimization repeats.

### Blending Strategy
To ensure C¹ continuity across cube boundaries, overlapping regions are blended using Smoothstep:

$$
\alpha(t) = 3t^2 - 2t^3, \quad t \in [0,1]
$$


Since $\alpha'(0) = \alpha'(1) = 0$, gradients transition smoothly without artifacts. The final field at a point is the weighted average of contributing cubes based on their distance to cube boundaries.

---

## Data Formats

G-EDF supports two output formats:

### Binary Format (`.bin`)
Memory-efficient packed format optimized for fast loading. Contains:
- **MapHeader**: Magic bytes (`GDF1`), version, cube count, bounds, and training parameters
- **CubeHeader[]**: Per-cube origin, MAE, and Gaussian count
- **GaussianData[]**: Packed Gaussian parameters (mean, sigma, weight)

A comprehensive reference script is provided to parse `.bin` files, document the format, and inspect model internals.

1. **Inspect a GDF1 file:**
   
    Loads the binary map and prints:
   - **Header Metadata**: Version, global MAE/StdDev, bounding box, and parameters.
   - **Data Sample**: Detailed metrics and Gaussian parameters for the first cube (for debugging).

    ```bash
    python3 scripts/gdf1_loader_reference.py /path/to/map.bin
    ```
  

2. **View Format Specification & Integration Guide:**

    Prints the detailed **Binary Format Specification** (byte-level layout) and a **Implementation Guide** with Python code snippets showing how to mathematically query the distance field (including the blending logic). Use this if you need to implement a loader in another language.Ç

   ```bash
   python3 scripts/gdf1_loader_reference.py
   ```
   

### CSV Format (`.csv`)
Human-readable format for debugging and visualization. Each line represents a single Gaussian:
```
CubeX,CubeY,CubeZ,MAE,StdDev,G_ID,MeanX,MeanY,MeanZ,SigmaX,SigmaY,SigmaZ,Weight
```

| Field         | Description                |
| ------------- | -------------------------- |
| `CubeX,Y,Z`   | Origin of the cube         |
| `MAE, StdDev` | Error metrics for the cube |
| `G_ID`        | Gaussian index within cube |
| `MeanX,Y,Z`   | Gaussian center (absolute) |
| `SigmaX,Y,Z`  | Standard deviations        |
| `Weight`      | Gaussian weight            |
