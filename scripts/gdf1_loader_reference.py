#!/usr/bin/env python3
"""
GDF1 Binary Map Format - Reference Implementation & Guide
=========================================================

This script serves as:
1. A documentation of the GDF1 binary format.
2. A reference Python implementation for loading these files.
3. A utility to inspect GDF1 files.

Usage:
    python3 gdf1_loader_reference.py [path_to_map.bin]

Format Specification (Little Endian):
-------------------------------------
1. Map Header (120 bytes):
   - Magic (4s): "GDF1"
   - Version (I): uint32
   - Num Cubes (I): uint32
   - Avg MAE (f): float32
   - Std Dev (f): float32
   - Bounds Min (3f): x, y, z
   - Bounds Max (3f): x, y, z
   - Cube Size (f): float32
   - Empty Margin (f): float32
   - Cube Margin (f): float32
   - Padding (64x): Reserved

2. Cube Data (Repeated 'Num Cubes' times):
   a. Cube Header (24 bytes):
      - Origin (3f): x, y, z (Bottom-left-back corner)s
      - MAE (f): float32
      - Std Dev (f): float32
      - Num Gaussians (I): uint32
   
   b. Gaussian Data (Repeated 'Num Gaussians' times):
      - ID (I): uint32
      - Mean (3f): x, y, z
      - Sigma (3f): sx, sy, sz (Standard deviation per axis)
      - Weight (f): float32
"""

import struct
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple

# ==========================================
# Data Structures
# ==========================================

@dataclass
class Gaussian:
    id: int
    mean: Tuple[float, float, float]
    sigma: Tuple[float, float, float]
    weight: float

@dataclass
class Cube:
    origin: Tuple[float, float, float]
    mae: float
    std_dev: float
    gaussians: List[Gaussian]

@dataclass
class MapHeader:
    magic: str
    version: int
    num_cubes: int
    avg_mae: float
    std_dev: float
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    cube_size: float
    empty_margin: float
    cube_margin: float

class GDF1Loader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header = None
        self.cubes = []

    def load(self, verbose=True):
        """Parses the binary file and populates the data structures."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        with open(self.filepath, 'rb') as f:
            # 1. Read Map Header
            # Format: 4s I I f f 3f 3f f f f 64x
            header_fmt = '<4sIIff3f3ffff64x'
            header_size = struct.calcsize(header_fmt)
            data = f.read(header_size)
            
            if len(data) != header_size:
                raise ValueError("Incomplete file header")

            (magic, version, num_cubes, avg_mae, std_dev, 
             min_x, min_y, min_z, max_x, max_y, max_z, 
             c_size, e_margin, c_margin) = struct.unpack(header_fmt, data)

            if magic != b'GDF1':
                raise ValueError(f"Invalid format. Expected 'GDF1', got {magic}")

            self.header = MapHeader(
                magic=magic.decode('utf-8'),
                version=version,
                num_cubes=num_cubes,
                avg_mae=avg_mae,
                std_dev=std_dev,
                bounds_min=(min_x, min_y, min_z),
                bounds_max=(max_x, max_y, max_z),
                cube_size=c_size,
                empty_margin=e_margin,
                cube_margin=c_margin
            )

            if verbose:
                self._print_header_info()

            # 2. Read Cubes
            # Pre-calc struct sizes for performance
            cube_header_fmt = '<3fffI'
            cube_header_size = struct.calcsize(cube_header_fmt)
            
            gauss_fmt = '<I3f3ff'
            gauss_size = struct.calcsize(gauss_fmt)

            print(f"Loading {num_cubes} cubes...")
            
            for i in range(num_cubes):
                # Read Cube Header
                data = f.read(cube_header_size)
                if len(data) != cube_header_size:
                    break
                
                ox, oy, oz, mae, c_std, num_gauss = struct.unpack(cube_header_fmt, data)
                
                # Read Gaussians for this cube
                gaussians = []
                for _ in range(num_gauss):
                    g_data = f.read(gauss_size)
                    gid, mx, my, mz, sx, sy, sz, w = struct.unpack(gauss_fmt, g_data)
                    gaussians.append(Gaussian(gid, (mx, my, mz), (sx, sy, sz), w))

                self.cubes.append(Cube((ox, oy, oz), mae, c_std, gaussians))

                if verbose and i % 10000 == 0:
                    sys.stdout.write(f"\rProgress: {i}/{num_cubes}")
                    sys.stdout.flush()
            
            if verbose:
                print("\rDone.                  ")

    def _print_header_info(self):
        h = self.header
        print("\n" + "="*40)
        print(f" GDF1 MAP INFO: {os.path.basename(self.filepath)}")
        print("="*40)
        print(f" Magic:        {h.magic}")
        print(f" Version:      {h.version}")
        print(f" Cubes:        {h.num_cubes}")
        print(f" Avg MAE:      {h.avg_mae:.4f} m")
        print(f" Std Dev:      {h.std_dev:.4f}")
        print(f" Cube Size:    {h.cube_size:.2f} m")
        print(f" Empty Margin: {h.empty_margin:.2f} m")
        print(f" Blending Mar: {h.cube_margin:.2f} m")
        print(f" Bounds Min:   {h.bounds_min}")
        print(f" Bounds Max:   {h.bounds_max}")
        print("-" * 40 + "\n")

def print_guide():
    print(__doc__)
    print("="*60)
    print(" IMPLEMENTATION GUIDE: How to Query the Map")
    print("="*60)
    
    print("""
1. Mathematical Model
---------------------
The value at any point p=(x,y,z) is the weighted sum of Gaussians:

    f(p) = Σ w_i * exp(-0.5 * D_i^2)

Where D_i^2 is the Mahalanobis distance:
    D_i^2 = (x-μx)^2/lx + (y-μy)^2/ly + (z-μz)^2/lz

    - μ (mu): Mean of the Gaussian (center)
    - l (lambda): Scale parameter (sigma^4)
    - w: Weight

2. Querying a Point (Code Examples)
-----------------------------------

Case A: Interior Point (Fast Path)
----------------------------------
If a point p lies in the center of a cube (not in the blending margin), 
you only need to sum the Gaussians of that single cube.

    def predict_cube(p, cube):
        val = 0.0
        for g in cube.gaussians:
            dx = p.x - g.mean.x
            dy = p.y - g.mean.y
            dz = p.z - g.mean.z
            
            # Squared distance normalized by scale (sigma^4)
            dsq = (dx**2)/(g.sigma.x**4) + (dy**2)/(g.sigma.y**4) + (dz**2)/(g.sigma.z**4)
            
            val += g.weight * math.exp(-0.5 * dsq)
        return val

Case B: Border Point (Blending Path)
------------------------------------
If a point lies within the 'margin' distance of a cube boundary, 
you must blend the predictions of the current cube and its neighbors.

    # 1. Identify Neighbors (3x3x3 block)
    # 2. Calculate Blend Weight for each valid neighbor
    
    def blend_weight(p, cube, margin):
        # Distance from point to cube boundary (inside is positive)
        dist_x = min(p.x - cube.x, cube.x + size - p.x)
        ... (repeat for y, z)
        min_dist = min(dist_x, dist_y, dist_z)
        
        if min_dist >= 0: return 1.0       # Fully inside
        if min_dist <= -margin: return 0.0 # Fully outside
        
        # Smooth interpolation (0 to 1)
        t = 1.0 + min_dist / margin
        return t * t * (3.0 - 2.0 * t)

    # 3. Weighted Average
    final_val = sum(w_i * predict_cube(p, cube_i)) / sum(w_i)
""")

def main():
    if len(sys.argv) < 2:
        print_guide()
        print("\n" + "="*60)
        print(" [TIP] Run with a file to see inspection output:")
        print("       python3 gdf1_loader_reference.py /path/to/map.bin")
        print("="*60)
        return

    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return

    try:
        loader = GDF1Loader(filepath)
        loader.load(verbose=True)
        
        # Improved Inspection Output
        if loader.cubes:
            print("\n" + "="*40)
            print(f" DATA INSPECTION: First Cube Sample")
            print("="*40)
            c = loader.cubes[0]
            
            print(f"Origin:    ({c.origin[0]:.2f}, {c.origin[1]:.2f}, {c.origin[2]:.2f})")
            print(f"Metrics:   MAE={c.mae:.4f}, StdDev={c.std_dev:.4f}")
            print(f"Gaussians: {len(c.gaussians)}")
            
            if c.gaussians:
                print("\nFirst 5 Gaussians:")
                print(f"{'ID':<5} | {'Mean (x,y,z)':<25} | {'Weight':<10}")
                print("-" * 45)
                for g in c.gaussians[:5]:
                    m_str = f"({g.mean[0]:.2f}, {g.mean[1]:.2f}, {g.mean[2]:.2f})"
                    print(f"{g.id:<5} | {m_str:<25} | {g.weight:.4f}")
                
                if len(c.gaussians) > 5:
                    print(f"... and {len(c.gaussians)-5} more.")
            print("="*40 + "\n")
                
    except Exception as e:
        print(f"\nError loading map: {e}")

if __name__ == "__main__":
    main()
