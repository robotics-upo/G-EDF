import struct
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ==========================================
# Configuration
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(SCRIPT_DIR, "..", "config", "plots.yaml")

def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file if it exists."""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # print(f"Loaded configuration from: {config_path}") # Reduced verbosity
                return config if config else {}
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            return {}
    return {}

# Load YAML config at module level
_yaml_config = load_yaml_config(CONFIG_FILE)

@dataclass
class Config:
    """Global configuration for the visualization script."""
    # Input File (Set this to your .bin file path)
    INPUT_FILE: str = _yaml_config.get('input_file', "output.bin")
    
    # Slice Configuration
    AXIS: str = _yaml_config.get('slice', {}).get('axis', 'z')
    SLICE_POSITION: float = _yaml_config.get('slice', {}).get('position', 0.5)
    
    # Region of Interest (Plotting Bounds)
    # If 0.0, they will be auto-set from the map bounds
    MIN_U: float = _yaml_config.get('region', {}).get('min_u', 0.0)
    MAX_U: float = _yaml_config.get('region', {}).get('max_u', 0.0)
    MIN_V: float = _yaml_config.get('region', {}).get('min_v', 0.0)
    MAX_V: float = _yaml_config.get('region', {}).get('max_v', 0.0)
    
    RESOLUTION: float = _yaml_config.get('resolution', 0.05)
    
    # Blending Defaults
    ENABLE_BLENDING: bool = _yaml_config.get('blending', {}).get('enable', False)
    OVERRIDE_MARGIN: Optional[float] = _yaml_config.get('blending', {}).get('override_margin', None)
    
    # Plotting
    CMAP: str = _yaml_config.get('plotting', {}).get('colormap', 'RdBu')
    INVERT_CMAP: bool = _yaml_config.get('plotting', {}).get('invert_colormap', False)
    LEVELS: int = _yaml_config.get('plotting', {}).get('levels', 41)
    SHOW_GRID: bool = _yaml_config.get('plotting', {}).get('show_grid', True)
    SHOW_ZERO_LEVEL: bool = _yaml_config.get('plotting', {}).get('show_zero_level', True)
    FIG_SIZE: Tuple[int, int] = tuple(_yaml_config.get('plotting', {}).get('figure_size', [10, 8]))
    SHOW_COLORBAR: bool = _yaml_config.get('plotting', {}).get('show_colorbar', True)
    FONT_SIZE: int = _yaml_config.get('plotting', {}).get('font_size', 12)
    
    # Visualization mode
    MODE: str = _yaml_config.get('visualization', {}).get('mode', 'edf')
    VMIN: Optional[float] = _yaml_config.get('visualization', {}).get('vmin', None)
    VMAX: Optional[float] = _yaml_config.get('visualization', {}).get('vmax', None)
    
    # Statistics
    GRADIENT_MIN_SDF: float = _yaml_config.get('statistics', {}).get('gradient_min_sdf', 0.1)

    # Output
    OUTPUT_DIR: str = _yaml_config.get('output_dir', "sdf_gradient_images")

# ==========================================
# Data Structures
# ==========================================
@dataclass
class Gaussian:
    x: float
    y: float
    z: float
    l0: float # sigma_x^4
    l1: float # sigma_y^4
    l2: float # sigma_z^4
    w: float

@dataclass
class Cube:
    ox: float
    oy: float
    oz: float
    gaussians: List[Gaussian]

# ==========================================
# Binary Loader
# ==========================================
def load_binary_map(path: str) -> Tuple[Dict[Tuple[int, int, int], Cube], float, float]:
    """
    Loads the GDF1 binary map format.
    Returns: (cubes_dict, cube_size, margin, bounds)
    bounds is (min_x, max_x, min_y, max_y, min_z, max_z)
    """
    print(f"Loading binary map: {path}")
    cubes = {}
    cube_size = 1.0
    margin = 0.0
    bounds = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    try:
        with open(path, 'rb') as f:
            # 1. Read Map Header
            header_fmt = '<4sIIff3f3ffff64x'
            header_size = struct.calcsize(header_fmt)
            data = f.read(header_size)
            if len(data) != header_size:
                raise ValueError("Invalid file header")
                
            (magic, version, num_cubes, avg_mae, std_dev, 
             min_x, min_y, min_z, max_x, max_y, max_z, 
             c_size, empty_margin, c_margin) = struct.unpack(header_fmt, data)
            
            if magic != b'GDF1':
                raise ValueError(f"Invalid magic bytes: {magic}")
                
            print(f"Map Info: {num_cubes} cubes, Size={c_size:.2f}m, Margin={c_margin:.2f}m")
            print(f"Bounds: [{min_x:.2f}, {max_x:.2f}] x [{min_y:.2f}, {max_y:.2f}] x [{min_z:.2f}, {max_z:.2f}]")
            cube_size = c_size
            margin = c_margin
            bounds = (min_x, max_x, min_y, max_y, min_z, max_z)

            # 2. Read Cubes
            for _ in range(num_cubes):
                ch_fmt = '<3fffI'
                ch_size = struct.calcsize(ch_fmt)
                data = f.read(ch_size)
                if len(data) != ch_size:
                    break
                    
                ox, oy, oz, mae, c_std, ng = struct.unpack(ch_fmt, data)
                
                gaussians = []
                gd_fmt = '<I3f3ff'
                gd_size = struct.calcsize(gd_fmt)
                
                for _ in range(ng):
                    g_data = f.read(gd_size)
                    if len(g_data) != gd_size:
                        break
                    gid, mx, my, mz, sx, sy, sz, w = struct.unpack(gd_fmt, g_data)
                    
                    eps = 1e-9
                    l0 = sx**4 + eps
                    l1 = sy**4 + eps
                    l2 = sz**4 + eps
                    
                    gaussians.append(Gaussian(mx, my, mz, l0, l1, l2, w))
                    
                ix = int(round(ox / cube_size))
                iy = int(round(oy / cube_size))
                iz = int(round(oz / cube_size))
                
                cubes[(ix, iy, iz)] = Cube(ox, oy, oz, gaussians)
                
    except FileNotFoundError:
        print(f"Error: File {path} not found.")
        exit(1)
    except Exception as e:
        print(f"Error reading binary file: {e}")
        exit(1)

    return cubes, cube_size, margin, bounds

# ==========================================
# Blending Logic
# ==========================================
def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def blend_weight(px, py, pz, cx, cy, cz, cube_size, margin):
    dist_x = min(px - cx, cx + cube_size - px)
    dist_y = min(py - cy, cy + cube_size - py)
    dist_z = min(pz - cz, cz + cube_size - pz)
    min_dist = min(dist_x, dist_y, dist_z)

    if min_dist >= 0:
        return 1.0
    if min_dist <= -margin:
        return 0.0
    
    return smoothstep(1.0 + min_dist / margin)

def predict(x, y, z, gaussians: List[Gaussian]):
    val = 0.0
    for g in gaussians:
        dx = x - g.x
        dy = y - g.y
        dz = z - g.z
        dsq = (dx*dx)/g.l0 + (dy*dy)/g.l1 + (dz*dz)/g.l2
        val += g.w * math.exp(-0.5 * dsq)
    return val

def predict_gradient(x, y, z, gaussians: List[Gaussian]) -> float:
    """Computes the magnitude of the analytical 3D gradient."""
    grad_x = 0.0
    grad_y = 0.0
    grad_z = 0.0
    
    for g in gaussians:
        dx = x - g.x
        dy = y - g.y
        dz = z - g.z
        
        # dsq = (dx^2)/l0 + ...
        dsq = (dx*dx)/g.l0 + (dy*dy)/g.l1 + (dz*dz)/g.l2
        
        # Derivative of gaussian:
        # d/dx (w * exp(-0.5 * dsq)) 
        # = w * exp(...) * (-0.5) * d(dsq)/dx
        # = w * exp(...) * (-0.5) * (2 * dx / l0)
        # = -w * exp(...) * dx / l0
        
        exp_val = math.exp(-0.5 * dsq)
        term = -g.w * exp_val
        
        grad_x += term * (dx / g.l0)
        grad_y += term * (dy / g.l1)
        grad_z += term * (dz / g.l2)
        
    return math.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

# ==========================================
# Main Script
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Visualize EDT Slice from Binary Gaussian Map")
    parser.add_argument("bin_file", nargs='?', default=Config.INPUT_FILE, help="Path to input BIN map")
    parser.add_argument("--axis", choices=['x', 'y', 'z'], default=Config.AXIS, help="Slice axis")
    parser.add_argument("--pos", type=float, default=Config.SLICE_POSITION, help="Position along slice axis")
    parser.add_argument("--min_u", type=float, default=Config.MIN_U, help="Min coord for first plot axis")
    parser.add_argument("--max_u", type=float, default=Config.MAX_U, help="Max coord for first plot axis")
    parser.add_argument("--min_v", type=float, default=Config.MIN_V, help="Min coord for second plot axis")
    parser.add_argument("--max_v", type=float, default=Config.MAX_V, help="Max coord for second plot axis")
    parser.add_argument("--res", type=float, default=Config.RESOLUTION, help="Resolution (meters)")
    parser.add_argument("--blending", action="store_true", default=Config.ENABLE_BLENDING, help="Enable blending")
    parser.add_argument("--margin", type=float, default=Config.OVERRIDE_MARGIN, help="Override blending margin")
    parser.add_argument("--mode", choices=['edf', 'gradient'], default=Config.MODE, help="Visualization mode")
    parser.add_argument("--vmin", type=float, default=Config.VMIN, help="Min value for color scale")
    parser.add_argument("--vmax", type=float, default=Config.VMAX, help="Max value for color scale")
    parser.add_argument("--output_dir", type=str, default=Config.OUTPUT_DIR, help="Directory to save output plots")

    args = parser.parse_args()

    # Set global font size
    plt.rcParams.update({'font.size': Config.FONT_SIZE})
    
    # Load Map
    cubes, cube_size, file_margin, bounds = load_binary_map(args.bin_file)
    (bx_min, bx_max, by_min, by_max, bz_min, bz_max) = bounds
    
    # Auto-detect bounds if 0
    min_u, max_u = args.min_u, args.max_u
    min_v, max_v = args.min_v, args.max_v
    
    # Determine file bounds for U and V based on axis
    if args.axis == 'z':
        # U=x, V=y
        fu_min, fu_max = bx_min, bx_max
        fv_min, fv_max = by_min, by_max
    elif args.axis == 'y':
        # U=x, V=z
        fu_min, fu_max = bx_min, bx_max
        fv_min, fv_max = bz_min, bz_max
    else: # x
        # U=y, V=z
        fu_min, fu_max = by_min, by_max
        fv_min, fv_max = bz_min, bz_max

    if min_u == 0.0 and max_u == 0.0:
        min_u, max_u = fu_min, fu_max
        print(f"Auto-set U bounds: [{min_u:.2f}, {max_u:.2f}]")
        
    if min_v == 0.0 and max_v == 0.0:
        min_v, max_v = fv_min, fv_max
        print(f"Auto-set V bounds: [{min_v:.2f}, {max_v:.2f}]")

    margin = file_margin
    if args.margin is not None:
        margin = args.margin
        print(f"Margin overridden: {margin}")

    print(f"Configuration: Axis={args.axis}, Pos={args.pos}, Res={args.res}m, Blending={args.blending}")
    
    # Define grid
    u_vals = np.arange(min_u, max_u, args.res)
    v_vals = np.arange(min_v, max_v, args.res)
    U, V = np.meshgrid(u_vals, v_vals)
    Z_grid = np.zeros_like(U)
    Grad_grid = np.zeros_like(U) # Store 3D gradient norm
    
    print(f"Computing slice... (Grid: {U.shape})")
    
    total_pts = U.size
    count = 0
    
    # Compute SDF
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            u, v = U[i, j], V[i, j]
            
            if args.axis == 'z':
                x, y, z = u, v, args.pos
            elif args.axis == 'y':
                x, y, z = u, args.pos, v
            else: # x
                x, y, z = args.pos, u, v
            
            val = 0.0
            grad_val = np.nan
            
            if not args.blending:
                # Find single cube
                ix = int(math.floor(x / cube_size))
                iy = int(math.floor(y / cube_size))
                iz = int(math.floor(z / cube_size))
                
                if (ix, iy, iz) in cubes:
                    c = cubes[(ix, iy, iz)]
                    if (x >= c.ox and x < c.ox + cube_size and
                        y >= c.oy and y < c.oy + cube_size and
                        z >= c.oz and z < c.oz + cube_size):
                        val = predict(x, y, z, c.gaussians)
                        grad_val = predict_gradient(x, y, z, c.gaussians)
                    else:
                        val = np.nan
                else:
                    val = np.nan
            else:
                # Blending logic
                px_i = int(math.floor(x))
                py_i = int(math.floor(y))
                pz_i = int(math.floor(z))
                
                weighted_sum = 0.0
                weighted_grad_sum = 0.0
                weight_total = 0.0
                
                # Check 3x3x3 neighbors (center + 26 neighbors)
                # We iterate -1 to 1 to cover all potential overlapping cubes
                for dz in range(-1, 2):
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            key = (px_i + dx, py_i + dy, pz_i + dz)
                            if key not in cubes: continue
                            
                            c = cubes[key]
                            
                            # Strict check: is the point actually within the influence zone of this neighbor?
                            if (x < c.ox - margin or x > c.ox + cube_size + margin or
                                y < c.oy - margin or y > c.oy + cube_size + margin or
                                z < c.oz - margin or z > c.oz + cube_size + margin):
                                continue
                                
                            w = blend_weight(x, y, z, c.ox, c.oy, c.oz, cube_size, margin)
                            if w > 1e-6:
                                pred = predict(x, y, z, c.gaussians)
                                grad = predict_gradient(x, y, z, c.gaussians)
                                
                                weighted_sum += w * pred
                                weighted_grad_sum += w * grad
                                weight_total += w
                
                if weight_total > 1e-6:
                    val = weighted_sum / weight_total
                    grad_val = weighted_grad_sum / weight_total
                else:
                    val = np.nan

            Z_grid[i, j] = val
            Grad_grid[i, j] = grad_val
            
            count += 1
            if count % (total_pts // 10) == 0:
                print(f"\rProgress: {int(count/total_pts*100)}%", end="")
    
    print("\rProgress: 100%")
    print("-" * 40)
    print(f"{'STATISTICS':^40}")
    print("-" * 40)
    
    # EDF Metrics
    valid_sdf = Z_grid[~np.isnan(Z_grid)]
    if len(valid_sdf) > 0:
        print(f"{'Metric':<20} | {'Value':<15}")
        print("-" * 40)
        print(f"{'EDF Min':<20} | {np.min(valid_sdf):.4f} m")
        print(f"{'EDF Max':<20} | {np.max(valid_sdf):.4f} m")
    else:
        print("No valid EDF data found.")

    # Gradient Metrics (Analytical 3D)
    valid_grad = Grad_grid[~np.isnan(Grad_grid)]
    
    if len(valid_grad) > 0:
        print(f"{'Grad Norm Mean':<20} | {np.mean(valid_grad):.4f}")
        print(f"{'Grad Norm Std':<20} | {np.std(valid_grad):.4f}")
    
    # Filtered Gradient Metrics (SDF > Threshold)
    if Config.GRADIENT_MIN_SDF is not None:
        mask = (Z_grid > Config.GRADIENT_MIN_SDF) & (~np.isnan(Grad_grid))
        filtered_grad = Grad_grid[mask]
        
        print("-" * 40)
        print(f"Filtered Gradient (SDF > {Config.GRADIENT_MIN_SDF}m):")
        if len(filtered_grad) > 0:
            print(f"{'Filt Grad Mean':<20} | {np.mean(filtered_grad):.4f}")
            print(f"{'Filt Grad Std':<20} | {np.std(filtered_grad):.4f}")
            print(f"{'Points Included':<20} | {len(filtered_grad)} ({len(filtered_grad)/len(valid_grad)*100:.1f}%)")
        else:
            print("No points satisfy the SDF threshold condition.")

    print("-" * 40)

    print("Generating plot...")
    
    plt.figure(figsize=Config.FIG_SIZE)
    
    if args.mode == 'gradient':
        cmap = plt.cm.viridis
        # Use analytical gradient for plotting
        plot_data = Grad_grid
        vmin = args.vmin if args.vmin is not None else np.nanmin(plot_data)
        vmax = args.vmax if args.vmax is not None else np.nanmax(plot_data)
        levels = np.linspace(vmin, vmax, 50)
        cf = plt.contourf(U, V, plot_data, levels=levels, cmap=cmap, extend='both')
        if Config.SHOW_COLORBAR:
            # Custom ticks: Min, 1.0, Max
            tick_min = args.vmin if args.vmin is not None else vmin
            tick_max = args.vmax if args.vmax is not None else vmax
            
            cbar_ticks = [tick_min, 1.0, tick_max]
            # Filter ticks that are out of bounds or too close
            cbar_ticks = sorted(list(set([t for t in cbar_ticks if tick_min <= t <= tick_max])))
            
            cbar = plt.colorbar(cf, ticks=cbar_ticks, label='Analytical Gradient Magnitude ||∇EDF||')
            cbar.ax.set_yticklabels([f"{t:.1f}" for t in cbar_ticks])
        # plt.title(f"Analytical Gradient Norm ({args.axis}={args.pos})")
    else:
        cmap_name = Config.CMAP
        if Config.INVERT_CMAP:
            # Append '_r' to reverse the colormap (darker = closer to obstacle)
            if not cmap_name.endswith('_r'):
                cmap_name = cmap_name + '_r'
            else:
                cmap_name = cmap_name[:-2]  # Remove '_r' if already reversed
        try:
            cmap = plt.colormaps[cmap_name]
        except AttributeError:
            cmap = plt.cm.get_cmap(cmap_name)
        vmin = args.vmin if args.vmin is not None else -1.0
        vmax = args.vmax if args.vmax is not None else 1.0
        
        from matplotlib.colors import TwoSlopeNorm, Normalize
        levels = np.linspace(vmin, vmax, Config.LEVELS)
        
        if vmin >= 0:
            # Pure positive range: use standard Normalize (no need to center at 0)
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            # Mixed range: center at 0 using TwoSlopeNorm
            if vmax <= 0:
                vmax = 0.001
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
        
        cf = plt.contourf(U, V, Z_grid, levels=levels, cmap=cmap, norm=norm, extend='both')
        if Config.SHOW_COLORBAR:
            # Custom ticks: Min, 1.0, Max
            # Use user-defined limits if provided, otherwise use calculated vmin/vmax
            tick_min = args.vmin if args.vmin is not None else vmin
            tick_max = args.vmax if args.vmax is not None else vmax
            
            cbar_ticks = [tick_min, 1.0, tick_max]
            # Filter ticks that are out of bounds or too close
            cbar_ticks = sorted(list(set([t for t in cbar_ticks if tick_min <= t <= tick_max])))
            cbar = plt.colorbar(cf, ticks=cbar_ticks, label='EDF Value (m)')
            cbar.ax.set_yticklabels([f"{t:.1f}" for t in cbar_ticks])
        
        if Config.SHOW_ZERO_LEVEL:
            plt.contour(U, V, Z_grid, levels=[0], colors='k', linewidths=2)
            
        # plt.title(f"EDF Slice ({args.axis}={args.pos})")
    
    if Config.SHOW_GRID:
        for u_line in range(int(min_u), int(max_u) + 1):
            plt.axvline(x=u_line, color='gray', linestyle=':', alpha=0.3)
        for v_line in range(int(min_v), int(max_v) + 1):
            plt.axhline(y=v_line, color='gray', linestyle=':', alpha=0.3)

    plt.xlabel(f"{'x' if args.axis != 'x' else 'y'} (m)")
    plt.ylabel(f"{'y' if args.axis == 'z' else 'z'} (m)")
    plt.xlabel(f"{'x' if args.axis != 'x' else 'y'} (m)")
    plt.ylabel(f"{'y' if args.axis == 'z' else 'z'} (m)")
    
    # Enforce strict bounds
    plt.xlim(min_u, max_u)
    plt.ylim(min_v, max_v)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Format axis ticks to 1 decimal place and reduce number of ticks
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    input_basename = os.path.splitext(os.path.basename(args.bin_file))[0]
    output_filename = f"{input_basename}_{args.axis}_{args.pos}_{args.mode}_blend{args.blending}.png"

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_filename)

    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved plot to: {output_file}")
    # plt.show() # Commented out for headless environments

if __name__ == "__main__":
    main()
