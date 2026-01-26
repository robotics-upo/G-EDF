#ifndef GAUSSIAN_TRAINER_HPP
#define GAUSSIAN_TRAINER_HPP

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include "edt_generator.hpp"
#include "solver/solver.hpp"

/// Training result for a single cube
struct TrainResult {
    std::vector<double> params;
    double mae = 0.0;
    double std_dev = 0.0;
    int num_gaussians = 0;
    bool valid = false;
};

/// Trainer configuration
struct TrainerConfig {
    int sample_points = 2000; 
    double mae_threshold_good = 0.02; 
    double mae_threshold_max = 0.20;
    EDTMode edt_mode = EDTMode::PURE;
    SolverConfig solver;
    bool positive_only = false;
    float margin = 0.0f;
};

struct Peak {
    int idx;
    double val;
};

struct Coord {
    int x, y, z;
};

/// Helper to check for local extremum
inline bool is_local_extremum(const LocalGrid& grid, int x, int y, int z, bool find_max, int radius) {
    double center_val = grid.sdf_data[grid.idx(x, y, z)];

    for (int dz = -radius; dz <= radius; ++dz) {
        int nz = z + dz;
        if (nz < 0 || nz >= grid.nz) continue;

        for (int dy = -radius; dy <= radius; ++dy) {
            int ny = y + dy;
            if (ny < 0 || ny >= grid.ny) continue;

            for (int dx = -radius; dx <= radius; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) continue;

                int nx = x + dx;
                if (nx < 0 || nx >= grid.nx) continue;

                double val = grid.sdf_data[grid.idx(nx, ny, nz)];

                if (find_max) {
                    if (val > center_val) return false;
                } else {
                    if (val < center_val) return false;
                }
            }
        }
    }
    return true;
}

inline Coord idx_to_coord(int idx, int nx, int ny) {
    int z = idx / (nx * ny);
    int rem = idx % (nx * ny);
    int y = rem / nx;
    int x = rem % nx;
    return {x, y, z};
}

/// Initialize Gaussians using NMS (Non-Maximum Suppression) on SDF peaks/valleys
inline std::vector<double> initializeSmartGaussians(const LocalGrid& grid,
                                                     int num_gaussians,
                                                     bool positive_only = false,
                                                     unsigned int seed = 42) {
    std::vector<double> params(num_gaussians * PARAMS_PER_GAUSSIAN);
    std::mt19937 rng(seed);

    double init_log_sigma = std::sqrt(0.08); 
    double w_init = 0.1;                     

    int peak_radius = 3;
    int suppression_radius = 3;

    std::vector<Peak> pos_candidates;
    std::vector<Peak> neg_candidates;

    // Scan ONLY inside the cube (ignoring margin)
    int ix_start = std::max(peak_radius, int((grid.cube_origin_x - grid.origin_x) / grid.voxel_size));
    int iy_start = std::max(peak_radius, int((grid.cube_origin_y - grid.origin_y) / grid.voxel_size));
    int iz_start = std::max(peak_radius, int((grid.cube_origin_z - grid.origin_z) / grid.voxel_size));
    
    int n_cube_voxels = int(grid.cube_size / grid.voxel_size);
    int ix_end = std::min(grid.nx - peak_radius, ix_start + n_cube_voxels);
    int iy_end = std::min(grid.ny - peak_radius, iy_start + n_cube_voxels);
    int iz_end = std::min(grid.nz - peak_radius, iz_start + n_cube_voxels);

    for (int z = iz_start; z < iz_end; ++z) {
        for (int y = iy_start; y < iy_end; ++y) {
            for (int x = ix_start; x < ix_end; ++x) {
                int idx = grid.idx(x, y, z);
                double val = grid.sdf_data[idx];

                // Positive Gaussians: Peaks (Maxima)
                // Keep a small threshold to avoid noise near zero
                if (val > 1e-3) {
                    if (is_local_extremum(grid, x, y, z, true, peak_radius))
                        pos_candidates.push_back({idx, val});
                } 
                
                // Negative Gaussians: Valleys (Minima)
                // In Signed mode: Deepest negative values.
                // In Pure mode: Values closest to zero (surface).
                // We accept any local minimum.
                if (is_local_extremum(grid, x, y, z, false, peak_radius)) {
                    neg_candidates.push_back({idx, val});
                }
            }
        }
    }

    // Sort candidates
    std::sort(pos_candidates.begin(), pos_candidates.end(), [](const Peak &a, const Peak &b) { return a.val > b.val; });
    std::sort(neg_candidates.begin(), neg_candidates.end(), [](const Peak &a, const Peak &b) { return a.val < b.val; });

    // NMS Function
    auto run_nms = [&](const std::vector<Peak> &candidates, int target_count) -> std::vector<int> {
        std::vector<int> selected_indices;
        std::vector<bool> suppressed(grid.nx * grid.ny * grid.nz, false);

        for (const auto &p : candidates) {
            if ((int)selected_indices.size() >= target_count) break;
            if (suppressed[p.idx]) continue;

            selected_indices.push_back(p.idx);

            Coord c = idx_to_coord(p.idx, grid.nx, grid.ny);
            int r = suppression_radius;

            for (int dz = -r; dz <= r; ++dz) {
                int nz = c.z + dz;
                if (nz < 0 || nz >= grid.nz) continue;
                for (int dy = -r; dy <= r; ++dy) {
                    int ny = c.y + dy;
                    if (ny < 0 || ny >= grid.ny) continue;
                    for (int dx = -r; dx <= r; ++dx) {
                        int nx = c.x + dx;
                        if (nx < 0 || nx >= grid.nx) continue;
                        suppressed[grid.idx(nx, ny, nz)] = true;
                    }
                }
            }
        }
        return selected_indices;
    };

    int n_pos_target = num_gaussians / 2;
    if (positive_only) n_pos_target = num_gaussians;
    int n_neg_target = num_gaussians - n_pos_target;

    std::vector<int> final_pos_indices = run_nms(pos_candidates, n_pos_target);
    std::vector<int> final_neg_indices = run_nms(neg_candidates, n_neg_target);

    int current_g = 0;

    auto write_gaussian = [&](int idx_grid, double weight_val) {
        if (current_g >= num_gaussians) return;

        Coord c = idx_to_coord(idx_grid, grid.nx, grid.ny);
        std::uniform_real_distribution<double> jitter(-0.1, 0.1);

        double px = grid.origin_x + (c.x + 0.5 + jitter(rng)) * grid.voxel_size;
        double py = grid.origin_y + (c.y + 0.5 + jitter(rng)) * grid.voxel_size;
        double pz = grid.origin_z + (c.z + 0.5 + jitter(rng)) * grid.voxel_size;

        int base = current_g * PARAMS_PER_GAUSSIAN;
        params[base + 0] = px;
        params[base + 1] = py;
        params[base + 2] = pz;
        params[base + 3] = init_log_sigma;
        params[base + 4] = init_log_sigma;
        params[base + 5] = init_log_sigma;
        params[base + 6] = weight_val;

        current_g++;
    };

    // 1. Fill Positive from NMS
    for (int idx : final_pos_indices) write_gaussian(idx, w_init);

    // 2. Fill remaining Positive randomly
    std::vector<int> all_pos_indices;
    while (current_g < n_pos_target) {
        if (all_pos_indices.empty()) {
             for (int z = iz_start; z < iz_end; ++z) {
                for (int y = iy_start; y < iy_end; ++y) {
                    for (int x = ix_start; x < ix_end; ++x) {
                        int idx = grid.idx(x, y, z);
                        if (grid.sdf_data[idx] > 0) all_pos_indices.push_back(idx);
                    }
                }
            }
        }
        if (all_pos_indices.empty()) break;

        std::uniform_int_distribution<int> dist(0, all_pos_indices.size() - 1);
        write_gaussian(all_pos_indices[dist(rng)], w_init);
    }

    // 3. Fill Negative from NMS
    for (int idx : final_neg_indices) write_gaussian(idx, -w_init);

    // 4. Fill remaining Negative randomly
    std::vector<int> all_neg_indices;
    while (current_g < num_gaussians) {
        if (all_neg_indices.empty()) {
             for (int z = iz_start; z < iz_end; ++z) {
                for (int y = iy_start; y < iy_end; ++y) {
                    for (int x = ix_start; x < ix_end; ++x) {
                        int idx = grid.idx(x, y, z);
                        if (grid.sdf_data[idx] < 0) all_neg_indices.push_back(idx);
                    }
                }
            }
            if (all_neg_indices.empty()) {
                 for (int z = iz_start; z < iz_end; ++z) {
                    for (int y = iy_start; y < iy_end; ++y) {
                        for (int x = ix_start; x < ix_end; ++x) {
                            int idx = grid.idx(x, y, z);
                            if (grid.sdf_data[idx] < 1e-3) all_neg_indices.push_back(idx);
                        }
                    }
                }
            }
        }
        if (all_neg_indices.empty()) break;

        std::uniform_int_distribution<int> dist(0, all_neg_indices.size() - 1);
        write_gaussian(all_neg_indices[dist(rng)], -w_init);
    }

    return params;
}

/// Trilinear interpolation
inline double getSdfTrilinear(const LocalGrid& grid, double x, double y, double z) {
    double fx = (x - grid.origin_x) / grid.voxel_size - 0.5;
    double fy = (y - grid.origin_y) / grid.voxel_size - 0.5;
    double fz = (z - grid.origin_z) / grid.voxel_size - 0.5;

    int x0 = std::max(0, std::min(int(std::floor(fx)), grid.nx - 2));
    int y0 = std::max(0, std::min(int(std::floor(fy)), grid.ny - 2));
    int z0 = std::max(0, std::min(int(std::floor(fz)), grid.nz - 2));
    
    int z1 = z0 + 1;

    double xd = fx - x0, yd = fy - y0, zd = fz - z0;
    xd = std::max(0.0, std::min(1.0, xd));
    yd = std::max(0.0, std::min(1.0, yd));
    zd = std::max(0.0, std::min(1.0, zd));

    auto get = [&](int ix, int iy, int iz) {
        return grid.sdf_data[iz * grid.nx * grid.ny + iy * grid.nx + ix];
    };

    double c00 = get(x0, y0, z0) * (1 - xd) + get(x0 + 1, y0, z0) * xd;
    double c10 = get(x0, y0 + 1, z0) * (1 - xd) + get(x0 + 1, y0 + 1, z0) * xd;
    double c01 = get(x0, y0, z1) * (1 - xd) + get(x0 + 1, y0, z1) * xd;
    double c11 = get(x0, y0 + 1, z1) * (1 - xd) + get(x0 + 1, y0 + 1, z1) * xd;
    double c0 = c00 * (1 - yd) + c10 * yd;
    double c1 = c01 * (1 - yd) + c11 * yd;
    return c0 * (1 - zd) + c1 * zd;
}

/// Sample points from grid for training (Uniform)
/// Samples from cube + margin area (training region), not the full grid (which includes edt_extension)
inline std::vector<GMMData> samplePointsFromGrid(const LocalGrid& grid, int n, float margin, unsigned int seed = 42) {
    std::vector<GMMData> data;
    data.reserve(n);
    std::mt19937 rng(seed);

    // Define sampling bounds: cube + margin (training region only)
    // Training bounds are cube origin - margin to cube origin + cube_size + margin
    double x0 = grid.cube_origin_x - margin;
    double x1 = grid.cube_origin_x + grid.cube_size + margin;
    double y0 = grid.cube_origin_y - margin;
    double y1 = grid.cube_origin_y + grid.cube_size + margin;
    double z0 = grid.cube_origin_z - margin;
    double z1 = grid.cube_origin_z + grid.cube_size + margin;

    std::uniform_real_distribution<double> dx(x0, x1), dy(y0, y1), dz(z0, z1);
    std::uniform_real_distribution<double> jitter(-0.5, 0.5);

    // Identify surface voxels within training region (SDF < 0.1)
    std::vector<int> surface_indices;
    
    // Calculate voxel bounds for training region
    int vx_start = std::max(0, int((x0 - grid.origin_x) / grid.voxel_size));
    int vx_end = std::min(grid.nx, int((x1 - grid.origin_x) / grid.voxel_size) + 1);
    int vy_start = std::max(0, int((y0 - grid.origin_y) / grid.voxel_size));
    int vy_end = std::min(grid.ny, int((y1 - grid.origin_y) / grid.voxel_size) + 1);
    int vz_start = std::max(0, int((z0 - grid.origin_z) / grid.voxel_size));
    int vz_end = std::min(grid.nz, int((z1 - grid.origin_z) / grid.voxel_size) + 1);
    
    for (int z = vz_start; z < vz_end; ++z) {
        for (int y = vy_start; y < vy_end; ++y) {
            for (int x = vx_start; x < vx_end; ++x) {
                int idx = grid.idx(x, y, z);
                if (idx >= 0 && std::abs(grid.sdf_data[idx]) < 0.1) {
                    surface_indices.push_back(idx);
                }
            }
        }
    }

    int n_surface = (surface_indices.empty()) ? 0 : n / 2;
    int n_uniform = n - n_surface;

    // 1. Sample near surface
    if (n_surface > 0) {
        std::uniform_int_distribution<int> dist(0, surface_indices.size() - 1);
        for (int i = 0; i < n_surface; ++i) {
            int idx = surface_indices[dist(rng)];
            Coord c = idx_to_coord(idx, grid.nx, grid.ny);
            
            // Jitter within voxel
            double px = grid.origin_x + (c.x + 0.5 + jitter(rng)) * grid.voxel_size;
            double py = grid.origin_y + (c.y + 0.5 + jitter(rng)) * grid.voxel_size;
            double pz = grid.origin_z + (c.z + 0.5 + jitter(rng)) * grid.voxel_size;
            
            data.push_back({px, py, pz, getSdfTrilinear(grid, px, py, pz)});
        }
    }

    // 2. Sample uniformly within training bounds
    for (int i = 0; i < n_uniform; ++i) {
        double x = dx(rng), y = dy(rng), z = dz(rng);
        data.push_back({x, y, z, getSdfTrilinear(grid, x, y, z)});
    }
    return data;
}

/// Predict SDF from Gaussian parameters
inline double predictSdf(double x, double y, double z, const std::vector<double>& params, int ng) {
    double pred = 0.0;
    constexpr double eps = 1e-6;

    for (int i = 0; i < ng; ++i) {
        int j = i * PARAMS_PER_GAUSSIAN;
        double vx = x - params[j], vy = y - params[j + 1], vz = z - params[j + 2];
        double l0 = params[j + 3] * params[j + 3] + eps;
        double l1 = params[j + 4] * params[j + 4] + eps;
        double l2 = params[j + 5] * params[j + 5] + eps;
        double dsq = (vx / l0) * (vx / l0) + (vy / l1) * (vy / l1) + (vz / l2) * (vz / l2);
        pred += params[j + 6] * std::exp(-0.5 * dsq);
    }
    return pred;
}

/// Calculate MAE and std deviation
inline void calculateMetrics(const LocalGrid& grid, const std::vector<double>& params, int ng,
                             double& mae, double& std_dev, float margin_val = 0.0f) {
    std::vector<double> errs;
    // Use a small safety margin (2 voxels) to avoid trilinear interpolation issues at grid boundaries
    int margin = 2;

    for (int z = margin; z < grid.nz - margin; ++z)
    for (int y = margin; y < grid.ny - margin; ++y)
    for (int x = margin; x < grid.nx - margin; ++x) {
        double wx = grid.origin_x + x * grid.voxel_size;
        double wy = grid.origin_y + y * grid.voxel_size;
        double wz = grid.origin_z + z * grid.voxel_size;

        if (wx >= grid.cube_origin_x - margin_val && wx < grid.cube_origin_x + grid.cube_size + margin_val &&
            wy >= grid.cube_origin_y - margin_val && wy < grid.cube_origin_y + grid.cube_size + margin_val &&
            wz >= grid.cube_origin_z - margin_val && wz < grid.cube_origin_z + grid.cube_size + margin_val) {

            double actual = grid.sdf_data[grid.idx(x, y, z)];
            double err = std::abs(predictSdf(wx, wy, wz, params, ng) - actual);
            errs.push_back(err);
        }
    }

    if (errs.empty()) { mae = std_dev = 0.0; return; }

    double sum = 0.0;
    for (double e : errs) sum += e;
    mae = sum / errs.size();

    double var = 0.0;
    for (double e : errs) var += (e - mae) * (e - mae);
    std_dev = std::sqrt(var / errs.size());
}

/// Train with specific number of gaussians using smart initialization
inline TrainResult trainWithGaussians(const LocalGrid& grid, std::vector<GMMData>& data,
                                       int num_gaussians, const SolverConfig& solver_cfg, bool positive_only, float margin = 0.0f) {
    TrainResult result;
    result.num_gaussians = num_gaussians;
    result.params = initializeSmartGaussians(grid, num_gaussians, positive_only);

    GMMSolver solver(solver_cfg);
    solver.solve(data, result.params);

    calculateMetrics(grid, result.params, num_gaussians, result.mae, result.std_dev, margin);
    result.valid = true;
    return result;
}

/// Fixed gaussian count training
inline TrainResult trainGaussians(const LocalGrid& grid, int num_gaussians, const TrainerConfig& cfg = {}) {
    auto data = samplePointsFromGrid(grid, cfg.sample_points, cfg.margin);
    if (data.empty()) return TrainResult{};

    TrainResult result = trainWithGaussians(grid, data, num_gaussians, cfg.solver, cfg.positive_only, cfg.margin);

    if (result.mae > cfg.mae_threshold_max) {
        result.valid = false;
    }

    return result;
}

#endif // GAUSSIAN_TRAINER_HPP
