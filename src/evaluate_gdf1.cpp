/**
 * @file evaluate_gdf1.cpp
 * @brief Evaluate GDF1 map accuracy against a ground truth pointcloud.
 *
 * Computes MAE, StdDev, RMSE for both EDF error and gradient magnitude.
 * Reports both global and per-cube (averaged) metrics.
 *
 * Usage:
 *   ./evaluate_gdf1 <map.bin> <gt.pcd|ply> [--step 0.2] [--no-blending]
 *                   [--max-distance 2.0] [--max-cube-mae 0.1]
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <tuple>
#include <numeric>
#include <algorithm>
#include <chrono>

#include <omp.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "pointcloud_loader.hpp"

// ==========================================
// Data Structures (identical to gaussian_to_ply.cpp)
// ==========================================

struct GaussianSoA {
    std::vector<float> x, y, z;
    std::vector<float> inv_l0, inv_l1, inv_l2;
    std::vector<float> w;

    void reserve(size_t n) {
        x.reserve(n); y.reserve(n); z.reserve(n);
        inv_l0.reserve(n); inv_l1.reserve(n); inv_l2.reserve(n);
        w.reserve(n);
    }

    void push_back(double px, double py, double pz, double l0, double l1, double l2, double pw) {
        x.push_back(static_cast<float>(px));
        y.push_back(static_cast<float>(py));
        z.push_back(static_cast<float>(pz));
        inv_l0.push_back(static_cast<float>(1.0 / l0));
        inv_l1.push_back(static_cast<float>(1.0 / l1));
        inv_l2.push_back(static_cast<float>(1.0 / l2));
        w.push_back(static_cast<float>(pw));
    }

    size_t size() const { return x.size(); }
};

struct CubeData {
    double ox, oy, oz;
    double mae = 0.0;
    GaussianSoA gaussians;
};

struct CubeKeyHash {
    std::size_t operator()(const std::tuple<int,int,int>& k) const {
        return std::hash<int>()(std::get<0>(k)) ^
               (std::hash<int>()(std::get<1>(k)) << 1) ^
               (std::hash<int>()(std::get<2>(k)) << 2);
    }
};

using CubeMap = std::unordered_map<std::tuple<int,int,int>, size_t, CubeKeyHash>;

// ==========================================
// Binary Format Structures (Must match Exporter)
// ==========================================

struct __attribute__((packed)) MapHeader {
    char magic[4];
    uint32_t version;
    uint32_t num_cubes;
    float avg_mae;
    float std_dev;
    float bounds_min[3];
    float bounds_max[3];
    float cube_size;
    float empty_search_margin;
    float cube_margin;
    uint8_t padding[64];
};

struct __attribute__((packed)) CubeHeader {
    float origin[3];
    float mae;
    float std_dev;
    uint32_t num_gaussians;
};

struct __attribute__((packed)) GaussianData {
    uint32_t id;
    float mean[3];
    float sigma[3];
    float weight;
};

// ==========================================
// Binary Loader (identical to gaussian_to_ply.cpp)
// ==========================================

struct MapInfo {
    std::vector<CubeData> cubes;
    CubeMap cube_map;
    float cube_size = 1.0f;
    float margin = 0.25f;
    float avg_mae = 0.0f;
    float std_dev = 0.0f;
    float bounds_min[3];
    float bounds_max[3];
    uint32_t version = 0;
};

bool loadBinary(const std::string& filename, MapInfo& info) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open: " << filename << std::endl;
        return false;
    }

    MapHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(MapHeader));

    if (std::strncmp(header.magic, "GDF1", 4) != 0) {
        std::cerr << "[ERROR] Invalid binary format or version mismatch" << std::endl;
        return false;
    }

    info.version = header.version;
    info.cube_size = header.cube_size;
    info.margin = header.cube_margin;
    info.avg_mae = header.avg_mae;
    info.std_dev = header.std_dev;
    std::memcpy(info.bounds_min, header.bounds_min, sizeof(float) * 3);
    std::memcpy(info.bounds_max, header.bounds_max, sizeof(float) * 3);

    info.cubes.reserve(header.num_cubes);

    for (uint32_t i = 0; i < header.num_cubes; ++i) {
        CubeHeader ch;
        file.read(reinterpret_cast<char*>(&ch), sizeof(CubeHeader));

        CubeData cube;
        cube.ox = ch.origin[0];
        cube.oy = ch.origin[1];
        cube.oz = ch.origin[2];
        cube.mae = ch.mae;
        cube.gaussians.reserve(ch.num_gaussians);

        for (uint32_t g = 0; g < ch.num_gaussians; ++g) {
            GaussianData gd;
            file.read(reinterpret_cast<char*>(&gd), sizeof(GaussianData));

            double l0 = std::pow(gd.sigma[0], 4);
            double l1 = std::pow(gd.sigma[1], 4);
            double l2 = std::pow(gd.sigma[2], 4);

            cube.gaussians.push_back(gd.mean[0], gd.mean[1], gd.mean[2], l0, l1, l2, gd.weight);
        }

        // Build cube map key
        int ix = static_cast<int>(std::floor(cube.ox / info.cube_size));
        int iy = static_cast<int>(std::floor(cube.oy / info.cube_size));
        int iz = static_cast<int>(std::floor(cube.oz / info.cube_size));
        info.cube_map[{ix, iy, iz}] = info.cubes.size();

        info.cubes.push_back(std::move(cube));
    }
    return true;
}

// ==========================================
// Predict EDF (scalar std::exp for accuracy)
// ==========================================

double predict(double x, double y, double z, const GaussianSoA& gs) {
    size_t n = gs.x.size();
    const float* __restrict__ px = gs.x.data();
    const float* __restrict__ py = gs.y.data();
    const float* __restrict__ pz = gs.z.data();
    const float* __restrict__ il0 = gs.inv_l0.data();
    const float* __restrict__ il1 = gs.inv_l1.data();
    const float* __restrict__ il2 = gs.inv_l2.data();
    const float* __restrict__ pw = gs.w.data();

    float fx = static_cast<float>(x);
    float fy = static_cast<float>(y);
    float fz = static_cast<float>(z);

    double val = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float dx = fx - px[i];
        float dy = fy - py[i];
        float dz = fz - pz[i];
        float dsq = (dx*dx)*il0[i] + (dy*dy)*il1[i] + (dz*dz)*il2[i];
        val += pw[i] * std::exp(-0.5f * dsq);
    }
    return val;
}

// ==========================================
// Predict Gradient Magnitude (scalar std::exp for accuracy)
// ==========================================

double predictGradient(double x, double y, double z, const GaussianSoA& gs) {
    size_t n = gs.x.size();
    const float* __restrict__ px = gs.x.data();
    const float* __restrict__ py = gs.y.data();
    const float* __restrict__ pz = gs.z.data();
    const float* __restrict__ il0 = gs.inv_l0.data();
    const float* __restrict__ il1 = gs.inv_l1.data();
    const float* __restrict__ il2 = gs.inv_l2.data();
    const float* __restrict__ pw = gs.w.data();

    float fx = static_cast<float>(x);
    float fy = static_cast<float>(y);
    float fz = static_cast<float>(z);

    double gx = 0, gy = 0, gz = 0;
    for (size_t i = 0; i < n; ++i) {
        float dx = fx - px[i];
        float dy = fy - py[i];
        float dz = fz - pz[i];
        float dsq = (dx*dx)*il0[i] + (dy*dy)*il1[i] + (dz*dz)*il2[i];
        float term = -pw[i] * std::exp(-0.5f * dsq);
        gx += term * dx * il0[i];
        gy += term * dy * il1[i];
        gz += term * dz * il2[i];
    }
    return std::sqrt(gx*gx + gy*gy + gz*gz);
}

// ==========================================
// Blending (identical to gaussian_to_ply.cpp)
// ==========================================

double smoothstep(double t) {
    if (t <= 0.0) return 0.0;
    if (t >= 1.0) return 1.0;
    return t * t * (3.0 - 2.0 * t);
}

double blendWeight(double px, double py, double pz,
                   double cx, double cy, double cz,
                   double cube_size, double margin) {
    double dx_min = std::min(px - cx, cx + cube_size - px);
    double dy_min = std::min(py - cy, cy + cube_size - py);
    double dz_min = std::min(pz - cz, cz + cube_size - pz);
    double min_dist = std::min({dx_min, dy_min, dz_min});
    return smoothstep(1.0 + min_dist / margin);
}

// Generic blending function: works for both predict and predictGradient
using EvalFn = double(*)(double, double, double, const GaussianSoA&);

struct BlendResult {
    double value;
    bool valid;
};

BlendResult evalWithBlending(double x, double y, double z,
                             const MapInfo& info, EvalFn fn) {
    double cs = info.cube_size;
    double margin = info.margin;

    int ix = static_cast<int>(std::floor(x / cs));
    int iy = static_cast<int>(std::floor(y / cs));
    int iz = static_cast<int>(std::floor(z / cs));

    auto it = info.cube_map.find({ix, iy, iz});
    if (it != info.cube_map.end()) {
        const CubeData& c = info.cubes[it->second];
        double dx_min = x - c.ox;
        double dx_max = c.ox + cs - x;
        double dy_min = y - c.oy;
        double dy_max = c.oy + cs - y;
        double dz_min = z - c.oz;
        double dz_max = c.oz + cs - z;

        bool near_edge = (dx_min < margin || dx_max < margin ||
                          dy_min < margin || dy_max < margin ||
                          dz_min < margin || dz_max < margin);

        if (!near_edge) {
            return {fn(x, y, z, c.gaussians), true};
        }
    }

    // Slow path: blend 3x3x3 neighbors
    double weighted_sum = 0.0;
    double weight_total = 0.0;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                auto nit = info.cube_map.find({ix+dx, iy+dy, iz+dz});
                if (nit == info.cube_map.end()) continue;

                const CubeData& nb = info.cubes[nit->second];
                if (x < nb.ox - margin || x > nb.ox + cs + margin ||
                    y < nb.oy - margin || y > nb.oy + cs + margin ||
                    z < nb.oz - margin || z > nb.oz + cs + margin)
                    continue;

                double w = blendWeight(x, y, z, nb.ox, nb.oy, nb.oz, cs, margin);
                if (w > 1e-6) {
                    weighted_sum += w * fn(x, y, z, nb.gaussians);
                    weight_total += w;
                }
            }
        }
    }

    if (weight_total < 1e-6) return {0.0, false};
    return {weighted_sum / weight_total, true};
}

BlendResult evalNoBlending(double x, double y, double z,
                           const MapInfo& info, EvalFn fn) {
    double cs = info.cube_size;
    int ix = static_cast<int>(std::floor(x / cs));
    int iy = static_cast<int>(std::floor(y / cs));
    int iz = static_cast<int>(std::floor(z / cs));

    auto it = info.cube_map.find({ix, iy, iz});
    if (it == info.cube_map.end()) return {0.0, false};

    const CubeData& c = info.cubes[it->second];
    if (x < c.ox || x >= c.ox + cs ||
        y < c.oy || y >= c.oy + cs ||
        z < c.oz || z >= c.oz + cs)
        return {0.0, false};

    return {fn(x, y, z, c.gaussians), true};
}

// ==========================================
// Per-cube metrics accumulator
// ==========================================

struct CubeMetrics {
    double sum_error = 0.0;
    double sum_error_sq = 0.0;
    double sum_grad = 0.0;
    double sum_grad_error_sq = 0.0;
    int count = 0;
};

// ==========================================
// Main
// ==========================================

int main(int argc, char** argv) {
    // ---- Parse arguments ----
    std::string bin_file, gt_file;
    double step = 0.2;
    bool blending = true;
    double max_cube_mae = -1.0;
    double min_sdf = 0.1; // Default to 0.1m as in plot_slice.py

    if (argc < 3) {
        std::cerr << "Usage: evaluate_gdf1 <map.bin> <gt.pcd|ply> [options]\n"
                  << "Options:\n"
                  << "  --step <float>          Sampling step in meters (default: 0.2)\n"
                  << "  --no-blending           Disable blending\n"
                  << "  --max-cube-mae <float>  Exclude cubes with MAE > this\n"
                  << "  --min-sdf <float>       Exclude points with SDF < this from gradient metrics (default: 0.1)\n"
                  << "  --outlier-percent <val> Exclude top X% worst points from global stats (e.g. 1.0)\n";
        return 1;
    }

    bin_file = argv[1];
    gt_file = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--step" && i + 1 < argc) step = std::stod(argv[++i]);
        else if (arg == "--no-blending") blending = false;
        else if (arg == "--max-cube-mae" && i + 1 < argc) max_cube_mae = std::stod(argv[++i]);
        else if (arg == "--min-sdf" && i + 1 < argc) min_sdf = std::stod(argv[++i]);
    }

    // ---- Load map ----
    MapInfo info;
    if (!loadBinary(bin_file, info)) return 1;

    // Print header
    std::string bn = bin_file.substr(bin_file.find_last_of("/\\") + 1);
    std::cout << "\n" << std::string(50, '=') << "\n"
              << " GDF1 MAP: " << bn << "\n"
              << std::string(50, '=') << "\n"
              << " Version:    " << info.version << "\n"
              << " Cubes:      " << info.cubes.size() << "\n"
              << " Avg MAE:    " << std::fixed << std::setprecision(4) << info.avg_mae << " m\n"
              << " Std Dev:    " << info.std_dev << "\n"
              << " Cube Size:  " << info.cube_size << " m\n"
              << " Margin:     " << info.margin << " m\n"
              << " Bounds:     (" << info.bounds_min[0] << ", " << info.bounds_min[1] << ", " << info.bounds_min[2]
              << ") -> (" << info.bounds_max[0] << ", " << info.bounds_max[1] << ", " << info.bounds_max[2] << ")\n"
              << std::string(50, '=') << "\n";

    // ---- Load GT pointcloud ----
    auto gt_cloud = loadPointCloud(gt_file);
    if (!gt_cloud || gt_cloud->empty()) {
        std::cerr << "[ERROR] Failed to load GT pointcloud" << std::endl;
        return 1;
    }

    // ---- Build KD-tree ----
    auto t_start = std::chrono::high_resolution_clock::now();
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(gt_cloud);

    // ---- Generate sample points inside existing cubes ----
    struct SamplePoint {
        float x, y, z;
        size_t cube_idx;
    };

    std::vector<SamplePoint> samples;
    double offset = step * 0.5;
    size_t skipped_cubes = 0;

    for (size_t ci = 0; ci < info.cubes.size(); ++ci) {
        const CubeData& cube = info.cubes[ci];
        if (max_cube_mae > 0 && cube.mae > max_cube_mae) {
            skipped_cubes++;
            continue;
        }

        double cs = info.cube_size;
        for (double sx = cube.ox + offset; sx < cube.ox + cs; sx += step) {
            for (double sy = cube.oy + offset; sy < cube.oy + cs; sy += step) {
                for (double sz = cube.oz + offset; sz < cube.oz + cs; sz += step) {
                    samples.push_back({static_cast<float>(sx), static_cast<float>(sy),
                                       static_cast<float>(sz), ci});
                }
            }
        }
    }

    size_t total = samples.size();
    std::cout << "\n[INFO] Sample points: " << total
              << " (" << step << "m step, " << (blending ? "blending" : "no blending") << ")\n";

    // ---- Compute GT distances via KD-tree ----
    std::vector<float> gt_distances(total);
    {
        std::vector<int> idx(1);
        std::vector<float> dist_sq(1);
        #pragma omp parallel for private(idx, dist_sq) schedule(dynamic, 1024)
        for (size_t i = 0; i < total; ++i) {
            pcl::PointXYZ pt;
            pt.x = samples[i].x;
            pt.y = samples[i].y;
            pt.z = samples[i].z;
            idx.resize(1); dist_sq.resize(1);
            kdtree.nearestKSearch(pt, 1, idx, dist_sq);
            gt_distances[i] = std::sqrt(dist_sq[0]);
        }
    }


    // ---- Evaluate predicted EDF and gradient (parallel) ----
    std::vector<double> errors(total);
    std::vector<double> grad_norms(total);
    std::vector<bool> valid(total, false);

    int progress_step = std::max(1, (int)(total / 20));

    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < total; ++i) {
        double x = samples[i].x;
        double y = samples[i].y;
        double z = samples[i].z;

        BlendResult edf_result, grad_result;
        if (blending) {
            edf_result = evalWithBlending(x, y, z, info, predict);
            grad_result = evalWithBlending(x, y, z, info, predictGradient);
        } else {
            edf_result = evalNoBlending(x, y, z, info, predict);
            grad_result = evalNoBlending(x, y, z, info, predictGradient);
        }

        if (edf_result.valid) {
            errors[i] = std::abs(edf_result.value - gt_distances[i]);
            // Only include gradient if valid AND SDF > min_sdf (to avoid instabilities near surface)
            if (grad_result.valid && edf_result.value > min_sdf) {
                grad_norms[i] = grad_result.value;
            } else {
                grad_norms[i] = std::nan("");
            }
            valid[i] = true;
        }

        if ((int)(i + 1) % progress_step == 0) {
            #pragma omp critical
            {
                int pct = static_cast<int>((i + 1) * 100 / total);
                std::cout << "\r Evaluating: " << pct << "% (" << (i+1) << "/" << total << ")" << std::flush;
            }
        }
    }
    std::cout << "\r Evaluating: 100% (" << total << "/" << total << ")" << std::endl;

    // ---- Collect valid results ----
    std::vector<double> valid_errors;
    std::vector<double> valid_grads;
    std::vector<size_t> valid_cube_ids;

    valid_errors.reserve(total);
    valid_grads.reserve(total);
    valid_cube_ids.reserve(total);

    for (size_t i = 0; i < total; ++i) {
        if (valid[i]) {
            valid_errors.push_back(errors[i]);
            valid_grads.push_back(grad_norms[i]);
            valid_cube_ids.push_back(samples[i].cube_idx);
        }
    }

    if (valid_errors.empty()) {
        std::cerr << "\n[ERROR] No valid evaluation points!" << std::endl;
        return 1;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // ---- Compute global metrics (Median & Outlier Filtering) ----
    size_t N = valid_errors.size(); // Re-declare N
    
    // Count valid gradients first for reservation
    size_t grad_count = 0;
    for(double g : valid_grads) {
        if(!std::isnan(g)) grad_count++;
    }

    // Sort to find median and percentiles
    std::sort(valid_errors.begin(), valid_errors.end());
    
    std::vector<double> valid_grads_norms;
    valid_grads_norms.reserve(grad_count);
    for(double g : valid_grads) {
        if(!std::isnan(g)) valid_grads_norms.push_back(g);
    }
    std::sort(valid_grads_norms.begin(), valid_grads_norms.end());

    double median_edf = 0.0;
    if (!valid_errors.empty()) {
        if (valid_errors.size() % 2 == 0) {
            median_edf = (valid_errors[valid_errors.size()/2 - 1] + valid_errors[valid_errors.size()/2]) / 2.0;
        } else {
            median_edf = valid_errors[valid_errors.size()/2];
        }
    }

    double median_grad = 0.0;
    if (!valid_grads_norms.empty()) {
        if (valid_grads_norms.size() % 2 == 0) {
            median_grad = (valid_grads_norms[valid_grads_norms.size()/2 - 1] + valid_grads_norms[valid_grads_norms.size()/2]) / 2.0;
        } else {
            median_grad = valid_grads_norms[valid_grads_norms.size()/2];
        }
    }
    
    // Outlier filtering (if outlier_percent > 0)
    double outlier_percent = 0.0; // Default 0
    
    // --- Re-parse args for outlier percent ---
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--outlier-percent" && i + 1 < argc) outlier_percent = std::stod(argv[++i]);
    }

    size_t n_filtered_edf = valid_errors.size();
    size_t n_filtered_grad = valid_grads_norms.size();

    if (outlier_percent > 0.0) {
        size_t cut_edf = static_cast<size_t>(valid_errors.size() * (1.0 - outlier_percent / 100.0));
        size_t cut_grad = static_cast<size_t>(valid_grads_norms.size() * (1.0 - outlier_percent / 100.0));
        n_filtered_edf = std::max((size_t)1, cut_edf);
        n_filtered_grad = std::max((size_t)1, cut_grad);
    }

    // Compute metrics on potentially filtered data
    double sum_err = 0, sum_err_sq = 0;
    for (size_t i = 0; i < n_filtered_edf; ++i) {
        sum_err += valid_errors[i];
        sum_err_sq += valid_errors[i] * valid_errors[i];
    }

    double sum_grad = 0, sum_grad_err_sq = 0, sum_grad_dev_sq = 0;
    for (size_t i = 0; i < n_filtered_grad; ++i) {
        sum_grad += valid_grads_norms[i];
        double ge = valid_grads_norms[i] - 1.0;
        sum_grad_err_sq += ge * ge;
    }

    double mae_global = sum_err / n_filtered_edf;
    double rmse_global = std::sqrt(sum_err_sq / n_filtered_edf);

    // Std Dev = sqrt(E[x²] - E[x]²)
    double var_err = (sum_err_sq / n_filtered_edf) - (mae_global * mae_global);
    double std_global = std::sqrt(std::max(0.0, var_err));

    double grad_mean = (n_filtered_grad > 0) ? sum_grad / n_filtered_grad : 0;
    double grad_rmse = (n_filtered_grad > 0) ? std::sqrt(sum_grad_err_sq / n_filtered_grad) : 0;
    
    // Recalc grad std dev with new mean
    for (size_t i = 0; i < n_filtered_grad; ++i) {
        double d = valid_grads_norms[i] - grad_mean;
        sum_grad_dev_sq += d * d;
    }
    double grad_std = (n_filtered_grad > 0) ? std::sqrt(sum_grad_dev_sq / n_filtered_grad) : 0;


    // ---- Compute per-cube metrics (Use ALL data for per-cube or filtered? Usually keep per-cube raw) ----
    std::unordered_map<size_t, CubeMetrics> cube_metrics;
    for (size_t i = 0; i < N; ++i) {
        auto& cm = cube_metrics[valid_cube_ids[i]];
        cm.sum_error += valid_errors[i];
        cm.sum_error_sq += valid_errors[i] * valid_errors[i];
        if (!std::isnan(valid_grads[i])) {
            cm.sum_grad += valid_grads[i];
            double ge = valid_grads[i] - 1.0;
            cm.sum_grad_error_sq += ge * ge;
        }
        cm.count++;
    }

    double pc_mae_sum = 0, pc_std_sum = 0, pc_rmse_sum = 0;
    double pc_grad_mean_sum = 0, pc_grad_std_sum = 0, pc_grad_rmse_sum = 0;
    size_t num_cubes_eval = cube_metrics.size();
    size_t num_cubes_grad = 0;

    for (auto& [cid, cm] : cube_metrics) {
        double cube_mae = cm.sum_error / cm.count;
        double cube_rmse = std::sqrt(cm.sum_error_sq / cm.count);
        double cube_var = (cm.sum_error_sq / cm.count) - (cube_mae * cube_mae);
        double cube_std = std::sqrt(std::max(0.0, cube_var));

        pc_mae_sum += cube_mae;
        pc_std_sum += cube_std;
        pc_rmse_sum += cube_rmse;
    }

    // Per-cube gradient: need per-cube mean and std of gradient norms
    // Group by cube
    std::unordered_map<size_t, std::vector<double>> cube_grad_vals;
    for (size_t i = 0; i < N; ++i) {
        if (!std::isnan(valid_grads[i])) {
            cube_grad_vals[valid_cube_ids[i]].push_back(valid_grads[i]);
        }
    }

    for (auto& [cid, gvals] : cube_grad_vals) {
        double mean = 0;
        for (double v : gvals) mean += v;
        mean /= gvals.size();

        double var = 0;
        double rmse_sq = 0;
        for (double v : gvals) {
            var += (v - mean) * (v - mean);
            rmse_sq += (v - 1.0) * (v - 1.0);
        }

        pc_grad_mean_sum += mean;
        pc_grad_std_sum += std::sqrt(var / gvals.size());
        pc_grad_rmse_sum += std::sqrt(rmse_sq / gvals.size());
        num_cubes_grad++;
    }

    double pc_mae = pc_mae_sum / num_cubes_eval;
    double pc_std = pc_std_sum / num_cubes_eval;
    double pc_rmse = pc_rmse_sum / num_cubes_eval;
    double pc_grad_mean = (num_cubes_grad > 0) ? pc_grad_mean_sum / num_cubes_grad : 0;
    double pc_grad_std = (num_cubes_grad > 0) ? pc_grad_std_sum / num_cubes_grad : 0;
    double pc_grad_rmse = (num_cubes_grad > 0) ? pc_grad_rmse_sum / num_cubes_grad : 0;

    // ---- Print results ----
    auto line = [](){ std::cout << std::string(50, '-') << "\n"; };
    auto dline = [](){ std::cout << std::string(50, '=') << "\n"; };

    std::cout << "\n";
    dline();
    std::cout << std::setw(35) << "EVALUATION RESULTS" << std::string(15, ' ') << "\n";
    dline();
    std::cout << " Points:  " << N << " (" << step << "m step, "
              << (blending ? "blending" : "no blending") << ")\n";
    if (outlier_percent > 0.0) {
        std::cout << " Filter:  Removed top " << outlier_percent << "% of errors (Robust Stats)\n";
    }
    line();
    std::cout << "              GLOBAL — EDF ERROR               \n";
    line();
    std::cout << std::fixed << std::setprecision(6);
    std::cout << " MAE:     " << mae_global << " m\n";
    std::cout << " Median:  " << median_edf << " m\n";
    std::cout << " Std Dev: " << std_global << "\n";
    std::cout << " RMSE:    " << rmse_global << " m\n";
    line();
    std::cout << "            GLOBAL — GRADIENT |grad|\n";
    line();
    std::cout << " Mean:    " << grad_mean << "\n";
    std::cout << " Median:  " << median_grad << "\n";
    std::cout << " Std Dev: " << grad_std << "\n";
    std::cout << " RMSE:    " << grad_rmse << "\n";
    line();
    std::cout << "          PER-CUBE (avg of avgs) — EDF ERROR\n";
    line();
    std::cout << " MAE:     " << pc_mae << " m\n";
    std::cout << " Std Dev: " << pc_std << "\n";
    std::cout << " RMSE:    " << pc_rmse << " m\n";
    if (num_cubes_grad > 0) {
        line();
        std::cout << "        PER-CUBE (avg of avgs) — GRADIENT |grad|\n";
        line();
        std::cout << " Mean:    " << pc_grad_mean << "\n";
        std::cout << " Std Dev: " << pc_grad_std << "\n";
        std::cout << " RMSE:    " << pc_grad_rmse << "\n";
    }
    line();
    std::cout << " Time:    " << std::fixed << std::setprecision(1) << elapsed << "s\n";
    dline();
    std::cout << std::endl;

    return 0;
}
