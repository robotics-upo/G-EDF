/**
 * @file main.cpp
 * @brief Gaussian SDF trainer from pointcloud files (adaptive + dual export)
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <omp.h>
#include <cstring>

#include "pointcloud_loader.hpp"
#include "cube_manager.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include "edt_generator.hpp"
#include "gaussian_trainer.hpp"
#include "csv_exporter.hpp"

// ANSI colors
const char* RST = "\033[0m";
const char* GRN = "\033[32m";
const char* YEL = "\033[33m";
const char* RED = "\033[31m";
const char* CYN = "\033[36m";
const char* BLD = "\033[1m";

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <input.ply|pcd> <output_base> [threads] [sample_points] [mode]\n"
              << "  mode: 'pure' (default) or 'signed'\n"
              << "  Outputs: <output_base>.csv and <output_base>.gsdf\n";
}

int main(int argc, char** argv) {
    if (argc < 3) { printUsage(argv[0]); return 1; }

    std::string input = argv[1];
    std::string output_base = argv[2];
    int threads = (argc >= 4) ? std::atoi(argv[3]) : omp_get_max_threads();
    int sample_points = (argc >= 5) ? std::atoi(argv[4]) : 1000;
    std::string mode_str = (argc >= 6) ? argv[5] : "pure";
    
    omp_set_num_threads(threads);

    EDTMode edt_mode = (mode_str == "signed") ? EDTMode::SIGNED : EDTMode::PURE;

    std::string csv_output = output_base + ".csv";


    std::cout << CYN << BLD << "\n=== GAUSSIAN TRAINER ===" << RST << "\n"
              << " Input:   " << input << "\n"
              << " CSV:     " << csv_output << "\n"
              << " Threads: " << threads << "\n"
              << " Samples: " << sample_points << "\n"
              << " Mode:    " << (edt_mode == EDTMode::PURE ? "PURE (Unsigned)" : "SIGNED (Felzenszwalb)") << "\n"
              << CYN << "------------------------------------" << RST << "\n";

    // Load pointcloud
    auto t0 = std::chrono::high_resolution_clock::now();
    auto cloud = loadPointCloud(input);
    if (!cloud || cloud->empty()) {
        std::cerr << RED << "[ERROR] Failed to load pointcloud" << RST << "\n";
        return 1;
    }

    // Compute cube metadata (streaming - no index storage)
    CubeManager mgr(1.0f);
    mgr.computeCubeMetadata(cloud);
    size_t total = mgr.getCubeCount();
    if (total == 0) { std::cerr << RED << "[ERROR] No cubes" << RST << "\n"; return 1; }

    // Setup exporters
    CSVExporter csv_exporter(csv_output);
    if (!csv_exporter.isOpen()) {
        std::cerr << RED << "[ERROR] Cannot open output files" << RST << "\n";
        return 1;
    }

    // Configure trainer
    // --- Optimization: Downsample for Fast Skipping (Manual) ---
    std::cout << "[INFO] Creating coarse KdTree for fast skipping..." << std::flush;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_coarse(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Simple downsampling: take 1 point every ~0.5m (approx)
    // Or just take every Nth point. Let's take every 10th point for now.
    int step = 10; 
    if (cloud->size() > 1000000) step = 50; // Aggressive downsampling for large clouds
    
    for (size_t i = 0; i < cloud->size(); i += step) {
        cloud_coarse->push_back(cloud->points[i]);
    }

    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_coarse(new pcl::KdTreeFLANN<pcl::PointXYZ>());
    if (cloud_coarse->empty()) {
        std::cout << "[WARNING] Coarse cloud empty, using original." << std::endl;
        kdtree_coarse->setInputCloud(cloud);
    } else {
        kdtree_coarse->setInputCloud(cloud_coarse);
    }
    std::cout << " done. (Reduced " << cloud->size() << " -> " << (cloud_coarse->empty() ? cloud->size() : cloud_coarse->size()) << " points)" << std::endl;

    TrainerConfig cfg;
    cfg.sample_points = sample_points;
    cfg.edt_mode = edt_mode;
    cfg.mae_threshold_good = 0.03;
    cfg.mae_threshold_max = 0.30;
    cfg.solver.max_iterations = 250; // Increased from 150
    cfg.solver.max_time_seconds = 2.0; // Increased time budget (s)
    cfg.solver.function_tolerance = 1e-4;
    cfg.solver.gradient_tolerance = 1e-4;
    cfg.solver.parameter_tolerance = 1e-4;

    std::cout << CYN << "\n=== TRAINING " << total << " CUBES (Adaptive) ===" << RST << "\n";
    std::cout << "[INFO] Starting parallel training..." << std::endl;

    std::atomic<size_t> done(0), valid(0), discarded(0);
    std::atomic<double> sum_mae(0.0);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Get cube infos for parallel processing
    const auto& cube_infos = mgr.getCubeInfos();

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < total; ++i) {
        auto tc = std::chrono::high_resolution_clock::now();
        TrainerConfig local_cfg = cfg; // Thread-local copy

        // Build Cube from CubeInfo
        const CubeInfo& info = cube_infos[i];
        Cube cube;
        cube.ix = info.ix;
        cube.iy = info.iy;
        cube.iz = info.iz;
        cube.origin_x = info.origin_x;
        cube.origin_y = info.origin_y;
        cube.origin_z = info.origin_z;

        // Extract point indices using KdTree
        pcl::PointXYZ center;
        center.x = info.centerX();
        center.y = info.centerY();
        center.z = info.centerZ();

        float radius = std::sqrt(3.0f) * mgr.getCubeSize() * 1.0f; // Increased from 0.6 to 1.0 to ensure coverage
        std::vector<int> indices;
        std::vector<float> dists;
        mgr.getKdTree().radiusSearch(center, radius, indices, dists);

        // Filter to cube bounds
        for (int idx : indices) {
            const auto& pt = cloud->points[idx];
            int cix = int((pt.x - mgr.getOriginX()) / mgr.getCubeSize());
            int ciy = int((pt.y - mgr.getOriginY()) / mgr.getCubeSize());
            int ciz = int((pt.z - mgr.getOriginZ()) / mgr.getCubeSize());
            if (cix == info.ix && ciy == info.iy && ciz == info.iz) {
                cube.point_indices.push_back(idx);
            }
        }

        // --- Adaptive Training Logic ---
        struct AdaptiveConfig {
            std::vector<int> populated_steps = {8, 16, 32};
            std::vector<int> empty_steps = {2, 4, 8, 16};
            double mae_threshold = 0.03;
            double empty_distance_threshold = 3.0;
        } adaptive_cfg;

        TrainResult result;
        bool trained = false;

        // CASE 1: Populated Cube
        if (!cube.point_indices.empty()) {
             // --- Simple Linear Densification ---
            // Create a local copy of points for the cube
            pcl::PointCloud<pcl::PointXYZ>::Ptr cube_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (int idx : cube.point_indices) {
                cube_cloud->push_back(cloud->points[idx]);
            }

            // --- Context Search (1.5x) ---
            // Find points in neighboring cubes to improve EDT boundary accuracy
            std::vector<int> context_indices;
            std::vector<int> search_indices;
            std::vector<float> search_dists;
            pcl::PointXYZ center;
            center.x = cube.centerX();
            center.y = cube.centerY();
            center.z = cube.centerZ();
            float search_radius = std::sqrt(3.0f) * mgr.getCubeSize() * 1.5f;

            if (mgr.getKdTree().radiusSearch(center, search_radius, search_indices, search_dists) > 0) {
                for (int idx : search_indices) {
                    const auto& pt = cloud->points[idx];
                    if (pt.x >= cube.origin_x && pt.x < cube.origin_x + mgr.getCubeSize() &&
                        pt.y >= cube.origin_y && pt.y < cube.origin_y + mgr.getCubeSize() &&
                        pt.z >= cube.origin_z && pt.z < cube.origin_z + mgr.getCubeSize()) {
                        // Inside cube, already in cube.point_indices
                        continue;
                    }
                    context_indices.push_back(idx);
                }
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr densified_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            *densified_cloud = *cube_cloud; 

            if (cube_cloud->size() > 1) {
                pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_local;
                kdtree_local.setInputCloud(cube_cloud);

                float max_dist = 0.0f; // 7cm threshold
                int points_to_add = 0;  // Add 2 points between neighbors

                std::vector<int> pointIdxRadiusSearch;
                std::vector<float> pointRadiusSquaredDistance;

                for (size_t i = 0; i < cube_cloud->size(); ++i) {
                    const auto& pt_a = cube_cloud->points[i];
                    
                    if (kdtree_local.radiusSearch(pt_a, max_dist, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
                        for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
                            int neighbor_idx = pointIdxRadiusSearch[j];
                            
                            // Avoid duplicates: only process if neighbor_idx > i
                            if (neighbor_idx > (int)i) {
                                const auto& pt_b = cube_cloud->points[neighbor_idx];
                                
                                // Linear interpolation
                                for (int k = 1; k <= points_to_add; ++k) {
                                    float ratio = (float)k / (points_to_add + 1);
                                    pcl::PointXYZ new_pt;
                                    new_pt.x = pt_a.x + ratio * (pt_b.x - pt_a.x);
                                    new_pt.y = pt_a.y + ratio * (pt_b.y - pt_a.y);
                                    new_pt.z = pt_a.z + ratio * (pt_b.z - pt_a.z);
                                    densified_cloud->push_back(new_pt);
                                }
                            }
                        }
                    }
                }
            }

            // Add context points to densified_cloud for EDT generation
            for (int idx : context_indices) {
                densified_cloud->push_back(cloud->points[idx]);
            }

            // Build LOCAL KdTree for EDT generation
            pcl::KdTreeFLANN<pcl::PointXYZ> local_kdtree_edt;
            local_kdtree_edt.setInputCloud(densified_cloud);

            float margin = (edt_mode == EDTMode::SIGNED) ? 0.3f : 0.0f;
            LocalGrid grid = generateEDT(cube, densified_cloud, local_kdtree_edt, 0.025f, margin, edt_mode);
            
            // Adaptive Loop for Populated
            for (int ng : adaptive_cfg.populated_steps) {
                result = trainGaussians(grid, ng, local_cfg);
                if (result.valid && result.mae <= adaptive_cfg.mae_threshold) {
                    trained = true;
                    break;
                }
            }
            // If still not good enough, keep the last result (or the best one? For now, last result)
            if (result.valid) trained = true;

        } 
        // CASE 2: Empty Cube
        else {

            // Check distance to nearest point in GLOBAL cloud using COARSE KdTree
            std::vector<int> indices(1);
            std::vector<float> sq_dists(1);
            pcl::PointXYZ searchPoint;
            searchPoint.x = center.x;
            searchPoint.y = center.y;
            searchPoint.z = center.z;

            if (kdtree_coarse->nearestKSearch(searchPoint, 1, indices, sq_dists) > 0) {
                float dist = std::sqrt(sq_dists[0]);
                if (dist < adaptive_cfg.empty_distance_threshold) {
                    // --- Optimization: Local KdTree for EDT ---
                    // 1. Calculate EXACT distance to nearest point to optimize search radius
                    std::vector<int> nn_indices(1);
                    std::vector<float> nn_dists(1);
                    float exact_dist = 0.0f;
                    if (mgr.getKdTree().nearestKSearch(searchPoint, 1, nn_indices, nn_dists) > 0) {
                        exact_dist = std::sqrt(nn_dists[0]);
                    }

                    // 2. Gather points within dynamic radius (exact_dist + 2.0m margin)
                    float search_radius = exact_dist + 0.5f;
                    std::vector<int> local_indices;
                    std::vector<float> local_dists;
                    
                    pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::KdTreeFLANN<pcl::PointXYZ> local_kdtree_edt;
                    
                    auto t_search_start = std::chrono::high_resolution_clock::now();
                    int found = mgr.getKdTree().radiusSearch(searchPoint, search_radius, local_indices, local_dists);
                    auto t_search_end = std::chrono::high_resolution_clock::now();

                    if (found > 0) {
                        auto t_build_start = std::chrono::high_resolution_clock::now();
                        for (int idx : local_indices) {
                            local_cloud->push_back(cloud->points[idx]);
                        }
                        local_kdtree_edt.setInputCloud(local_cloud);
                        auto t_build_end = std::chrono::high_resolution_clock::now();
                        
                        // Generate EDT using LOCAL KdTree and PURE mode
                        auto t_edt_start = std::chrono::high_resolution_clock::now();
                        LocalGrid grid = generateEDT(cube, local_cloud, local_kdtree_edt, 0.025f, 0.0f, EDTMode::PURE);
                        auto t_edt_end = std::chrono::high_resolution_clock::now();
                        
                        // Adaptive Loop for Empty
                        local_cfg.positive_only = true;
                        local_cfg.solver.use_importance_weighting = false; // Disable importance weighting for empty cubes
                        
                        auto t_train_start = std::chrono::high_resolution_clock::now();
                        for (int ng : adaptive_cfg.empty_steps) {
                            result = trainGaussians(grid, ng, local_cfg);
                            if (result.valid && result.mae <= adaptive_cfg.mae_threshold) {
                                trained = true;
                                break;
                            }
                        }
                        if (result.valid) trained = true;
                        auto t_train_end = std::chrono::high_resolution_clock::now();

                        long ms_search = std::chrono::duration_cast<std::chrono::milliseconds>(t_search_end - t_search_start).count();
                        long ms_build = std::chrono::duration_cast<std::chrono::milliseconds>(t_build_end - t_build_start).count();
                        long ms_edt = std::chrono::duration_cast<std::chrono::milliseconds>(t_edt_end - t_edt_start).count();
                        long ms_train = std::chrono::duration_cast<std::chrono::milliseconds>(t_train_end - t_train_start).count();
                        long ms_total = ms_search + ms_build + ms_edt + ms_train;

                        if (ms_total > 1000) {
                             #pragma omp critical
                             {
                                std::cout << YEL << "[SLOW] Cube (" << cube.origin_x << "," << cube.origin_y << "," << cube.origin_z << ") "
                                          << "Total: " << ms_total << "ms "
                                          << "[Search: " << ms_search << "ms, "
                                          << "Build: " << ms_build << "ms (" << local_indices.size() << " pts), "
                                          << "EDT: " << ms_edt << "ms, "
                                          << "Train: " << ms_train << "ms]" << RST << std::endl;
                             }
                        }
                    } else {
                        // Should not happen if distance check passed, but just in case
                        // If no points found in large radius, skip
                    }
                }
            }
        }

        if (!trained) {
            ++done;
            continue;
        }

        auto te = std::chrono::high_resolution_clock::now();
        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(te - tc).count();

        size_t n = ++done;
        if (result.valid) {
            valid++;
            double old = sum_mae.load();
            while (!sum_mae.compare_exchange_weak(old, old + result.mae));

            // Write to both formats
            csv_exporter.writeCube(cube, result, result.num_gaussians);
        } else {
            discarded++;
        }

        if (trained) {
            size_t current_trained_count = valid + discarded;
            if (current_trained_count <= 10 || current_trained_count % 1 == 0 || n == total) {
                #pragma omp critical
                {
                    const char* col = (result.mae > 0.05 ? RED : (result.mae > 0.02 ? YEL : GRN));
                    std::cout << "[" << std::setw(6) << n << "/" << total << "] "
                              << "(" << std::setw(4) << (int)cube.origin_x << ","
                              << std::setw(4) << (int)cube.origin_y << ","
                              << std::setw(4) << (int)cube.origin_z << ") "
                              << col << "G:" << result.num_gaussians
                              << " MAE:" << std::fixed << std::setprecision(4) << result.mae
                              << "±" << result.std_dev << RST << " " << ms << "ms" << "\n";
                }
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    csv_exporter.flush();

    long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count();
    long train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    double avg_mae = valid > 0 ? sum_mae.load() / valid : 0.0;

    std::cout << CYN << "\n=== SUMMARY ===" << RST << "\n"
              << " CSV:       " << csv_output << "\n"
              << " CSV:       " << csv_output << "\n"
              << " Valid:     " << valid << "/" << total << "\n"
              << " Discarded: " << discarded << "\n"
              << " Avg MAE:   " << std::fixed << std::setprecision(4) << avg_mae << " m\n"
              << " Time:      " << std::setprecision(2) << (total_ms / 1000.0) << " s\n"
              << " Speed:     " << std::setprecision(1) << (total * 1000.0 / train_ms) << " cubes/s\n"
              << CYN << "===============" << RST << "\n";

    return 0;
}
