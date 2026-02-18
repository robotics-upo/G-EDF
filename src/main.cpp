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
#include "binary_exporter.hpp"
#include "ElapsedTime.hpp"
#include "config_loader.hpp"

// ANSI colors
const char* RST = "\033[0m";
const char* GRN = "\033[32m";
const char* YEL = "\033[33m";
const char* RED = "\033[31m";
const char* CYN = "\033[36m";
const char* BLD = "\033[1m";

void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog << " <config.yaml>\n"
              << "  All parameters must be set in config.yaml\n";
}

int main(int argc, char** argv) {
    if (argc != 2) { printUsage(argv[0]); return 1; }

    std::string config_file = argv[1];

    // Load configuration
    AppConfig app_cfg;
    if (!ConfigLoader::load(config_file, app_cfg)) {
        std::cerr << RED << "[ERROR] Failed to load config: " << config_file << RST << std::endl;
        return 1;
    }
    std::cout << GRN << "[INFO] Loaded config: " << config_file << RST << std::endl;

    // Validate required parameters
    if (app_cfg.input_file.empty() || app_cfg.output_base.empty()) {
        std::cerr << RED << "[ERROR] input_file and output_base must be set in config.yaml" << RST << std::endl;
        return 1;
    }

    std::string input = app_cfg.input_file;
    std::string output_base = app_cfg.output_base;

    int threads = (app_cfg.num_threads == 0) ? omp_get_max_threads() : app_cfg.num_threads;
    omp_set_num_threads(threads);

    std::string csv_output = app_cfg.output_base + ".csv";
    std::string bin_output = app_cfg.output_base + ".bin";

    std::cout << CYN << BLD << "\n=== GAUSSIAN TRAINER ===" << RST << "\n"
              << " Input:   " << input << "\n";
    if (app_cfg.export_csv) std::cout << " CSV:     " << csv_output << "\n";
    if (app_cfg.export_bin) std::cout << " BIN:     " << bin_output << "\n";
    std::cout << " Threads: " << threads << "\n"
              << " Samples: " << app_cfg.sample_points << "\n"
              << " Mode:    " << (app_cfg.edt_mode == EDTMode::PURE ? "PURE (Unsigned)" : "SIGNED (Felzenszwalb)") << "\n"
              << CYN << "------------------------------------" << RST << "\n";

    // Load pointcloud
    auto t0 = std::chrono::high_resolution_clock::now();
    auto cloud = loadPointCloud(input);
    if (!cloud || cloud->empty()) {
        std::cerr << RED << "[ERROR] Failed to load pointcloud" << RST << "\n";
        return 1;
    }

    // Compute cube metadata (streaming - no index storage)
    CubeManager mgr(app_cfg.cube_size);
    mgr.computeCubeMetadata(cloud);
    size_t total = mgr.getCubeCount();
    if (total == 0) { std::cerr << RED << "[ERROR] No cubes" << RST << "\n"; return 1; }

    // Setup exporters
    std::unique_ptr<CSVExporter> csv_exporter;
    std::unique_ptr<BinaryExporter> bin_exporter;

    if (app_cfg.export_csv) {
        csv_exporter = std::make_unique<CSVExporter>(csv_output);
        if (!csv_exporter->isOpen()) {
            std::cerr << RED << "[ERROR] Cannot create CSV output!" << RST << std::endl;
            return 1;
        }
    }

    if (app_cfg.export_bin) {
        bin_exporter = std::make_unique<BinaryExporter>(bin_output);
        if (!bin_exporter->isOpen()) {
            std::cerr << RED << "[ERROR] Cannot create BIN output!" << RST << std::endl;
            return 1;
        }
    }

    // Configure trainer
    // --- Optimization: Downsample for Fast Skipping (Manual) ---
    std::cout << "[INFO] Creating coarse KdTree for fast skipping..." << std::flush;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_coarse(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Configurable downsampling
    int step = app_cfg.downsample_step;
    
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
    cfg.sample_points = app_cfg.sample_points;
    cfg.edt_mode = app_cfg.edt_mode;
    cfg.mae_threshold_good = app_cfg.mae_threshold_good;
    cfg.mae_threshold_max = app_cfg.mae_threshold_max;
    cfg.solver = app_cfg.solver;
    cfg.margin = (app_cfg.edt_mode == EDTMode::SIGNED) ? app_cfg.margin_signed : app_cfg.margin_pure;

    std::cout << CYN << "\n=== TRAINING " << total << " CUBES (Adaptive) ===" << RST << "\n";
    std::cout << "[INFO] Starting parallel training..." << std::endl;

    std::atomic<size_t> done(0), valid(0), discarded(0);
    std::atomic<double> sum_mae(0.0);
    std::atomic<double> sum_std_dev(0.0);
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
        // 1. Cube Classification Search: Strictly inside the cube
        float cube_radius = std::sqrt(3.0f) * mgr.getCubeSize() / 2.0f;
        
        pcl::PointXYZ center;
        center.x = info.centerX();
        center.y = info.centerY();
        center.z = info.centerZ();

        std::vector<int> indices;
        std::vector<float> dists;
        mgr.getKdTree().radiusSearch(center, cube_radius, indices, dists);

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
            std::vector<int> populated_steps;
            std::vector<int> empty_steps;
            double mae_threshold;
            double empty_distance_threshold;
        } adaptive_cfg;
        adaptive_cfg.populated_steps = app_cfg.adaptive_populated_steps;
        adaptive_cfg.empty_steps = app_cfg.adaptive_empty_steps;
        adaptive_cfg.mae_threshold = app_cfg.adaptive_mae_threshold;
        adaptive_cfg.empty_distance_threshold = app_cfg.empty_distance_threshold;

        TrainResult result;
        bool trained = false;

        // CASE 1: Populated Cube
        if (!cube.point_indices.empty()) {
             // --- Simple Linear Densification ---
            // Create a local cloud for EDT generation (cube points + context)
            pcl::PointCloud<pcl::PointXYZ>::Ptr edt_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            
            // Add cube points
            for (int idx : cube.point_indices) {
                edt_cloud->push_back(cloud->points[idx]);
            }

            // --- Context Search for EDT ---
            // EDT needs points beyond training area (margin + half cube extension)
            // Search radius covers: cube_size/2 (to corner) + margin + edt_extension
            float margin = (app_cfg.edt_mode == EDTMode::SIGNED) ? app_cfg.margin_signed : app_cfg.margin_pure;
            float edt_search_radius = std::sqrt(3.0f) * (mgr.getCubeSize() / 2.0f + margin + app_cfg.edt_extension); 

            std::vector<int> context_indices;
            std::vector<int> search_indices;
            std::vector<float> search_dists;
            
            if (mgr.getKdTree().radiusSearch(center, edt_search_radius, search_indices, search_dists) > 0) {
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

            // Add context points to edt_cloud
            for (int idx : context_indices) {
                edt_cloud->push_back(cloud->points[idx]);
            }

            // Build LOCAL KdTree for EDT generation
            pcl::KdTreeFLANN<pcl::PointXYZ> local_kdtree_edt;
            local_kdtree_edt.setInputCloud(edt_cloud);

            float edt_extension = app_cfg.edt_extension;
            LocalGrid grid = generateEDT(cube, edt_cloud, local_kdtree_edt, app_cfg.voxel_size, margin, mgr.getCubeSize(), edt_extension, app_cfg.edt_mode);
            
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

                    // 2. Gather points using Hybrid (Radius + Max Count)
                    // Use radius to ensure geometric validity, but limit count for speed
                    float search_radius = exact_dist + app_cfg.empty_search_margin;
                    unsigned int max_neighbors = app_cfg.empty_nearby_count;
                    
                    std::vector<int> local_indices;
                    std::vector<float> local_dists;
                    
                    pcl::PointCloud<pcl::PointXYZ>::Ptr local_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                    pcl::KdTreeFLANN<pcl::PointXYZ> local_kdtree_edt;
                    
                    int found = mgr.getKdTree().radiusSearch(searchPoint, search_radius, local_indices, local_dists, max_neighbors);

                    if (found > 0) {
                        for (int idx : local_indices) {
                            local_cloud->push_back(cloud->points[idx]);
                        }
                        local_kdtree_edt.setInputCloud(local_cloud);
                        
                        // Generate EDT using LOCAL KdTree and PURE mode
                        float edt_extension = app_cfg.edt_extension;
                        LocalGrid grid = generateEDT(cube, local_cloud, local_kdtree_edt, app_cfg.voxel_size, app_cfg.margin_pure, mgr.getCubeSize(), edt_extension, EDTMode::PURE);
                        
                        // Adaptive Loop for Empty
                        local_cfg.positive_only = true;
                        local_cfg.solver.use_importance_weighting = false;
                        local_cfg.margin = app_cfg.margin_pure;
                        
                        for (int ng : adaptive_cfg.empty_steps) {
                            result = trainGaussians(grid, ng, local_cfg);
                            if (result.valid && result.mae <= adaptive_cfg.mae_threshold) {
                                trained = true;
                                break;
                            }
                        }
                        if (result.valid) trained = true;
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
            double old_mae = sum_mae.load();
            while (!sum_mae.compare_exchange_weak(old_mae, old_mae + result.mae));

            double old_std = sum_std_dev.load();
            while (!sum_std_dev.compare_exchange_weak(old_std, old_std + result.std_dev));

            // Write to selected formats
            if (csv_exporter) csv_exporter->writeCube(cube, result, result.num_gaussians);
            if (bin_exporter) bin_exporter->writeCube(cube, result, result.num_gaussians);
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
    if (csv_exporter) csv_exporter->flush();
    
    // Finalize binary file with stats
    double avg_mae = valid > 0 ? sum_mae.load() / valid : 0.0;
    double avg_std_dev = valid > 0 ? sum_std_dev.load() / valid : 0.0;
    if (bin_exporter) {
        bin_exporter->finalize(
            static_cast<float>(avg_mae), 
            static_cast<float>(avg_std_dev), // Global std_dev
            app_cfg.cube_size,
            app_cfg.edt_mode == EDTMode::SIGNED ? app_cfg.margin_signed : app_cfg.margin_pure,
            app_cfg.empty_search_margin
        );
    }

    long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count();
    long train_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    std::cout << CYN << "\n=== SUMMARY ===" << RST << "\n";
    if (app_cfg.export_csv) std::cout << " CSV:       " << csv_output << "\n";
    if (app_cfg.export_bin) std::cout << " BIN:       " << bin_output << "\n";
    std::cout << " Valid:     " << valid << "/" << total << "\n"
              << " Discarded: " << discarded << "\n"
              << " Avg MAE:   " << std::fixed << std::setprecision(4) << avg_mae << " m\n"
              << " Time:      " << std::setprecision(2) << (total_ms / 1000.0) << " s\n"
              << " Speed:     " << std::setprecision(1) << (total * 1000.0 / train_ms) << " cubes/s\n"
              << CYN << "===============" << RST << "\n";

    return 0;
}
