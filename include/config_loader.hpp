#ifndef CONFIG_LOADER_HPP
#define CONFIG_LOADER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>
#include "solver/solver.hpp"
#include "edt_generator.hpp"

/// Complete application configuration loaded from YAML
struct AppConfig {
    // Input/Output
    std::string input_file;
    std::string output_base;
    bool export_csv = true;
    bool export_bin = true;

    // Processing
    int num_threads = 0;
    float cube_size = 1.0f;

    // Downsampling
    int downsample_step = 10;

    // Trainer
    int sample_points = 1000;
    double mae_threshold_good = 0.03;
    double mae_threshold_max = 0.30;
    EDTMode edt_mode = EDTMode::PURE;

    // Solver
    SolverConfig solver;

    // Adaptive
    std::vector<int> adaptive_populated_steps = {8, 16, 32};
    std::vector<int> adaptive_empty_steps = {2, 4, 8, 16};
    double adaptive_mae_threshold = 0.03;
    double empty_distance_threshold = 3.0;

    // EDT
    float voxel_size = 0.025f;
    float margin_signed = 0.3f;
    float margin_pure = 0.0f;
    float empty_search_margin = 0.5f;
    float edt_extension = 0.1f;
    int empty_nearby_count = 100;
};

/// Load configuration from YAML file
class ConfigLoader {
public:
    static bool load(const std::string& filepath, AppConfig& cfg) {
        try {
            YAML::Node root = YAML::LoadFile(filepath);

            // IO
            if (root["io"]) {
                auto node = root["io"];
                cfg.input_file = node["input_file"].as<std::string>(cfg.input_file);
                cfg.output_base = node["output_base"].as<std::string>(cfg.output_base);
                cfg.export_csv = node["export_csv"].as<bool>(cfg.export_csv);
                cfg.export_bin = node["export_bin"].as<bool>(cfg.export_bin);
            }

            // Processing
            if (root["processing"]) {
                auto node = root["processing"];
                cfg.num_threads = node["num_threads"].as<int>(cfg.num_threads);
                cfg.cube_size = node["cube_size"].as<float>(cfg.cube_size);
            }

            // Downsampling
            if (root["downsampling"]) {
                auto node = root["downsampling"];
                cfg.downsample_step = node["step"].as<int>(cfg.downsample_step);
            }

            // Trainer
            if (root["trainer"]) {
                auto node = root["trainer"];
                cfg.sample_points = node["sample_points"].as<int>(cfg.sample_points);
                cfg.mae_threshold_good = node["mae_threshold_good"].as<double>(cfg.mae_threshold_good);
                cfg.mae_threshold_max = node["mae_threshold_max"].as<double>(cfg.mae_threshold_max);
                
                std::string mode = node["edt_mode"].as<std::string>("pure");
                cfg.edt_mode = (mode == "signed") ? EDTMode::SIGNED : EDTMode::PURE;
            }

            // Solver (only user-relevant params, rest are hardcoded in SolverConfig)
            if (root["solver"]) {
                auto node = root["solver"];
                cfg.solver.max_iterations = node["max_iterations"].as<int>(cfg.solver.max_iterations);
                cfg.solver.max_time_seconds = node["max_time_seconds"].as<double>(cfg.solver.max_time_seconds);
            }

            // Adaptive
            if (root["adaptive"]) {
                auto node = root["adaptive"];
                if (node["populated_steps"])
                    cfg.adaptive_populated_steps = node["populated_steps"].as<std::vector<int>>();
                if (node["empty_steps"])
                    cfg.adaptive_empty_steps = node["empty_steps"].as<std::vector<int>>();
                cfg.adaptive_mae_threshold = node["mae_threshold"].as<double>(cfg.adaptive_mae_threshold);
                cfg.empty_distance_threshold = node["empty_distance_threshold"].as<double>(cfg.empty_distance_threshold);
            }

            // EDT
            if (root["edt"]) {
                auto node = root["edt"];
                cfg.voxel_size = node["voxel_size"].as<float>(cfg.voxel_size);
                cfg.margin_signed = node["margin_signed"].as<float>(cfg.margin_signed);
                cfg.margin_pure = node["margin_pure"].as<float>(cfg.margin_pure);
                cfg.empty_search_margin = node["empty_search_margin"].as<float>(cfg.empty_search_margin);
                cfg.edt_extension = node["edt_extension"].as<float>(cfg.edt_extension);
                cfg.empty_nearby_count = node["empty_nearby_count"].as<int>(cfg.empty_nearby_count);
            }

            return true;

        } catch (const YAML::Exception& e) {
            std::cerr << "[CONFIG] YAML parse error: " << e.what() << std::endl;
            return false;
        } catch (const std::exception& e) {
            std::cerr << "[CONFIG] Error loading config: " << e.what() << std::endl;
            return false;
        }
    }

    /// Print current configuration
    static void print(const AppConfig& cfg) {
        std::cout << "\n=== Configuration ===" << std::endl;
        std::cout << "  Threads:        " << (cfg.num_threads == 0 ? "auto" : std::to_string(cfg.num_threads)) << std::endl;
        std::cout << "  Cube size:      " << cfg.cube_size << " m" << std::endl;
        std::cout << "  Sample points:  " << cfg.sample_points << std::endl;
        std::cout << "  MAE threshold:  " << cfg.mae_threshold_good << " / " << cfg.mae_threshold_max << std::endl;
        std::cout << "  Solver iters:   " << cfg.solver.max_iterations << std::endl;
        std::cout << "  Voxel size:     " << cfg.voxel_size << " m" << std::endl;
        std::cout << "=====================\n" << std::endl;
    }
};

#endif // CONFIG_LOADER_HPP
