#ifndef BINARY_EXPORTER_HPP
#define BINARY_EXPORTER_HPP

#include <fstream>
#include <vector>
#include <mutex>
#include <cstring>
#include "cube_manager.hpp"
#include "gaussian_trainer.hpp"

// ==========================================
// Binary Format Structures (Packed)
// ==========================================

struct MapHeader {
    char magic[4] = {'G','D','F','1'}; // "GDF1" Format Identifier
    uint32_t version = 1;
    uint32_t num_cubes = 0;
    float avg_mae = 0.0f;
    float std_dev = 0.0f;
    float bounds_min[3] = {0,0,0};
    float bounds_max[3] = {0,0,0};
    float cube_size = 1.0f;
    float empty_search_margin = 0.0f;
    float cube_margin = 0.0f;
    uint8_t padding[64]; // Reserved for future use
};

struct CubeHeader {
    float origin[3];
    float mae;
    float std_dev;
    uint32_t num_gaussians;
};

struct GaussianData {
    uint32_t id;
    float mean[3];
    float sigma[3];
    float weight;
};

// ==========================================
// Exporter Class
// ==========================================

class BinaryExporter {
public:
    explicit BinaryExporter(const std::string& path) : path_(path) {
        file_.open(path, std::ios::binary);
        // Write placeholder header
        if (file_.is_open()) {
            MapHeader header;
            file_.write(reinterpret_cast<const char*>(&header), sizeof(MapHeader));
        }
    }

    ~BinaryExporter() {
        if (file_.is_open()) file_.close();
    }

    bool isOpen() const { return file_.is_open(); }

    void writeCube(const Cube& cube, const TrainResult& result, int ng) {
        if (!result.valid) return;
        
        std::lock_guard<std::mutex> lock(mtx_);

        // 1. Write Cube Header
        CubeHeader ch;
        ch.origin[0] = cube.origin_x;
        ch.origin[1] = cube.origin_y;
        ch.origin[2] = cube.origin_z;
        ch.mae = static_cast<float>(result.mae);
        ch.std_dev = static_cast<float>(result.std_dev);
        ch.num_gaussians = static_cast<uint32_t>(ng);
        
        file_.write(reinterpret_cast<const char*>(&ch), sizeof(CubeHeader));

        // 2. Write Gaussians
        for (int g = 0; g < ng; ++g) {
            int b = g * PARAMS_PER_GAUSSIAN;
            GaussianData gd;
            gd.id = static_cast<uint32_t>(g);
            
            // Params: [mx, my, mz, sx, sy, sz, w]
            gd.mean[0] = static_cast<float>(result.params[b]);
            gd.mean[1] = static_cast<float>(result.params[b+1]);
            gd.mean[2] = static_cast<float>(result.params[b+2]);
            
            gd.sigma[0] = static_cast<float>(result.params[b+3]);
            gd.sigma[1] = static_cast<float>(result.params[b+4]);
            gd.sigma[2] = static_cast<float>(result.params[b+5]);
            
            gd.weight = static_cast<float>(result.params[b+6]);

            file_.write(reinterpret_cast<const char*>(&gd), sizeof(GaussianData));
        }
        
        // Update stats
        num_cubes_++;
        // Bounds update (simple approximation using cube origins)
        if (num_cubes_ == 1) {
            min_b_[0] = cube.origin_x; min_b_[1] = cube.origin_y; min_b_[2] = cube.origin_z;
            max_b_[0] = cube.origin_x; max_b_[1] = cube.origin_y; max_b_[2] = cube.origin_z;
        } else {
            if (cube.origin_x < min_b_[0]) min_b_[0] = cube.origin_x;
            if (cube.origin_y < min_b_[1]) min_b_[1] = cube.origin_y;
            if (cube.origin_z < min_b_[2]) min_b_[2] = cube.origin_z;
            
            if (cube.origin_x > max_b_[0]) max_b_[0] = cube.origin_x;
            if (cube.origin_y > max_b_[1]) max_b_[1] = cube.origin_y;
            if (cube.origin_z > max_b_[2]) max_b_[2] = cube.origin_z;
        }
    }

    void finalize(float avg_mae, float std_dev, float cube_size, float margin, float empty_margin) {
        if (!file_.is_open()) return;
        
        std::lock_guard<std::mutex> lock(mtx_);
        
        // Go back to start
        file_.seekp(0);
        
        MapHeader header;
        header.num_cubes = num_cubes_;
        header.avg_mae = avg_mae;
        header.std_dev = std_dev;
        
        header.bounds_min[0] = min_b_[0];
        header.bounds_min[1] = min_b_[1];
        header.bounds_min[2] = min_b_[2];
        
        header.bounds_max[0] = max_b_[0] + cube_size; // Extend to full cube
        header.bounds_max[1] = max_b_[1] + cube_size;
        header.bounds_max[2] = max_b_[2] + cube_size;
        
        header.cube_size = cube_size;
        header.cube_margin = margin;
        header.empty_search_margin = empty_margin;
        
        file_.write(reinterpret_cast<const char*>(&header), sizeof(MapHeader));
        file_.flush();
    }

private:
    std::string path_;
    std::ofstream file_;
    std::mutex mtx_;
    
    uint32_t num_cubes_ = 0;
    float min_b_[3] = {0,0,0};
    float max_b_[3] = {0,0,0};
};

#endif // BINARY_EXPORTER_HPP
