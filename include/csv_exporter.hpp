#ifndef CSV_EXPORTER_HPP
#define CSV_EXPORTER_HPP

#include <fstream>
#include <iomanip>
#include <vector>
#include <mutex>
#include "cube_manager.hpp"
#include "gaussian_trainer.hpp"

/// Write CSV header
inline void writeCSVHeader(std::ofstream& f) {
    f << "CubeX,CubeY,CubeZ,MAE,StdDev,G_ID,MeanX,MeanY,MeanZ,SigmaX,SigmaY,SigmaZ,Weight\n";
}

/// Write Gaussian params for one cube (thread-safe)
inline void writeCubeGaussians(std::ofstream& f, const Cube& cube,
                               const TrainResult& result, int ng, std::mutex& mtx) {
    if (!result.valid) return;
    std::lock_guard<std::mutex> lock(mtx);

    for (int g = 0; g < ng; ++g) {
        int b = g * PARAMS_PER_GAUSSIAN;
        f << std::fixed << std::setprecision(6)
          << cube.origin_x << "," << cube.origin_y << "," << cube.origin_z << ","
          << result.mae << "," << result.std_dev << "," << g
          << "," << result.params[b]       // Absolute X
          << "," << result.params[b + 1]   // Absolute Y
          << "," << result.params[b + 2]   // Absolute Z
          << "," << result.params[b + 3]   // Sigma X
          << "," << result.params[b + 4]   // Sigma Y
          << "," << result.params[b + 5]   // Sigma Z
          << "," << result.params[b + 6]   // Weight
          << "\n";
    }
}

/// Thread-safe CSV exporter
class CSVExporter {
public:
    explicit CSVExporter(const std::string& path) : path_(path) {
        file_.open(path);
        if (file_.is_open()) writeCSVHeader(file_);
    }
    ~CSVExporter() { if (file_.is_open()) file_.close(); }

    bool isOpen() const { return file_.is_open(); }

    void writeCube(const Cube& cube, const TrainResult& result, int ng) {
        writeCubeGaussians(file_, cube, result, ng, mtx_);
    }

    void flush() { std::lock_guard<std::mutex> lock(mtx_); file_.flush(); }

private:
    std::string path_;
    std::ofstream file_;
    std::mutex mtx_;
};

#endif // CSV_EXPORTER_HPP
