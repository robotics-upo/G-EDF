/**
 * @file gaussian_to_ply.cpp
 * @brief Convert Gaussian CSV/GSDF to PLY mesh (marching cubes)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>



struct Gaussian {
    double x, y, z;
    double l0, l1, l2; // squared scales
    double w;
};

struct CubeData {
    int ix, iy, iz;
    double ox, oy, oz;
    std::vector<Gaussian> gaussians;
};

// Load CSV (One Gaussian per line)
std::vector<CubeData> loadCSV(const std::string& filename) {
    std::vector<CubeData> cubes;
    std::ifstream file(filename);
    if (!file.is_open()) return cubes;

    std::string line;
    // Skip header
    std::getline(file, line);

    int line_num = 1;
    CubeData current_cube;
    current_cube.ix = -999999; // Sentinel
    current_cube.ox = -999999;
    current_cube.oy = -999999;
    current_cube.oz = -999999;

    while (std::getline(file, line)) {
        line_num++;
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) tokens.push_back(token);

        if (tokens.size() < 13) continue;

        double ox = std::stod(tokens[0]);
        double oy = std::stod(tokens[1]);
        double oz = std::stod(tokens[2]);

        // Check if new cube
        if (std::abs(ox - current_cube.ox) > 1e-4 || 
            std::abs(oy - current_cube.oy) > 1e-4 || 
            std::abs(oz - current_cube.oz) > 1e-4) {
            
            if (current_cube.ix != -999999) {
                cubes.push_back(current_cube);
            }
            
            current_cube.ix = 0; // Dummy
            current_cube.iy = 0;
            current_cube.iz = 0;
            current_cube.ox = ox;
            current_cube.oy = oy;
            current_cube.oz = oz;
            current_cube.gaussians.clear();
        }

        Gaussian g;
        g.x = std::stod(tokens[6]);
        g.y = std::stod(tokens[7]);
        g.z = std::stod(tokens[8]);
        g.l0 = std::pow(std::stod(tokens[9]), 4);
        g.l1 = std::pow(std::stod(tokens[10]), 4);
        g.l2 = std::pow(std::stod(tokens[11]), 4);
        g.w = std::stod(tokens[12]);
        current_cube.gaussians.push_back(g);
    }
    
    // Push last cube
    if (current_cube.ix != -999999) {
        cubes.push_back(current_cube);
    }

    std::cout << "CSV Load Summary: Read " << line_num << " lines, Loaded " << cubes.size() << " cubes.\n";
    return cubes;
}

double predict(double x, double y, double z, const std::vector<Gaussian>& gs) {
    double val = 0.0;
    for (const auto& g : gs) {
        double dx = x - g.x;
        double dy = y - g.y;
        double dz = z - g.z;
        double dsq = (dx*dx)/g.l0 + (dy*dy)/g.l1 + (dz*dz)/g.l2;
        val += g.w * std::exp(-0.5 * dsq);
    }
    return val;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.csv> <output.ply> [threshold] [resolution] [xmin xmax ymin ymax zmin zmax]\n";
        return 1;
    }

    std::string input = argv[1];
    std::string output = argv[2];
    double threshold = (argc >= 4) ? std::stod(argv[3]) : 0.05;
    double resolution = (argc >= 5) ? std::stod(argv[4]) : 0.02;

    // Region filter
    bool use_region = (argc >= 11);
    double r_xmin = -1e9, r_xmax = 1e9, r_ymin = -1e9, r_ymax = 1e9, r_zmin = -1e9, r_zmax = 1e9;
    if (use_region) {
        r_xmin = std::stod(argv[5]); r_xmax = std::stod(argv[6]);
        r_ymin = std::stod(argv[7]); r_ymax = std::stod(argv[8]);
        r_zmin = std::stod(argv[9]); r_zmax = std::stod(argv[10]);
    }

    std::vector<CubeData> cubes = loadCSV(input);

    std::cout << "Loaded " << cubes.size() << " cubes.\n";

    pcl::PointCloud<pcl::PointXYZ> cloud;

    int empty_cubes = 0;
    for (const auto& c : cubes) {
        // Check region
        if (c.ox < r_xmin || c.ox > r_xmax || c.oy < r_ymin || c.oy > r_ymax || c.oz < r_zmin || c.oz > r_zmax)
            continue;

        size_t points_before = cloud.size();
        for (double z = c.oz; z < c.oz + 1.0; z += resolution) {
            for (double y = c.oy; y < c.oy + 1.0; y += resolution) {
                for (double x = c.ox; x < c.ox + 1.0; x += resolution) {
                    double val = predict(x, y, z, c.gaussians);
                    if (std::abs(val) < threshold) {
                        cloud.push_back(pcl::PointXYZ(x, y, z));
                    }
                }
            }
        }

        size_t points_added = cloud.size() - points_before;
        if (points_added == 0) {
            std::cout << "[WARNING] Cube at (" << c.ox << ", " << c.oy << ", " << c.oz << ") generated 0 points (Threshold: " << threshold << ")\n";
            empty_cubes++;
        }
    }

    if (empty_cubes > 0) {
        std::cout << "Total empty cubes: " << empty_cubes << " / " << cubes.size() << "\n";
    }

    if (cloud.empty()) {
        std::cerr << "No points generated!\n";
        return 1;
    }

    // Manual PLY writing for maximum compatibility (Blender, MeshLab, etc.)
    std::ofstream ofs(output, std::ios::binary);
    if (!ofs) {
        std::cerr << "Cannot open output file: " << output << "\n";
        return 1;
    }

    ofs << "ply\n";
    ofs << "format binary_little_endian 1.0\n";
    ofs << "comment Generated by GaussianTrainer\n";
    ofs << "element vertex " << cloud.size() << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "end_header\n";

    for (const auto& p : cloud.points) {
        float x = p.x;
        float y = p.y;
        float z = p.z;
        ofs.write(reinterpret_cast<const char*>(&x), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&y), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&z), sizeof(float));
    }
    ofs.close();

    std::cout << "Saved " << cloud.size() << " points to " << output << "\n";

    return 0;
}
