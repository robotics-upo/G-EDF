/**
 * @file gaussian_to_ply.cpp
 * @brief Convert Gaussian CSV to PLY point cloud with optional smooth blending
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <cstring>
#include <omp.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>

const char* RST = "\033[0m";
const char* GRN = "\033[32m";
const char* RED = "\033[31m";
const char* CYN = "\033[36m";

struct ReconstructionConfig {
    std::string input_csv;
    std::string output_ply;
    double threshold = 0.05;
    double resolution = 0.02;
    double max_mae = 100.0;
    bool region_enabled = false;
    double x_min = 0, x_max = 0;
    double y_min = 0, y_max = 0;
    double z_min = 0, z_max = 0;
    bool blending_enabled = false;
    double blending_margin = 0.2;
};

bool loadConfig(const std::string& filepath, ReconstructionConfig& cfg) {
    try {
        YAML::Node root = YAML::LoadFile(filepath);
        if (root["io"]) {
            cfg.input_csv = root["io"]["input_csv"].as<std::string>(cfg.input_csv);
            cfg.output_ply = root["io"]["output_ply"].as<std::string>(cfg.output_ply);
        }
        if (root["reconstruction"]) {
            cfg.threshold = root["reconstruction"]["threshold"].as<double>(cfg.threshold);
            cfg.resolution = root["reconstruction"]["resolution"].as<double>(cfg.resolution);
            cfg.max_mae = root["reconstruction"]["max_mae"].as<double>(cfg.max_mae);
        }
        if (root["region"]) {
            cfg.region_enabled = root["region"]["enabled"].as<bool>(cfg.region_enabled);
            cfg.x_min = root["region"]["x_min"].as<double>(cfg.x_min);
            cfg.x_max = root["region"]["x_max"].as<double>(cfg.x_max);
            cfg.y_min = root["region"]["y_min"].as<double>(cfg.y_min);
            cfg.y_max = root["region"]["y_max"].as<double>(cfg.y_max);
            cfg.z_min = root["region"]["z_min"].as<double>(cfg.z_min);
            cfg.z_max = root["region"]["z_max"].as<double>(cfg.z_max);
        }
        if (root["blending"]) {
            cfg.blending_enabled = root["blending"]["enabled"].as<bool>(cfg.blending_enabled);
            cfg.blending_margin = root["blending"]["margin"].as<double>(cfg.blending_margin);
        }
        return true;
    } catch (const YAML::Exception& e) {
        std::cerr << RED << "[ERROR] YAML: " << e.what() << RST << std::endl;
        return false;
    }
}

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
    
    void clear() {
        x.clear(); y.clear(); z.clear();
        inv_l0.clear(); inv_l1.clear(); inv_l2.clear();
        w.clear();
    }
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

// ==========================================
// Binary Format Structures (Must match Exporter)
// ==========================================
struct MapHeader {
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

std::vector<CubeData> loadBinary(const std::string& filename, float* margin_out = nullptr) {
    std::vector<CubeData> cubes;
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return cubes;

    MapHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(MapHeader));
    
    if (std::strncmp(header.magic, "GDF1", 4) != 0) {
        std::cerr << RED << "[ERROR] Invalid binary format or version mismatch" << RST << std::endl;
        return cubes;
    }
    
    if (margin_out) *margin_out = header.cube_margin;

    std::cout << "[INFO] Binary Map Info:\n"
              << "       Version: " << header.version << "\n"
              << "       Cubes:   " << header.num_cubes << "\n"
              << "       Avg MAE: " << header.avg_mae << " m\n"
              << "       Margin:  " << header.cube_margin << " m\n"
              << "       Bounds:  [" << header.bounds_min[0] << ", " << header.bounds_min[1] << ", " << header.bounds_min[2] << "] to ["
              << header.bounds_max[0] << ", " << header.bounds_max[1] << ", " << header.bounds_max[2] << "]\n";

    cubes.reserve(header.num_cubes);

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
            
            // In binary we stored raw sigma, here we need squared scales (l0, l1, l2)
            // Matches CSVExporter behavior where params[3] is the scale parameter (sigma^4)
            
            double l0 = std::pow(gd.sigma[0], 4);
            double l1 = std::pow(gd.sigma[1], 4);
            double l2 = std::pow(gd.sigma[2], 4);
            
            cube.gaussians.push_back(gd.mean[0], gd.mean[1], gd.mean[2], l0, l1, l2, gd.weight);
        }
        cubes.push_back(cube);
    }
    return cubes;
}


std::vector<CubeData> loadCSV(const std::string& filename, float* margin_out = nullptr);

std::vector<CubeData> loadMap(const std::string& filename, float* margin_out = nullptr) {
    if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".bin") {
        return loadBinary(filename, margin_out);
    }
    return loadCSV(filename, margin_out);
}
std::vector<CubeData> loadCSV(const std::string& filename, float* margin_out) {
    std::vector<CubeData> cubes;
    std::ifstream file(filename);
    if (!file.is_open()) return cubes;

    std::string line;
    std::getline(file, line);

    CubeData current;
    current.ox = -999999;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) tokens.push_back(token);
        if (tokens.size() < 13) continue;

        double ox = std::stod(tokens[0]);
        double oy = std::stod(tokens[1]);
        double oz = std::stod(tokens[2]);

        if (std::abs(ox - current.ox) > 1e-4 || 
            std::abs(oy - current.oy) > 1e-4 || 
            std::abs(oz - current.oz) > 1e-4) {
            if (current.ox != -999999) cubes.push_back(current);
            current.ox = ox; current.oy = oy; current.oz = oz;
            current.gaussians.clear();
        }

        double gx = std::stod(tokens[6]);
        double gy = std::stod(tokens[7]);
        double gz = std::stod(tokens[8]);
        double l0 = std::pow(std::stod(tokens[9]), 4);
        double l1 = std::pow(std::stod(tokens[10]), 4);
        double l2 = std::pow(std::stod(tokens[11]), 4);
        double w = std::stod(tokens[12]);
        
        current.gaussians.push_back(gx, gy, gz, l0, l1, l2, w);
    }
    if (current.ox != -999999) cubes.push_back(current);
    return cubes;
}

#include <immintrin.h>

// ==========================================
// AVX2 Fast Exponential Approximation
// ==========================================
// Based on typical high-performance implementations (e.g., fmath, VCL)
// Computes exp(x) for x <= 0. Approximates exp(x) = 2^(x * log2(e))
inline __m256 fast_exp_avx(__m256 x) {
    // Constants
    const __m256 log2e = _mm256_set1_ps(1.44269504088896340736f);
    const __m256 ln2   = _mm256_set1_ps(0.69314718056f);
    
    // Underflow check: exp(-88) is approx 5.9e-39, close to FLT_MIN.
    // Inputs smaller than this should return 0.0f to avoid garbage from bit shifting negative exponents.
    const __m256 min_input = _mm256_set1_ps(-88.0f);
    __m256 zero_mask = _mm256_cmp_ps(x, min_input, _CMP_GT_OS); // 0xFFFFFFFF if x > -88, else 0

    // 1. x * log2(e)
    __m256 t = _mm256_mul_ps(x, log2e);

    // 2. Round to integer (exponent part)
    __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    
    // 3. Fraction part: r = x - n*ln2
    __m256 r = _mm256_fnmadd_ps(n, ln2, x);

    // 4. Polynomial approximation for exp(r) in [-0.5*ln2, 0.5*ln2]
    // Degree 4 Taylor series: 1 + r + r^2/2 + r^3/6 + r^4/24
    const __m256 p0 = _mm256_set1_ps(1.0f / 24.0f);
    const __m256 p1 = _mm256_set1_ps(1.0f / 6.0f);
    const __m256 p2 = _mm256_set1_ps(0.5f);
    const __m256 p3 = _mm256_set1_ps(1.0f);
    
    __m256 poly = _mm256_fmadd_ps(p0, r, p1); // p0*r + p1
    poly = _mm256_fmadd_ps(poly, r, p2);      // (..)*r + p2
    poly = _mm256_fmadd_ps(poly, r, p3);      // (..)*r + p3
    poly = _mm256_fmadd_ps(poly, r, p3);      // (..)*r + 1

    // 5. Build 2^n
    __m256i n_int = _mm256_cvtps_epi32(n);
    // Add bias (127) and shift to exponent position (23)
    __m256i e_int = _mm256_slli_epi32(_mm256_add_epi32(n_int, _mm256_set1_epi32(127)), 23);
    __m256 pow2n = _mm256_castsi256_ps(e_int);

    // 6. Combine and apply zero mask
    __m256 result = _mm256_mul_ps(poly, pow2n);
    return _mm256_and_ps(result, zero_mask);
}

double predict(double x, double y, double z, const GaussianSoA& gs) {
    float fx = static_cast<float>(x);
    float fy = static_cast<float>(y);
    float fz = static_cast<float>(z);
    
    size_t n = gs.x.size();
    const float* __restrict__ px = gs.x.data();
    const float* __restrict__ py = gs.y.data();
    const float* __restrict__ pz = gs.z.data();
    const float* __restrict__ il0 = gs.inv_l0.data();
    const float* __restrict__ il1 = gs.inv_l1.data();
    const float* __restrict__ il2 = gs.inv_l2.data();
    const float* __restrict__ pw = gs.w.data();

    // AVX2 Loop
    __m256 v_val = _mm256_setzero_ps();
    __m256 v_fx = _mm256_set1_ps(fx);
    __m256 v_fy = _mm256_set1_ps(fy);
    __m256 v_fz = _mm256_set1_ps(fz);
    __m256 v_minus_half = _mm256_set1_ps(-0.5f);
    
    size_t i = 0;
    // Process 8 elements at a time
    for (; i + 7 < n; i += 8) {
        // Load Data (Unaligned loads are fast on modern CPUs)
        __m256 v_px = _mm256_loadu_ps(px + i);
        __m256 v_py = _mm256_loadu_ps(py + i);
        __m256 v_pz = _mm256_loadu_ps(pz + i);
        
        __m256 v_il0 = _mm256_loadu_ps(il0 + i);
        __m256 v_il1 = _mm256_loadu_ps(il1 + i);
        __m256 v_il2 = _mm256_loadu_ps(il2 + i);
        __m256 v_pw  = _mm256_loadu_ps(pw + i);

        // Calculate deltas
        __m256 dx = _mm256_sub_ps(v_fx, v_px);
        __m256 dy = _mm256_sub_ps(v_fy, v_py);
        __m256 dz = _mm256_sub_ps(v_fz, v_pz);

        // Square and scale: dsq = dx*dx*il0 + ...
        __m256 dsq = _mm256_mul_ps(_mm256_mul_ps(dx, dx), v_il0);
        dsq = _mm256_fmadd_ps(_mm256_mul_ps(dy, dy), v_il1, dsq);
        dsq = _mm256_fmadd_ps(_mm256_mul_ps(dz, dz), v_il2, dsq);

        // Calculate exp(-0.5 * dsq)
        __m256 arg = _mm256_mul_ps(dsq, v_minus_half);
        __m256 exp_val = fast_exp_avx(arg);

        // Accumulate: val += w * exp
        v_val = _mm256_fmadd_ps(v_pw, exp_val, v_val);
    }

    // Horizontal sum of AVX register
    // (There are faster ways but this is not the bottleneck compared to the loop)
    float temp[8];
    _mm256_storeu_ps(temp, v_val);
    double val = 0.0;
    for (int k = 0; k < 8; ++k) val += temp[k];

    // Tail loop (scalar)
    for (; i < n; ++i) {
        float dx = fx - px[i];
        float dy = fy - py[i];
        float dz = fz - pz[i];
        float dsq = (dx*dx)*il0[i] + (dy*dy)*il1[i] + (dz*dz)*il2[i];
        val += pw[i] * std::exp(-0.5f * dsq);
    }
    
    return val;
}

/**
 * Smoothstep: C1 continuous interpolation (3t² - 2t³)
 * t in [0,1] → output in [0,1]
 */
inline double smoothstep(double t) {
    t = std::max(0.0, std::min(1.0, t));
    return t * t * (3.0 - 2.0 * t);
}

/**
 * Calculate blend weight: 1.0 at center, fades to 0 at margin boundary
 * Uses per-axis minimum distance to cube boundary
 */
double blendWeight(double px, double py, double pz,
                   double cx, double cy, double cz,
                   double cube_size, double margin) {
    // Distance inside cube (positive = inside, negative = in margin)
    double dist_x = std::min(px - cx, cx + cube_size - px);
    double dist_y = std::min(py - cy, cy + cube_size - py);
    double dist_z = std::min(pz - cz, cz + cube_size - pz);
    double min_dist = std::min({dist_x, dist_y, dist_z});

    if (min_dist >= 0) return 1.0;  // Inside cube core
    if (min_dist <= -margin) return 0.0;  // Outside margin

    // In margin zone: smooth falloff
    return smoothstep(1.0 + min_dist / margin);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>\n";
        return 1;
    }

    ReconstructionConfig cfg;
    if (!loadConfig(argv[1], cfg)) return 1;
    std::cout << GRN << "[INFO] Loaded config: " << argv[1] << RST << std::endl;

    if (cfg.input_csv.empty() || cfg.output_ply.empty()) {
        std::cerr << RED << "[ERROR] input_csv and output_ply required" << RST << std::endl;
        return 1;
    }

    std::cout << CYN << "\n=== GAUSSIAN TO PLY ===" << RST << "\n"
              << " Input:      " << cfg.input_csv << "\n"
              << " Output:     " << cfg.output_ply << "\n"
              << " Threshold:  " << cfg.threshold << " m\n"
              << " Resolution: " << cfg.resolution << " m\n"
              << " Max MAE:    " << cfg.max_mae << " m\n"
              << " Blending:   " << (cfg.blending_enabled ? "ON" : "off") << "\n"
              << CYN << "------------------------" << RST << "\n";

    float file_margin = -1.0f;
    std::vector<CubeData> cubes = loadMap(cfg.input_csv, &file_margin);
    std::cout << "[INFO] Loaded " << cubes.size() << " cubes\n";

    if (file_margin > 0.0f && cfg.blending_enabled) {
        cfg.blending_margin = file_margin;
    }

    // Build spatial index
    const double cube_size = 1.0;
    std::unordered_map<std::tuple<int,int,int>, size_t, CubeKeyHash> cube_index;
    for (size_t i = 0; i < cubes.size(); ++i) {
        int ix = static_cast<int>(std::round(cubes[i].ox));
        int iy = static_cast<int>(std::round(cubes[i].oy));
        int iz = static_cast<int>(std::round(cubes[i].oz));
        cube_index[{ix, iy, iz}] = i;
    }

    pcl::PointCloud<pcl::PointXYZ> cloud;
    const double margin = cfg.blending_margin;
    
    // Timing stats
    std::atomic<size_t> total_evals{0};
    std::atomic<size_t> progress{0};
    std::atomic<double> total_cpu_time{0.0};  // Sum of all thread times
    auto time_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        pcl::PointCloud<pcl::PointXYZ> local_cloud;
        size_t local_evals = 0;
        auto thread_start = std::chrono::high_resolution_clock::now();

        #pragma omp for schedule(dynamic, 10)
        for (size_t i = 0; i < cubes.size(); ++i) {
            const CubeData& c = cubes[i];
            
            // Filter by MAE
            if (c.mae > cfg.max_mae) {
                progress++;
                continue;
            }
            
            size_t p = ++progress;
            if (p % 500 == 0) {
                #pragma omp critical
                std::cout << "\r[" << p << "/" << cubes.size() << "]" << std::flush;
            }

        if (cfg.region_enabled) {
            if (c.ox < cfg.x_min || c.ox > cfg.x_max ||
                c.oy < cfg.y_min || c.oy > cfg.y_max ||
                c.oz < cfg.z_min || c.oz > cfg.z_max)
                continue;
        }

        // Pre-fetch neighbors for this cube to avoid hash lookups per point
        std::vector<const CubeData*> neighbors;
        neighbors.reserve(27);
        
        if (cfg.blending_enabled) {
            int ix = static_cast<int>(std::round(c.ox));
            int iy = static_cast<int>(std::round(c.oy));
            int iz = static_cast<int>(std::round(c.oz));

            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        // Skip center? No, we need it for the weighted sum if we are in the margin.
                        // Actually, if we are in the margin of a neighbor, we blend.
                        // If we are in the center of the current cube, we might still be in the margin of a neighbor.
                        // So we just collect all 27 potential neighbors.
                        
                        auto key = std::make_tuple(ix + dx, iy + dy, iz + dz);
                        auto it = cube_index.find(key);
                        if (it != cube_index.end()) {
                            neighbors.push_back(&cubes[it->second]);
                        }
                    }
                }
            }
        }

        for (double z = c.oz; z < c.oz + cube_size; z += cfg.resolution) {
            for (double y = c.oy; y < c.oy + cube_size; y += cfg.resolution) {
                for (double x = c.ox; x < c.ox + cube_size; x += cfg.resolution) {
                    
                    double val;

                    if (!cfg.blending_enabled) {
                        val = predict(x, y, z, c.gaussians);
                    } else {
                        // Check if point is near any edge (within margin distance)
                        double dx_min = x - c.ox;
                        double dx_max = c.ox + cube_size - x;
                        double dy_min = y - c.oy;
                        double dy_max = c.oy + cube_size - y;
                        double dz_min = z - c.oz;
                        double dz_max = c.oz + cube_size - z;
                        
                        bool near_edge = (dx_min < margin || dx_max < margin ||
                                         dy_min < margin || dy_max < margin ||
                                         dz_min < margin || dz_max < margin);

                        if (!near_edge) {
                            // Fast path: point is in center, only current cube contributes
                            val = predict(x, y, z, c.gaussians);
                        } else {
                            // Slow path: point near edge, must blend with neighbors
                            double weighted_sum = 0.0;
                            double weight_total = 0.0;

                            for (const auto* nb_ptr : neighbors) {
                                const CubeData& nb = *nb_ptr;
                                
                                // Strict check: is the point actually within the influence zone of this neighbor?
                                // Influence zone = [ox - margin, ox + size + margin]
                                if (x < nb.ox - margin || x > nb.ox + cube_size + margin ||
                                    y < nb.oy - margin || y > nb.oy + cube_size + margin ||
                                    z < nb.oz - margin || z > nb.oz + cube_size + margin)
                                    continue;

                                double w = blendWeight(x, y, z, nb.ox, nb.oy, nb.oz, cube_size, margin);
                                if (w > 1e-6) {
                                    weighted_sum += w * predict(x, y, z, nb.gaussians);
                                    weight_total += w;
                                }
                            }

                            if (weight_total < 1e-6) continue;
                            val = weighted_sum / weight_total;
                        }
                    }

                    if (std::abs(val) < cfg.threshold) {
                        local_cloud.push_back(pcl::PointXYZ(x, y, z));
                    }
                    ++local_evals;
                }
            }
        }
    }

    auto thread_end = std::chrono::high_resolution_clock::now();
    double thread_sec = std::chrono::duration<double>(thread_end - thread_start).count();
    
    #pragma omp critical
    {
        cloud += local_cloud;
        total_evals += local_evals;
        double old = total_cpu_time.load();
        while (!total_cpu_time.compare_exchange_weak(old, old + thread_sec));
    }
    } // end parallel
    
    std::cout << "\r                                              \r";
    
    auto time_end = std::chrono::high_resolution_clock::now();
    double wall_sec = std::chrono::duration<double>(time_end - time_start).count();
    double cpu_sec = total_cpu_time.load();
    double avg_us = (total_evals > 0) ? (cpu_sec * 1e6 / total_evals) : 0;
    
    std::cout << "[STATS] " << total_evals << " points in " 
              << std::fixed << std::setprecision(2) << wall_sec << " s (wall), "
              << cpu_sec << " s (CPU)\n";
    std::cout << "[STATS] Avg CPU time per point: " << std::setprecision(3) << avg_us << " µs\n";

    if (cloud.empty()) {
        std::cerr << RED << "[ERROR] No points generated!" << RST << std::endl;
        return 1;
    }

    std::ofstream ofs(cfg.output_ply, std::ios::binary);
    if (!ofs) {
        std::cerr << RED << "[ERROR] Cannot open: " << cfg.output_ply << RST << std::endl;
        return 1;
    }

    ofs << "ply\nformat binary_little_endian 1.0\n";
    ofs << "comment Generated by GaussDF\n";
    ofs << "element vertex " << cloud.size() << "\n";
    ofs << "property float x\nproperty float y\nproperty float z\nend_header\n";

    for (const auto& p : cloud.points) {
        float fx = p.x, fy = p.y, fz = p.z;
        ofs.write(reinterpret_cast<const char*>(&fx), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&fy), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&fz), sizeof(float));
    }

    std::cout << GRN << "[SUCCESS] " << cloud.size() << " points → " << cfg.output_ply << RST << std::endl;
    return 0;
}
