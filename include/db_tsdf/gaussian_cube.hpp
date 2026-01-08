#ifndef __GAUSSIAN_CUBE_HPP__
#define __GAUSSIAN_CUBE_HPP__

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>
#include "db_tsdf/voxel_cube.hpp"
#include "solver/solver.hpp"

#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkMarchingCubes.h>
#include <vtkPolyData.h>

#define NUM_GAUSSIANS 16
#define PARAMS_PER_GAUSSIAN 7
#define N_PARAMS (NUM_GAUSSIANS * PARAMS_PER_GAUSSIAN)

struct CubeExportData {
    uint32_t index;
    float x, y, z;
    double mae;
    std::vector<double> params; 
};

struct Peak
{
    int idx;
    double val;
};

struct Coord
{
    int x, y, z;
};

class GaussianCube
{
public:
    std::vector<double> params;
    bool is_valid = false;
    vtkSmartPointer<vtkPolyData> mesh;
    uint32_t last_trained_frame = 0;

    double mae = 0.0;
    GaussianCube() : params(N_PARAMS, 0.0), is_valid(false), last_trained_frame(0), mae(0.0) {}

    void train(const VoxelCube& voxel_cube, Workspace& ws, int n_points)
    {
        const DenseGrid& grid = voxel_cube.toDenseGrid(ws);
        
        params = generate_initial_params(grid);
        
        sample_points_for_gd(grid, n_points, 0.0, 0.0, ws);
        std::vector<GMMData>& data = ws.gmm_data;
        
        if (data.empty()) {
            is_valid = false; 
            return;
        }

        GMMSolver solver;
        solver.solve(data, params);
        
        double error_sum = 0.0;
        int valid_count = 0;

        for (int z = 0; z < grid.nz; ++z) {
            for (int y = 0; y < grid.ny; ++y) {
                for (int x = 0; x < grid.nx; ++x) {
                    double wx = grid.min_x + x * grid.voxel_size;
                    double wy = grid.min_y + y * grid.voxel_size;
                    double wz = grid.min_z + z * grid.voxel_size;
                    
                    int idx = z * grid.nx * grid.ny + y * grid.nx + x;
                    double actual = grid.sdf_data[idx];
                    double pred = predict_sdf(wx, wy, wz);
                    
                    error_sum += std::abs(pred - actual);
                    valid_count++;
                }
            }
        }
        
        // MAE del cubo
        mae = (valid_count == 0) ? 0.0 : (error_sum / valid_count);
        is_valid = true;
    }
        
    static Coord idx_to_coord(int idx, int nx, int ny)
    {
        int z = idx / (nx * ny);
        int rem = idx % (nx * ny);
        int y = rem / nx;
        int x = rem % nx;
        return {x, y, z};
    }

    static bool is_local_extremum(const DenseGrid &grid, int x, int y, int z, bool find_max, int radius)
    {
        double center_val = grid.sdf_data[grid.idx(x, y, z)];

        for (int dz = -radius; dz <= radius; ++dz)
        {
            int nz = z + dz;
            if (nz < 0 || nz >= grid.nz) continue;

            for (int dy = -radius; dy <= radius; ++dy)
            {
                int ny = y + dy;
                if (ny < 0 || ny >= grid.ny) continue;

                for (int dx = -radius; dx <= radius; ++dx)
                {
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

    static std::vector<double> generate_initial_params(const DenseGrid &grid)
    {
        std::vector<double> x_params(N_PARAMS);
        std::mt19937 rng(42);

        double init_log_sigma = std::sqrt(0.08); 
        double w_init = 0.1;

        int peak_radius = 3;            
        int suppression_radius = 3;     

        std::vector<Peak> pos_candidates;
        std::vector<Peak> neg_candidates;

        for (int z = peak_radius; z < grid.nz - peak_radius; ++z)
        {
            for (int y = peak_radius; y < grid.ny - peak_radius; ++y)
            {
                for (int x_idx = peak_radius; x_idx < grid.nx - peak_radius; ++x_idx)
                {
                    int idx = grid.idx(x_idx, y, z);
                    double val = grid.sdf_data[idx];

                    if (val > 1e-3) {
                        if (is_local_extremum(grid, x_idx, y, z, true, peak_radius))
                            pos_candidates.push_back({idx, val});
                    } else if (val < -1e-3) {
                        if (is_local_extremum(grid, x_idx, y, z, false, peak_radius))
                            neg_candidates.push_back({idx, val});
                    }
                }
            }
        }

        std::sort(pos_candidates.begin(), pos_candidates.end(), [](const Peak &a, const Peak &b) { return a.val > b.val; });
        std::sort(neg_candidates.begin(), neg_candidates.end(), [](const Peak &a, const Peak &b) { return a.val < b.val; });

        auto run_nms = [&](const std::vector<Peak> &candidates, int target_count) -> std::vector<int>
        {
            std::vector<int> selected_indices;
            std::vector<bool> suppressed(grid.nx * grid.ny * grid.nz, false);

            for (const auto &p : candidates)
            {
                if ((int)selected_indices.size() >= target_count) break;
                if (suppressed[p.idx]) continue;

                selected_indices.push_back(p.idx);

                Coord c = idx_to_coord(p.idx, grid.nx, grid.ny);
                int r = suppression_radius;

                for (int dz = -r; dz <= r; ++dz)
                {
                    int nz = c.z + dz;
                    if (nz < 0 || nz >= grid.nz) continue;
                    for (int dy = -r; dy <= r; ++dy)
                    {
                        int ny = c.y + dy;
                        if (ny < 0 || ny >= grid.ny) continue;
                        for (int dx = -r; dx <= r; ++dx)
                        {
                            int nx = c.x + dx;
                            if (nx < 0 || nx >= grid.nx) continue;
                            suppressed[grid.idx(nx, ny, nz)] = true;
                        }
                    }
                }
            }
            return selected_indices;
        };

        int n_pos_target = NUM_GAUSSIANS / 2;
        int n_neg_target = NUM_GAUSSIANS - n_pos_target;

        std::vector<int> final_pos_indices = run_nms(pos_candidates, n_pos_target);
        std::vector<int> final_neg_indices = run_nms(neg_candidates, n_neg_target);

        int current_g = 0;

        auto write_gaussian = [&](int idx_grid, double weight_val)
        {
            if (current_g >= NUM_GAUSSIANS) return;

            Coord c = idx_to_coord(idx_grid, grid.nx, grid.ny);
            std::uniform_real_distribution<double> jitter(-0.1, 0.1);

            double px = grid.min_x + (c.x + 0.5 + jitter(rng)) * grid.voxel_size;
            double py = grid.min_y + (c.y + 0.5 + jitter(rng)) * grid.voxel_size;
            double pz = grid.min_z + (c.z + 0.5 + jitter(rng)) * grid.voxel_size;

            int base = current_g * PARAMS_PER_GAUSSIAN;
            x_params[base + 0] = px;
            x_params[base + 1] = py;
            x_params[base + 2] = pz;
            x_params[base + 3] = init_log_sigma; 
            x_params[base + 4] = init_log_sigma; 
            x_params[base + 5] = init_log_sigma; 
            x_params[base + 6] = weight_val;

            current_g++;
        };

        for (int idx : final_pos_indices) write_gaussian(idx, w_init);

        std::vector<int> all_pos_indices; 
        while (current_g < n_pos_target)
        {
            if (all_pos_indices.empty())
            {
                for (size_t i = 0; i < grid.sdf_data.size(); ++i)
                    if (grid.sdf_data[i] > 0) all_pos_indices.push_back(i);
            }
            if (all_pos_indices.empty()) break; 

            std::uniform_int_distribution<int> dist(0, all_pos_indices.size() - 1);
            write_gaussian(all_pos_indices[dist(rng)], w_init);
        }

        for (int idx : final_neg_indices) write_gaussian(idx, -w_init);

        std::vector<int> all_neg_indices;
        while (current_g < NUM_GAUSSIANS)
        {
            if (all_neg_indices.empty())
            {
                for (size_t i = 0; i < grid.sdf_data.size(); ++i)
                    if (grid.sdf_data[i] < 0) all_neg_indices.push_back(i);
            }
            if (all_neg_indices.empty()) break;

            std::uniform_int_distribution<int> dist(0, all_neg_indices.size() - 1);
            write_gaussian(all_neg_indices[dist(rng)], -w_init);
        }

        return x_params;
    }

    static double get_sdf_trilinear(const DenseGrid &grid, double x, double y, double z)
    {
        double fx = (x - grid.min_x) / grid.voxel_size;
        double fy = (y - grid.min_y) / grid.voxel_size;
        double fz = (z - grid.min_z) / grid.voxel_size;

        int x0 = static_cast<int>(std::floor(fx));
        int y0 = static_cast<int>(std::floor(fy));
        int z0 = static_cast<int>(std::floor(fz));

        if (x0 < 0) x0 = 0;
        if (x0 >= grid.nx - 1) x0 = grid.nx - 2;
        if (y0 < 0) y0 = 0;
        if (y0 >= grid.ny - 1) y0 = grid.ny - 2;
        if (z0 < 0) z0 = 0;
        if (z0 >= grid.nz - 1) z0 = grid.nz - 2;

        int x1 = x0 + 1;
        int y1 = y0 + 1;
        int z1 = z0 + 1;
        double xd = fx - x0;
        double yd = fy - y0;
        double zd = fz - z0;

        auto get_val = [&](int ix, int iy, int iz)
        { return grid.sdf_data[iz * grid.nx * grid.ny + iy * grid.nx + ix]; };

        double c00 = get_val(x0, y0, z0) * (1 - xd) + get_val(x1, y0, z0) * xd;
        double c10 = get_val(x0, y1, z0) * (1 - xd) + get_val(x1, y1, z0) * xd;
        double c01 = get_val(x0, y0, z1) * (1 - xd) + get_val(x1, y0, z1) * xd;
        double c11 = get_val(x0, y1, z1) * (1 - xd) + get_val(x1, y1, z1) * xd;
        double c0 = c00 * (1 - yd) + c10 * yd;
        double c1 = c01 * (1 - yd) + c11 * yd;
        return c0 * (1 - zd) + c1 * zd;
    }

    static void sample_points_for_gd(const DenseGrid &grid, int n_total, double band_frac, double band_tau, Workspace& ws)
    {
        int n_uniform = n_total;
        
        std::vector<GMMData>& data = ws.gmm_data;
        data.clear();

        if (data.capacity() < (size_t)n_total) data.reserve(n_total);
        
        std::mt19937 rng(42);

        double max_x = grid.min_x + (grid.nx - 1) * grid.voxel_size;
        double max_y = grid.min_y + (grid.ny - 1) * grid.voxel_size;
        double max_z = grid.min_z + (grid.nz - 1) * grid.voxel_size;

        std::uniform_real_distribution<double> dist_x(grid.min_x, max_x);
        std::uniform_real_distribution<double> dist_y(grid.min_y, max_y);
        std::uniform_real_distribution<double> dist_z(grid.min_z, max_z);

        for (int i = 0; i < n_uniform; ++i)
        {
            double x = dist_x(rng);
            double y = dist_y(rng);
            double z = dist_z(rng);
            data.push_back({x, y, z, get_sdf_trilinear(grid, x, y, z)});
        }
    }
    // --- Prediction ---
    double predict_sdf(double x, double y, double z) const
    {
        double sdf_pred = 0.0;
        double eps = 1e-6;

        for (int i = 0; i < NUM_GAUSSIANS; ++i)
        {
            int j = i * PARAMS_PER_GAUSSIAN;

            // Read params
            double mx = params[j + 0];
            double my = params[j + 1];
            double mz = params[j + 2];

            double p_l00 = params[j + 3];
            double l00 = p_l00 * p_l00 + eps;
            
            double p_l11 = params[j + 4];
            double l11 = p_l11 * p_l11 + eps;
            
            double p_l22 = params[j + 5];
            double l22 = p_l22 * p_l22 + eps;

            double w = params[j + 6];

            // Simplified Calculation (Diagonal)
            double vx = x - mx;
            double vy = y - my;
            double vz = z - mz;

            // z = v / L_diag
            double z0 = vx / l00;
            double z1 = vy / l11;
            double z2 = vz / l22;

            double dist_sq = z0 * z0 + z1 * z1 + z2 * z2;
            sdf_pred += w * std::exp(-0.5 * dist_sq);
        }
        return sdf_pred;
    }

    // --- Mesh Generation ---
    vtkSmartPointer<vtkPolyData> get_mesh(float min_x, float min_y, float min_z, float voxel_size, int nx, int ny, int nz, Workspace& ws) const
    {
        if (!is_valid) return nullptr;

        const int UPSAMPLE = 1; 
        
        double spacing = voxel_size / UPSAMPLE;
        int dimX = nx * UPSAMPLE + 1;
        int dimY = ny * UPSAMPLE + 1;
        int dimZ = nz * UPSAMPLE + 1;

        vtkSmartPointer<vtkImageData>& imageData = ws.vtk_image;
        imageData->SetDimensions(dimX, dimY, dimZ);
        imageData->SetSpacing(spacing, spacing, spacing);
        
        double origin_offset = -0.5 * voxel_size; 
        imageData->SetOrigin(min_x + origin_offset, min_y + origin_offset, min_z + origin_offset);
        
        if (imageData->GetScalarPointer() == nullptr) {
            imageData->AllocateScalars(VTK_FLOAT, 1);
        }

        float* scalars = static_cast<float*>(imageData->GetScalarPointer());

        for (int z = 0; z < dimZ; ++z)
        {
            for (int y = 0; y < dimY; ++y)
            {
                for (int x = 0; x < dimX; ++x)
                {
                    double wx = (min_x + origin_offset) + x * spacing;
                    double wy = (min_y + origin_offset) + y * spacing;
                    double wz = (min_z + origin_offset) + z * spacing;
                    
                    double val = predict_sdf(wx, wy, wz);
                    scalars[z * dimY * dimX + y * dimX + x] = static_cast<float>(val);
                }
            }
        }

        ws.vtk_mc->Update(); 
        
        vtkSmartPointer<vtkPolyData> output = vtkSmartPointer<vtkPolyData>::New();
        output->DeepCopy(ws.vtk_mc->GetOutput());
        
        return output;
    }
};

#endif
