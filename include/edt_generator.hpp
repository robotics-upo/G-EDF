#ifndef EDT_GENERATOR_HPP
#define EDT_GENERATOR_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "cube_manager.hpp"

enum class EDTMode {
    PURE,   // Unsigned Euclidean Distance (Direct KdTree)
    SIGNED  // Signed Distance (Felzenszwalb Propagation)
};

/// Local voxel grid
struct LocalGrid {
    int nx, ny, nz;
    float voxel_size;
    float cube_size;  // Size of the cube (not hardcoded to 1.0)
    float origin_x, origin_y, origin_z;
    float cube_origin_x, cube_origin_y, cube_origin_z;
    std::vector<float> sdf_data;
    std::vector<int8_t> sign_data; // Used for SIGNED mode (0=surface, 1=air)

    int idx(int x, int y, int z) const {
        if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return -1;
        return z * nx * ny + y * nx + x;
    }
};

// ==========================================
//  SIGNED EDT ALGORITHMS (Felzenszwalb)
// ==========================================

inline void dt_1d(const std::vector<float>& f, std::vector<float>& d,
                  std::vector<int>& v, std::vector<float>& z, int n) {
    int k = 0;
    v[0] = 0;
    z[0] = -1e9f;
    z[1] = 1e9f;

    for (int q = 1; q < n; ++q) {
        float s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
        while (s <= z[k]) {
            --k;
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0f * q - 2.0f * v[k]);
        }
        ++k;
        v[k] = q;
        z[k] = s;
        z[k + 1] = 1e9f;
    }

    k = 0;
    for (int q = 0; q < n; ++q) {
        while (z[k + 1] < q) ++k;
        float dx = q - v[k];
        d[q] = dx * dx + f[v[k]];
    }
}

inline void dt_3d(std::vector<float>& grid, int nx, int ny, int nz) {
    int max_dim = std::max({nx, ny, nz});
    std::vector<float> f(max_dim), d(max_dim);
    std::vector<int> v(max_dim);
    std::vector<float> z(max_dim + 1);

    // X pass
    for (int zi = 0; zi < nz; ++zi) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) f[x] = grid[zi * nx * ny + y * nx + x];
            dt_1d(f, d, v, z, nx);
            for (int x = 0; x < nx; ++x) grid[zi * nx * ny + y * nx + x] = d[x];
        }
    }
    // Y pass
    for (int zi = 0; zi < nz; ++zi) {
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) f[y] = grid[zi * nx * ny + y * nx + x];
            dt_1d(f, d, v, z, ny);
            for (int y = 0; y < ny; ++y) grid[zi * nx * ny + y * nx + x] = d[y];
        }
    }
    // Z pass
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            for (int zi = 0; zi < nz; ++zi) f[zi] = grid[zi * nx * ny + y * nx + x];
            dt_1d(f, d, v, z, nz);
            for (int zi = 0; zi < nz; ++zi) grid[zi * nx * ny + y * nx + x] = d[zi];
        }
    }
}

// ==========================================
//  GENERATORS
// ==========================================

inline void generateEDT_Pure(LocalGrid& grid,
                             const pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree) {
    std::vector<int> indices(1);
    std::vector<float> sq_distances(1);

    for (int iz = 0; iz < grid.nz; ++iz) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            for (int ix = 0; ix < grid.nx; ++ix) {
                pcl::PointXYZ query;
                query.x = grid.origin_x + (ix + 0.5f) * grid.voxel_size;
                query.y = grid.origin_y + (iy + 0.5f) * grid.voxel_size;
                query.z = grid.origin_z + (iz + 0.5f) * grid.voxel_size;

                float distance = 0.0f;
                if (kdtree.nearestKSearch(query, 1, indices, sq_distances) > 0) {
                    distance = std::sqrt(sq_distances[0]);
                }
                grid.sdf_data[grid.idx(ix, iy, iz)] = distance;
            }
        }
    }
}

inline void generateEDT_Signed(LocalGrid& grid,
                               const Cube& cube,
                               const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                               const pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree,
                               float margin,
                               float cube_size) {
    size_t total = grid.nx * grid.ny * grid.nz;
    grid.sign_data.assign(total, 1); // Default air

    // Search nearby points to mark surface
    pcl::PointXYZ center;
    center.x = cube.centerX();
    center.y = cube.centerY();
    center.z = cube.centerZ();

    float radius = std::sqrt(3.0f) * (cube_size / 2.0f + margin) * 1.5f;
    std::vector<int> indices;
    std::vector<float> dists;
    kdtree.radiusSearch(center, radius, indices, dists);

    for (int idx : indices) {
        const auto& pt = cloud->points[idx];
        int vx = int((pt.x - grid.origin_x) / grid.voxel_size);
        int vy = int((pt.y - grid.origin_y) / grid.voxel_size);
        int vz = int((pt.z - grid.origin_z) / grid.voxel_size);

        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            int i = grid.idx(vx + dx, vy + dy, vz + dz);
            if (i >= 0) grid.sign_data[i] = 0; // Surface
        }
    }

    // Compute transforms
    constexpr float INF = 1e9f;
    std::vector<float> dist_obs(total), dist_air(total);

    for (size_t i = 0; i < total; ++i) {
        dist_obs[i] = (grid.sign_data[i] == 0) ? 0.0f : INF;
        dist_air[i] = (grid.sign_data[i] == 1) ? 0.0f : INF;
    }

    dt_3d(dist_obs, grid.nx, grid.ny, grid.nz);
    dt_3d(dist_air, grid.nx, grid.ny, grid.nz);

    for (size_t i = 0; i < total; ++i) {
        float d_obs = std::sqrt(dist_obs[i]) * grid.voxel_size;
        float d_air = std::sqrt(dist_air[i]) * grid.voxel_size;
        
        // + outside (air), - inside (surface)
        grid.sdf_data[i] = (grid.sign_data[i] == 1) ? d_obs : -d_air;
    }
}

/// Wrapper function
inline LocalGrid generateEDT(const Cube& cube,
                             const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                             const pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree,
                             float voxel_size,
                             float margin,
                             float cube_size,
                             float edt_extension,
                             EDTMode mode) {
    LocalGrid grid;
    grid.voxel_size = voxel_size;
    grid.cube_size = cube_size;
    grid.cube_origin_x = cube.origin_x;
    grid.cube_origin_y = cube.origin_y;
    grid.cube_origin_z = cube.origin_z;
    
    // Grid extends: cube + margin (for training) + edt_extension (for blending context)
    float total_extension = margin + edt_extension;
    grid.origin_x = cube.origin_x - total_extension;
    grid.origin_y = cube.origin_y - total_extension;
    grid.origin_z = cube.origin_z - total_extension;

    // Grid covers: cube_size + 2 * (margin + edt_extension)
    float size = cube_size + 2.0f * total_extension;
    grid.nx = int(std::ceil(size / voxel_size)) + (mode == EDTMode::SIGNED ? 1 : 0);
    grid.ny = grid.nx;
    grid.nz = grid.nx;

    grid.sdf_data.resize(grid.nx * grid.ny * grid.nz);

    if (mode == EDTMode::PURE) {
        generateEDT_Pure(grid, kdtree);
    } else {
        generateEDT_Signed(grid, cube, cloud, kdtree, margin, cube_size);
    }

    return grid;
}

#endif // EDT_GENERATOR_HPP
