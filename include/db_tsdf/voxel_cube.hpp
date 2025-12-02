#ifndef __VOXEL_CUBE_HPP__
#define __VOXEL_CUBE_HPP__

#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "db_tsdf/grid_16.hpp" // For VoxelData struct
#include "solver/solver.hpp" // For GMMData

struct VoxelData
{
	uint16_t d;		// Manhattan mask (bit-count -> distance)
	uint8_t s;		// bit0: sign (0 occ / 1 free)
	uint8_t hits;	// hit counter

    int8_t offX; 
    int8_t offY;
    int8_t offZ;
    uint8_t posHits;
};
static_assert(sizeof(VoxelData) == 8, "VoxelData must be 8-bytes aligned");

struct DenseGrid
{
    int nx, ny, nz;
    double voxel_size;
    double min_x, min_y, min_z;
    std::vector<float> sdf_data;
    
    int idx(int x, int y, int z) const {
        if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return -1;
        return z * nx * ny + y * nx + x;
    }
};

struct Workspace {
    std::vector<int8_t> temp_grid;
    std::vector<float> dist_to_obstacle;
    std::vector<float> dist_to_air;
    // For dt_3d
    std::vector<float> f, d, z;
    std::vector<int> v;
    // Result
    DenseGrid dense_grid;
    
    // Gaussian Training Data
    std::vector<GMMData> gmm_data;

    vtkSmartPointer<vtkImageData> vtk_image;
    vtkSmartPointer<vtkMarchingCubes> vtk_mc;

    Workspace() {
        vtk_image = vtkSmartPointer<vtkImageData>::New();
        vtk_mc = vtkSmartPointer<vtkMarchingCubes>::New();
        vtk_mc->SetInputData(vtk_image); // Conectamos una vez para siempre
        vtk_mc->ComputeNormalsOff();    
        vtk_mc->ComputeGradientsOff();
    }
    
    // Helper to ensure size
    void resize(size_t total_voxels, int max_dim) {
        if (temp_grid.size() < total_voxels) temp_grid.resize(total_voxels);
        if (dist_to_obstacle.size() < total_voxels) dist_to_obstacle.resize(total_voxels);
        if (dist_to_air.size() < total_voxels) dist_to_air.resize(total_voxels);
        if (dense_grid.sdf_data.size() < total_voxels) dense_grid.sdf_data.resize(total_voxels);
        
        if (f.size() < (size_t)max_dim) f.resize(max_dim);
        if (d.size() < (size_t)max_dim) d.resize(max_dim);
        if (z.size() < (size_t)max_dim + 1) z.resize(max_dim + 1);
        if (v.size() < (size_t)max_dim) v.resize(max_dim);
        
        // gmm_data is resized in sample_points_for_gd, but we can reserve
        if (gmm_data.capacity() < 500) gmm_data.reserve(500);
    }
};

class VoxelCube
{
public:
    VoxelCube(VoxelData* data, uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ, float res, float x0, float y0, float z0)
        : _data(data), _sizeX(sizeX), _sizeY(sizeY), _sizeZ(sizeZ), _res(res), _x0(x0), _y0(y0), _z0(z0)
    {
    }

    const DenseGrid& toDenseGrid(Workspace& ws) const
    {
        DenseGrid& grid = ws.dense_grid;
        // Add padding +2
        grid.nx = _sizeX + 2;
        grid.ny = _sizeY + 2;
        grid.nz = _sizeZ + 2;
        grid.voxel_size = _res;
        // Fix: The grid has 1-voxel padding at the start, so the physical origin of index 0 is shifted by -res
        // AND we need to align to voxel centers to match main.cpp (which reads CSVs with centers)
        // CSV min is Center. grid.min_x = Center - res.
        // _x0 is Corner. Center = _x0 + 0.5*res.
        // So grid.min_x = (_x0 + 0.5*res) - res = _x0 - 0.5*res.
        grid.min_x = _x0 - 0.5 * _res;
        grid.min_y = _y0 - 0.5 * _res;
        grid.min_z = _z0 - 0.5 * _res;
        
        size_t total_voxels = grid.nx * grid.ny * grid.nz;
        int max_dim = std::max({grid.nx, grid.ny, grid.nz});
        
        // Ensure workspace memory
        ws.resize(total_voxels, max_dim);
        
        // Reset temp_grid to default 1 (Air)
        // We only need to reset the part we use, but memset is fast
        // std::fill(ws.temp_grid.begin(), ws.temp_grid.begin() + total_voxels, 1);
        // Actually, we overwrite everything except padding corners?
        // Safer to fill.
        std::fill(ws.temp_grid.begin(), ws.temp_grid.begin() + total_voxels, 1);

        // 1. Fill from VoxelData (Inner part)
        for (uint32_t z = 0; z < _sizeZ; ++z)
        {
            for (uint32_t y = 0; y < _sizeY; ++y)
            {
                for (uint32_t x = 0; x < _sizeX; ++x)
                {
                    uint32_t idx_v = 1 + x + y * _sizeX + z * _sizeX * _sizeY;
                    VoxelData& v = _data[idx_v];
                    
                    // Map to padded grid (x+1, y+1, z+1)
                    int idx = (z + 1) * grid.nx * grid.ny + (y + 1) * grid.nx + (x + 1);
                    
                    // Logic from main.cpp: s is bit 0. 0=occ, 1=free.
                    // VoxelData.s: bit0: sign (0 occ / 1 free)
                    int s = (v.s & 0x01u);
                    ws.temp_grid[idx] = static_cast<int8_t>(s);
                }
            }
        }

        // 2. Padding 'Edge'
        // A. Padding X
        for (int z = 0; z < grid.nz; ++z) {
            for (int y = 0; y < grid.ny; ++y) {
                ws.temp_grid[z * grid.nx * grid.ny + y * grid.nx + 0] = ws.temp_grid[z * grid.nx * grid.ny + y * grid.nx + 1];
                ws.temp_grid[z * grid.nx * grid.ny + y * grid.nx + grid.nx - 1] = ws.temp_grid[z * grid.nx * grid.ny + y * grid.nx + grid.nx - 2];
            }
        }
        // B. Padding Y
        for (int z = 0; z < grid.nz; ++z) {
            for (int x = 0; x < grid.nx; ++x) {
                ws.temp_grid[z * grid.nx * grid.ny + 0 * grid.nx + x] = ws.temp_grid[z * grid.nx * grid.ny + 1 * grid.nx + x];
                ws.temp_grid[z * grid.nx * grid.ny + (grid.ny - 1) * grid.nx + x] = ws.temp_grid[z * grid.nx * grid.ny + (grid.ny - 2) * grid.nx + x];
            }
        }
        // C. Padding Z
        for (int y = 0; y < grid.ny; ++y) {
            for (int x = 0; x < grid.nx; ++x) {
                ws.temp_grid[0 * grid.nx * grid.ny + y * grid.nx + x] = ws.temp_grid[1 * grid.nx * grid.ny + y * grid.nx + x];
                ws.temp_grid[(grid.nz - 1) * grid.nx * grid.ny + y * grid.nx + x] = ws.temp_grid[(grid.nz - 2) * grid.nx * grid.ny + y * grid.nx + x];
            }
        }

        // 3. EDT
        float INF = 1e9;
        
        for (size_t i = 0; i < total_voxels; ++i) {
            ws.dist_to_obstacle[i] = (ws.temp_grid[i] == 0) ? 0.0f : INF;
            ws.dist_to_air[i] = (ws.temp_grid[i] == 1) ? 0.0f : INF;
        }

        dt_3d(ws.dist_to_obstacle, grid.nx, grid.ny, grid.nz, ws);
        dt_3d(ws.dist_to_air, grid.nx, grid.ny, grid.nz, ws);

        // 4. SDF
        // grid.sdf_data is already resized in ws.resize
        for (size_t i = 0; i < total_voxels; ++i) {
            double d_obs = std::sqrt(ws.dist_to_obstacle[i]);
            double d_air = std::sqrt(ws.dist_to_air[i]);

            if (ws.temp_grid[i] == 1) { // Air
                grid.sdf_data[i] = (d_obs - 0.5) * grid.voxel_size;
            } else { // Obstacle
                grid.sdf_data[i] = -(d_air - 0.5) * grid.voxel_size;
            }
        }
        
        return grid;
    }

    // const DenseGrid& toDenseGrid(Workspace& ws) const
    // {
    //     DenseGrid& grid = ws.dense_grid;
    //     // Add padding +2
    //     grid.nx = _sizeX + 2;
    //     grid.ny = _sizeY + 2;
    //     grid.nz = _sizeZ + 2;
    //     grid.voxel_size = _res;
        
    //     // Fix coordinates
    //     grid.min_x = _x0 - 0.5 * _res;
    //     grid.min_y = _y0 - 0.5 * _res;
    //     grid.min_z = _z0 - 0.5 * _res;
        
    //     size_t total_voxels = grid.nx * grid.ny * grid.nz;
    //     int max_dim = std::max({grid.nx, grid.ny, grid.nz});
        
    //     ws.resize(total_voxels, max_dim);
        
    //     // Reset buffers
    //     std::fill(ws.temp_grid.begin(), ws.temp_grid.begin() + total_voxels, 1);
        
    //     // Factor para convertir int8 a unidades de voxel (0..0.5) para el EDT
    //     // El EDT interno trabaja en "unidades de voxel", no metros.
    //     // 254.0f cubre el rango completo (-0.5 a 0.5)
    //     const float offset_to_voxel_unit = 1.0f / 254.0f;

    //     // 1. Fill from VoxelData
    //     for (uint32_t z = 0; z < _sizeZ; ++z)
    //     {
    //         for (uint32_t y = 0; y < _sizeY; ++y)
    //         {
    //             for (uint32_t x = 0; x < _sizeX; ++x)
    //             {
    //                 uint32_t idx_v = 1 + x + y * _sizeX + z * _sizeX * _sizeY;
    //                 VoxelData& v = _data[idx_v];
                    
    //                 int idx = (z + 1) * grid.nx * grid.ny + (y + 1) * grid.nx + (x + 1);
                    
    //                 int s = (v.s & 0x01u);
    //                 ws.temp_grid[idx] = static_cast<int8_t>(s);

    //                 // --- INYECCIÓN DE PRECISIÓN SUB-VÓXEL ---
    //                 if (s == 0) // Si es obstáculo
    //                 {
    //                     // Descomprimimos offset a unidades de voxel
    //                     float dx = static_cast<float>(v.offX) * offset_to_voxel_unit;
    //                     float dy = static_cast<float>(v.offY) * offset_to_voxel_unit;
    //                     float dz = static_cast<float>(v.offZ) * offset_to_voxel_unit;
                        
    //                     // Inicializamos la distancia con el offset cuadrado
    //                     // El algoritmo EDT propagará esto correctamente
    //                     ws.dist_to_obstacle[idx] = (dx*dx + dy*dy + dz*dz);
    //                 }
    //                 else
    //                 {
    //                     ws.dist_to_obstacle[idx] = 1e9f; // INF
    //                 }
    //             }
    //         }
    //     }

    //     // 2. Padding 'Edge' (Copia simple)
    //     // ... (Mantén tus bucles de padding A, B y C aquí, no cambian) ...
    //     for (int z = 0; z < grid.nz; ++z) {
    //         for (int y = 0; y < grid.ny; ++y) {
    //             ws.temp_grid[z * grid.nx * grid.ny + y * grid.nx + 0] = ws.temp_grid[z * grid.nx * grid.ny + y * grid.nx + 1];
    //             ws.temp_grid[z * grid.nx * grid.ny + y * grid.nx + grid.nx - 1] = ws.temp_grid[z * grid.nx * grid.ny + y * grid.nx + grid.nx - 2];
    //             // Nota: No copiamos dist_to_obstacle en el padding porque el EDT lo sobreescribirá
    //             // o se puede inicializar a INF, pero el temp_grid manda.
    //         }
    //     }
    //     for (int z = 0; z < grid.nz; ++z) {
    //         for (int x = 0; x < grid.nx; ++x) {
    //             ws.temp_grid[z * grid.nx * grid.ny + 0 * grid.nx + x] = ws.temp_grid[z * grid.nx * grid.ny + 1 * grid.nx + x];
    //             ws.temp_grid[z * grid.nx * grid.ny + (grid.ny - 1) * grid.nx + x] = ws.temp_grid[z * grid.nx * grid.ny + (grid.ny - 2) * grid.nx + x];
    //         }
    //     }
    //     for (int y = 0; y < grid.ny; ++y) {
    //         for (int x = 0; x < grid.nx; ++x) {
    //             ws.temp_grid[0 * grid.nx * grid.ny + y * grid.nx + x] = ws.temp_grid[1 * grid.nx * grid.ny + y * grid.nx + x];
    //             ws.temp_grid[(grid.nz - 1) * grid.nx * grid.ny + y * grid.nx + x] = ws.temp_grid[(grid.nz - 2) * grid.nx * grid.ny + y * grid.nx + x];
    //         }
    //     }

    //     // 3. Inicializar dist_to_air (Esto sigue igual, binario)
    //     float INF = 1e9f;
    //     for (size_t i = 0; i < total_voxels; ++i) {
    //         // dist_to_obstacle YA ESTÁ INICIALIZADO ARRIBA CON OFFSETS
    //         if (ws.temp_grid[i] == 1) ws.dist_to_obstacle[i] = INF; // Asegurar INF en aire
            
    //         ws.dist_to_air[i] = (ws.temp_grid[i] == 1) ? 0.0f : INF;
    //     }

    //     dt_3d(ws.dist_to_obstacle, grid.nx, grid.ny, grid.nz, ws);
    //     dt_3d(ws.dist_to_air, grid.nx, grid.ny, grid.nz, ws);

    //     // 4. SDF Final
    //     for (size_t i = 0; i < total_voxels; ++i) {
    //         double d_obs = std::sqrt(ws.dist_to_obstacle[i]);
    //         double d_air = std::sqrt(ws.dist_to_air[i]);

    //         if (ws.temp_grid[i] == 1) { // Air
    //             grid.sdf_data[i] = (d_obs - 0.5) * grid.voxel_size;
    //         } else { // Obstacle
    //             grid.sdf_data[i] = -(d_air - 0.5) * grid.voxel_size;
    //         }
    //     }
        
    //     return grid;
    // }

private:
    VoxelData* _data;
    uint32_t _sizeX, _sizeY, _sizeZ;
    float _res;
    float _x0, _y0, _z0;

    static void dt_1d(const std::vector<float> &f, std::vector<float> &d, std::vector<int> &v, std::vector<float> &z, int n)
    {
        int k = 0;
        v[0] = 0;
        z[0] = -1e9;
        z[1] = 1e9;
        for (int q = 1; q < n; q++)
        {
            float s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
            while (s <= z[k])
            {
                k--;
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
            }
            k++;
            v[k] = q;
            z[k] = s;
            z[k + 1] = 1e9;
        }
        k = 0;
        for (int q = 0; q < n; q++)
        {
            while (z[k + 1] < q) k++;
            d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
        }
    }

    static void dt_3d(std::vector<float> &grid, int nx, int ny, int nz, Workspace& ws)
    {
        // Use workspace vectors
        std::vector<float>& f = ws.f;
        std::vector<float>& d = ws.d;
        std::vector<float>& z = ws.z;
        std::vector<int>& v = ws.v;

        // Pass 1: X
        for (int z_i = 0; z_i < nz; ++z_i) {
            for (int y_i = 0; y_i < ny; ++y_i) {
                for (int x_i = 0; x_i < nx; ++x_i)
                    f[x_i] = grid[z_i * nx * ny + y_i * nx + x_i];
                dt_1d(f, d, v, z, nx);
                for (int x_i = 0; x_i < nx; ++x_i)
                    grid[z_i * nx * ny + y_i * nx + x_i] = d[x_i];
            }
        }
        // Pass 2: Y
        for (int z_i = 0; z_i < nz; ++z_i) {
            for (int x_i = 0; x_i < nx; ++x_i) {
                for (int y_i = 0; y_i < ny; ++y_i)
                    f[y_i] = grid[z_i * nx * ny + y_i * nx + x_i];
                dt_1d(f, d, v, z, ny);
                for (int y_i = 0; y_i < ny; ++y_i)
                    grid[z_i * nx * ny + y_i * nx + x_i] = d[y_i];
            }
        }
        // Pass 3: Z
        for (int y_i = 0; y_i < ny; ++y_i) {
            for (int x_i = 0; x_i < nx; ++x_i) {
                for (int z_i = 0; z_i < nz; ++z_i)
                    f[z_i] = grid[z_i * nx * ny + y_i * nx + x_i];
                dt_1d(f, d, v, z, nz);
                for (int z_i = 0; z_i < nz; ++z_i)
                    grid[z_i * nx * ny + y_i * nx + x_i] = d[z_i];
            }
        }
    }
};

#endif
