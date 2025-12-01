#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <tuple>
#include <omp.h>

#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkMarchingCubes.h>
#include <vtkAppendPolyData.h>
#include <vtkSTLWriter.h>

namespace fs = std::filesystem;

#define NUM_GAUSSIANS 10
#define PARAMS_PER_GAUSSIAN 7
#define N_PARAMS (NUM_GAUSSIANS * PARAMS_PER_GAUSSIAN)

#include "solver/solver.hpp"

struct Grid3D
{
    int nx = 0, ny = 0, nz = 0;
    double voxel_size = 0.05;
    double min_x = 0, min_y = 0, min_z = 0;
    std::vector<int8_t> sign_data; // 0: Interior (Obstaculo), 1: Exterior (Aire)
    std::vector<float> sdf_data;   // Negativo: Interior, Positivo: Exterior

    int idx(int x, int y, int z) const
    {
        if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz)
            return -1;
        return z * nx * ny + y * nx + x;
    }
};

struct MetricsResult
{
    double mae;
    double std_dev;
};
struct Peak
{
    int idx;
    double val;
};

void dt_1d(const std::vector<float> &f, std::vector<float> &d, std::vector<int> &v, std::vector<float> &z, int n)
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
        while (z[k + 1] < q)
            k++;
        d[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
    }
}

void dt_3d(std::vector<float> &grid, int nx, int ny, int nz)
{
    std::vector<float> f(std::max({nx, ny, nz})), d(std::max({nx, ny, nz})), z(std::max({nx, ny, nz}) + 1);
    std::vector<int> v(std::max({nx, ny, nz}));

    // Pass 1: X
    for (int z_i = 0; z_i < nz; ++z_i)
    {
        for (int y_i = 0; y_i < ny; ++y_i)
        {
            for (int x_i = 0; x_i < nx; ++x_i)
                f[x_i] = grid[z_i * nx * ny + y_i * nx + x_i];
            dt_1d(f, d, v, z, nx);
            for (int x_i = 0; x_i < nx; ++x_i)
                grid[z_i * nx * ny + y_i * nx + x_i] = d[x_i];
        }
    }
    // Pass 2: Y
    for (int z_i = 0; z_i < nz; ++z_i)
    {
        for (int x_i = 0; x_i < nx; ++x_i)
        {
            for (int y_i = 0; y_i < ny; ++y_i)
                f[y_i] = grid[z_i * nx * ny + y_i * nx + x_i];
            dt_1d(f, d, v, z, ny);
            for (int y_i = 0; y_i < ny; ++y_i)
                grid[z_i * nx * ny + y_i * nx + x_i] = d[y_i];
        }
    }
    // Pass 3: Z
    for (int y_i = 0; y_i < ny; ++y_i)
    {
        for (int x_i = 0; x_i < nx; ++x_i)
        {
            for (int z_i = 0; z_i < nz; ++z_i)
                f[z_i] = grid[z_i * nx * ny + y_i * nx + x_i];
            dt_1d(f, d, v, z, nz);
            for (int z_i = 0; z_i < nz; ++z_i)
                grid[z_i * nx * ny + y_i * nx + x_i] = d[z_i];
        }
    }
}

Grid3D load_perfect_grid(const std::string &filepath)
{
    std::ifstream file(filepath);
    std::string line;
    if (std::getline(file, line))
    {
    }; // Saltar cabecera

    struct RawData
    {
        double x, y, z;
        int s;
    };
    std::vector<RawData> points;

    // 1. LEER TODO Y ENCONTRAR LIMITES FLOTANTES EXACTOS
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();

    double voxel_size = 0.05;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string val;
        std::vector<std::string> row;
        while (std::getline(ss, val, ','))
            row.push_back(val);

        if (row.size() >= 5)
        {
            double x = std::stod(row[0]);
            double y = std::stod(row[1]);
            double z = std::stod(row[2]);
            // CSV: x,y,z,d_manhattan,s,hits -> s es row[4]
            // Aseguramos que sea 0 o 1
            int s = std::stoi(row[4]) & 1;

            points.push_back({x, y, z, s});

            if (x < min_x)
                min_x = x;
            if (x > max_x)
                max_x = x;
            if (y < min_y)
                min_y = y;
            if (y > max_y)
                max_y = y;
            if (z < min_z)
                min_z = z;
            if (z > max_z)
                max_z = z;
        }
    }

    Grid3D grid;
    grid.voxel_size = voxel_size;

    // Calculamos dimensiones basadas en la extensión física
    // Usamos lround para asegurar el entero más cercano
    int real_nx = std::lround((max_x - min_x) / voxel_size) + 1;
    int real_ny = std::lround((max_y - min_y) / voxel_size) + 1;
    int real_nz = std::lround((max_z - min_z) / voxel_size) + 1;

    // Dimensiones finales con Padding (+2 en total)
    grid.nx = real_nx + 2;
    grid.ny = real_ny + 2;
    grid.nz = real_nz + 2;

    // El origen del grid se desplaza 1 voxel hacia atrás por el padding
    grid.min_x = min_x - voxel_size;
    grid.min_y = min_y - voxel_size;
    grid.min_z = min_z - voxel_size;

    size_t total_voxels = (size_t)grid.nx * grid.ny * grid.nz;

    // Inicializamos a 1 (Aire). Como el CSV es denso, esto solo quedará
    // en el padding hasta que hagamos el paso de copiar bordes.
    std::vector<int8_t> temp_grid(total_voxels, 1);

    // 2. RELLENADO DIRECTO (SIN HUECOS)
    // Usamos coordenadas relativas para evitar errores de redondeo
    for (const auto &p : points)
    {
        // Indice relativo al mínimo + 1 por el padding izquierdo
        int ix = std::lround((p.x - min_x) / voxel_size) + 1;
        int iy = std::lround((p.y - min_y) / voxel_size) + 1;
        int iz = std::lround((p.z - min_z) / voxel_size) + 1;

        // Asignación directa
        if (ix >= 0 && ix < grid.nx &&
            iy >= 0 && iy < grid.ny &&
            iz >= 0 && iz < grid.nz)
        {
            temp_grid[grid.idx(ix, iy, iz)] = static_cast<int8_t>(p.s);
        }
    }

    // 3. PADDING 'EDGE' (Imitar np.pad mode='edge')
    // Copiamos la última capa válida de datos a la capa de padding vacía

    // A. Padding en X
    for (int z = 0; z < grid.nz; ++z)
    {
        for (int y = 0; y < grid.ny; ++y)
        {
            temp_grid[grid.idx(0, y, z)] = temp_grid[grid.idx(1, y, z)];                     // Copia capa 1 a 0
            temp_grid[grid.idx(grid.nx - 1, y, z)] = temp_grid[grid.idx(grid.nx - 2, y, z)]; // Copia penúltima a última
        }
    }
    // B. Padding en Y
    for (int z = 0; z < grid.nz; ++z)
    {
        for (int x = 0; x < grid.nx; ++x)
        {
            temp_grid[grid.idx(x, 0, z)] = temp_grid[grid.idx(x, 1, z)];
            temp_grid[grid.idx(x, grid.ny - 1, z)] = temp_grid[grid.idx(x, grid.ny - 2, z)];
        }
    }
    // C. Padding en Z
    for (int y = 0; y < grid.ny; ++y)
    {
        for (int x = 0; x < grid.nx; ++x)
        {
            temp_grid[grid.idx(x, y, 0)] = temp_grid[grid.idx(x, y, 1)];
            temp_grid[grid.idx(x, y, grid.nz - 1)] = temp_grid[grid.idx(x, y, grid.nz - 2)];
        }
    }

    // 4. TRANSFORMADA DE DISTANCIA (EDT)
    float INF = 1e9;
    std::vector<float> dist_to_obstacle(total_voxels);
    std::vector<float> dist_to_air(total_voxels);

    for (size_t i = 0; i < total_voxels; ++i)
    {
        dist_to_obstacle[i] = (temp_grid[i] == 0) ? 0.0f : INF; // 0 es obstáculo
        dist_to_air[i] = (temp_grid[i] == 1) ? 0.0f : INF;      // 1 es aire
    }

    dt_3d(dist_to_obstacle, grid.nx, grid.ny, grid.nz);
    dt_3d(dist_to_air, grid.nx, grid.ny, grid.nz);

    // 5. CALCULAR SDF FINAL
    grid.sign_data = temp_grid;
    grid.sdf_data.resize(total_voxels);

    for (size_t i = 0; i < total_voxels; ++i)
    {
        double d_obs = std::sqrt(dist_to_obstacle[i]);
        double d_air = std::sqrt(dist_to_air[i]);

        if (temp_grid[i] == 1)
        { // AIRE
            // Positivo: Distancia al obstáculo - 0.5 voxel
            grid.sdf_data[i] = (d_obs - 0.5) * grid.voxel_size;
        }
        else
        { // OBSTÁCULO
            // Negativo: Distancia al aire - 0.5 voxel
            grid.sdf_data[i] = -(d_air - 0.5) * grid.voxel_size;
        }
    }

    return grid;
}

double get_sdf_trilinear(const Grid3D &grid, double x, double y, double z)
{
    double fx = (x - grid.min_x) / grid.voxel_size;
    double fy = (y - grid.min_y) / grid.voxel_size;
    double fz = (z - grid.min_z) / grid.voxel_size;

    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));
    int z0 = static_cast<int>(std::floor(fz));

    if (x0 < 0)
        x0 = 0;
    if (x0 >= grid.nx - 1)
        x0 = grid.nx - 2;
    if (y0 < 0)
        y0 = 0;
    if (y0 >= grid.ny - 1)
        y0 = grid.ny - 2;
    if (z0 < 0)
        z0 = 0;
    if (z0 >= grid.nz - 1)
        z0 = grid.nz - 2;

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;
    double xd = fx - x0;
    double yd = fy - y0;
    double zd = fz - z0;

    // idx check simplificado por velocidad (asumimos bounds correctos por clamping arriba)
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

struct Coord
{
    int x, y, z;
};

Coord idx_to_coord(int idx, int nx, int ny)
{
    int z = idx / (nx * ny);
    int rem = idx % (nx * ny);
    int y = rem / nx;
    int x = rem % nx;
    return {x, y, z};
}

bool is_local_extremum(const Grid3D &grid, int x, int y, int z, bool find_max, int radius)
{
    double center_val = grid.sdf_data[grid.idx(x, y, z)];

    // Iteramos en la ventana [-radius, +radius]
    for (int dz = -radius; dz <= radius; ++dz)
    {
        int nz = z + dz;
        if (nz < 0 || nz >= grid.nz)
            continue;

        for (int dy = -radius; dy <= radius; ++dy)
        {
            int ny = y + dy;
            if (ny < 0 || ny >= grid.ny)
                continue;

            for (int dx = -radius; dx <= radius; ++dx)
            {
                if (dx == 0 && dy == 0 && dz == 0)
                    continue; // No compararse a sí mismo

                int nx = x + dx;
                if (nx < 0 || nx >= grid.nx)
                    continue;

                double val = grid.sdf_data[grid.idx(nx, ny, nz)];

                if (find_max)
                {
                    if (val > center_val)
                        return false; // Hay uno mayor cerca, no soy máximo
                }
                else
                {
                    if (val < center_val)
                        return false; // Hay uno menor cerca, no soy mínimo
                }
            }
        }
    }
    return true;
}

std::vector<double> generate_initial_params(const Grid3D &grid)
{
    std::vector<double> x_params(N_PARAMS);
    std::mt19937 rng(42);

    // --- CONFIGURACIÓN (Idéntica a Python) ---
    double init_log_sigma = std::sqrt(0.08); // Sigma ~8cm
    double w_init = 0.1;

    int peak_radius = 3;            // Equivalente a filter_size=7 en Python
    int suppression_radius = 3;     // Radio de "espacio personal" tras elegir un pico
    int min_gaussians_per_sign = 6; // Mínimo asegurado por signo

    // --- 1. DETECCIÓN DE CANDIDATOS (Find Peaks) ---
    std::vector<Peak> pos_candidates;
    std::vector<Peak> neg_candidates;

    // Evitamos bordes extremos
    for (int z = peak_radius; z < grid.nz - peak_radius; ++z)
    {
        for (int y = peak_radius; y < grid.ny - peak_radius; ++y)
        {
            for (int x_idx = peak_radius; x_idx < grid.nx - peak_radius; ++x_idx)
            {
                int idx = grid.idx(x_idx, y, z);
                double val = grid.sdf_data[idx];

                // Umbral pequeño para evitar ruido cerca de 0
                if (val > 1e-3)
                {
                    if (is_local_extremum(grid, x_idx, y, z, true, peak_radius))
                        pos_candidates.push_back({idx, val});
                }
                else if (val < -1e-3)
                {
                    if (is_local_extremum(grid, x_idx, y, z, false, peak_radius))
                        neg_candidates.push_back({idx, val});
                }
            }
        }
    }

    // Ordenar por magnitud (Fuerza del pico)
    // Positivos: mayor a menor
    std::sort(pos_candidates.begin(), pos_candidates.end(), [](const Peak &a, const Peak &b)
              { return a.val > b.val; });
    // Negativos: menor a mayor (más negativo primero)
    std::sort(neg_candidates.begin(), neg_candidates.end(), [](const Peak &a, const Peak &b)
              { return a.val < b.val; });

    // --- 2. SELECCIÓN CON SUPRESIÓN (NMS) ---

    auto run_nms = [&](const std::vector<Peak> &candidates, int target_count) -> std::vector<int>
    {
        std::vector<int> selected_indices;
        std::vector<bool> suppressed(grid.nx * grid.ny * grid.nz, false);

        for (const auto &p : candidates)
        {
            if (selected_indices.size() >= target_count)
                break;
            if (suppressed[p.idx])
                continue; // Si ya está cubierto por otro pico, saltar

            selected_indices.push_back(p.idx);

            // Aplicar supresión alrededor de este pico
            Coord c = idx_to_coord(p.idx, grid.nx, grid.ny);
            int r = suppression_radius;

            // Marcamos la caja alrededor como suprimida
            for (int dz = -r; dz <= r; ++dz)
            {
                int nz = c.z + dz;
                if (nz < 0 || nz >= grid.nz)
                    continue;
                for (int dy = -r; dy <= r; ++dy)
                {
                    int ny = c.y + dy;
                    if (ny < 0 || ny >= grid.ny)
                        continue;
                    for (int dx = -r; dx <= r; ++dx)
                    {
                        int nx = c.x + dx;
                        if (nx < 0 || nx >= grid.nx)
                            continue;
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

    // --- 3. GENERACIÓN DE PARÁMETROS (Con Fallback aleatorio) ---

    int current_g = 0;

    // Lambda para escribir parámetros en el vector plano
    auto write_gaussian = [&](int idx_grid, double weight_val)
    {
        if (current_g >= NUM_GAUSSIANS)
            return;

        Coord c = idx_to_coord(idx_grid, grid.nx, grid.ny);
        // Centrar en el voxel (+0.5) + pequeño ruido opcional para romper simetría
        std::uniform_real_distribution<double> jitter(-0.1, 0.1);

        double px = grid.min_x + (c.x + 0.5 + jitter(rng)) * grid.voxel_size;
        double py = grid.min_y + (c.y + 0.5 + jitter(rng)) * grid.voxel_size;
        double pz = grid.min_z + (c.z + 0.5 + jitter(rng)) * grid.voxel_size;

        int base = current_g * PARAMS_PER_GAUSSIAN;
        // x_params[base + 0] = px;
        // x_params[base + 1] = py;
        // x_params[base + 2] = pz;
        // x_params[base + 3] = init_log_sigma; // l00
        // x_params[base + 4] = 0.0;            // l10
        // x_params[base + 5] = init_log_sigma; // l11
        // x_params[base + 6] = 0.0;            // l20
        // x_params[base + 7] = 0.0;            // l21
        // x_params[base + 8] = init_log_sigma; // l22
        // x_params[base + 9] = weight_val;

        x_params[base + 0] = px;
        x_params[base + 1] = py;
        x_params[base + 2] = pz;
        x_params[base + 3] = init_log_sigma; 
        x_params[base + 4] = init_log_sigma; 
        x_params[base + 5] = init_log_sigma; 
        
        // Peso
        x_params[base + 6] = weight_val;

        current_g++;
    };

    // Rellenar Positivas
    for (int idx : final_pos_indices)
        write_gaussian(idx, w_init);

    // Fallback Positivo (si NMS encontró pocas)
    std::vector<int> all_pos_indices; // Cache perezosa
    while (current_g < min_gaussians_per_sign || (current_g < n_pos_target))
    {
        // Si es la primera vez que entramos, buscamos todos los indices positivos
        if (all_pos_indices.empty())
        {
            for (int i = 0; i < grid.sdf_data.size(); ++i)
                if (grid.sdf_data[i] > 0)
                    all_pos_indices.push_back(i);
        }
        if (all_pos_indices.empty())
            break; // No hay nada positivo en el grid

        std::uniform_int_distribution<int> dist(0, all_pos_indices.size() - 1);
        write_gaussian(all_pos_indices[dist(rng)], w_init);
    }

    // Guardamos cuantos positivos llevamos para saber el límite de negativos
    int total_pos_filled = current_g;

    // Rellenar Negativas
    for (int idx : final_neg_indices)
        write_gaussian(idx, -w_init);

    // Fallback Negativo
    std::vector<int> all_neg_indices;
    while (current_g < NUM_GAUSSIANS)
    {
        if (all_neg_indices.empty())
        {
            for (int i = 0; i < grid.sdf_data.size(); ++i)
                if (grid.sdf_data[i] < 0)
                    all_neg_indices.push_back(i);
        }
        if (all_neg_indices.empty())
            break;

        std::uniform_int_distribution<int> dist(0, all_neg_indices.size() - 1);
        write_gaussian(all_neg_indices[dist(rng)], -w_init);
    }

    return x_params;
}

std::vector<GMMData> sample_points_for_gd(const Grid3D &grid, int n_total, double band_frac, double band_tau)
{
    int n_band = static_cast<int>(n_total * band_frac);
    int n_uniform = n_total - n_band;
    std::vector<GMMData> data;
    data.reserve(n_total);
    std::mt19937 rng(42);

    double max_x = grid.min_x + (grid.nx - 1) * grid.voxel_size;
    double max_y = grid.min_y + (grid.ny - 1) * grid.voxel_size;
    double max_z = grid.min_z + (grid.nz - 1) * grid.voxel_size;

    std::uniform_real_distribution<double> dist_x(grid.min_x, max_x);
    std::uniform_real_distribution<double> dist_y(grid.min_y, max_y);
    std::uniform_real_distribution<double> dist_z(grid.min_z, max_z);

    // Uniformes
    for (int i = 0; i < n_uniform; ++i)
    {
        double x = dist_x(rng);
        double y = dist_y(rng);
        double z = dist_z(rng);
        data.push_back({x, y, z, get_sdf_trilinear(grid, x, y, z)});
    }

    // Banda estrecha (cerca de SDF=0)
    std::vector<int> band_idxs;
    for (size_t i = 0; i < grid.sdf_data.size(); ++i)
        if (std::abs(grid.sdf_data[i]) < band_tau)
            band_idxs.push_back(i);

    std::uniform_real_distribution<double> jitter(-0.5, 0.5);
    if (!band_idxs.empty())
    {
        std::uniform_int_distribution<int> dist_b(0, band_idxs.size() - 1);
        for (int i = 0; i < n_band; ++i)
        {
            int idx = band_idxs[dist_b(rng)];
            int iz = idx / (grid.nx * grid.ny);
            int rem = idx % (grid.nx * grid.ny);
            int iy = rem / grid.nx;
            int ix = rem % grid.nx;

            double x = grid.min_x + (ix + jitter(rng)) * grid.voxel_size;
            double y = grid.min_y + (iy + jitter(rng)) * grid.voxel_size;
            double z = grid.min_z + (iz + jitter(rng)) * grid.voxel_size;

            // Clamp
            x = std::max(grid.min_x, std::min(x, max_x));
            y = std::max(grid.min_y, std::min(y, max_y));
            z = std::max(grid.min_z, std::min(z, max_z));

            data.push_back({x, y, z, get_sdf_trilinear(grid, x, y, z)});
        }
    }
    else
    {
        for (int i = 0; i < n_band; ++i)
        {
            double x = dist_x(rng);
            double y = dist_y(rng);
            double z = dist_z(rng);
            data.push_back({x, y, z, get_sdf_trilinear(grid, x, y, z)});
        }
    }
    return data;
}


// double predict_sdf(double x, double y, double z, const std::vector<double> &params)
// {
//     double sdf_pred = 0.0;
//     double eps = 1e-6; // Epsilon para estabilidad

//     for (int i = 0; i < NUM_GAUSSIANS; ++i)
//     {
//         int j = i * PARAMS_PER_GAUSSIAN;

//         // --- LEER PARÁMETROS (Sin exp, usando cuadrado para diagonales) ---
//         double mx = params[j++];
//         double my = params[j++];
//         double mz = params[j++];

//         double p_l00 = params[j++];
//         double l00 = p_l00 * p_l00 + eps;
//         double l10 = params[j++];
//         double p_l11 = params[j++];
//         double l11 = p_l11 * p_l11 + eps;
//         double l20 = params[j++];
//         double l21 = params[j++];
//         double p_l22 = params[j++];
//         double l22 = p_l22 * p_l22 + eps;

//         double w = params[j++];

//         // --- CÁLCULO MAHALANOBIS ---
//         double vx = x - mx;
//         double vy = y - my;
//         double vz = z - mz;

//         // Resolver L*z = v (Triangular inferior)
//         double z0 = vx / l00;
//         double z1 = (vy - l10 * z0) / l11;
//         double z2 = (vz - l20 * z0 - l21 * z1) / l22;

//         double dist_sq = z0 * z0 + z1 * z1 + z2 * z2;

//         sdf_pred += w * std::exp(-0.5 * dist_sq);
//     }
//     return sdf_pred;
// }

double predict_sdf(double x, double y, double z, const std::vector<double> &params)
{
    double sdf_pred = 0.0;
    double eps = 1e-6;

    for (int i = 0; i < NUM_GAUSSIANS; ++i)
    {
        int j = i * PARAMS_PER_GAUSSIAN;

        // Leer
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

        // Calculo Simplificado (Diagonal)
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

MetricsResult calculate_metrics(const Grid3D &grid, const std::vector<double> &params)
{
    std::vector<double> errors;
    // Reservamos memoria, aunque probablemente usaremos menos
    errors.reserve(grid.nx * grid.ny * grid.nz);

    // --- CONFIGURACIÓN DEL UMBRAL ---
    const double ERROR_THRESHOLD = 1.0; // 1 metro

    for (int z = 0; z < grid.nz; ++z)
    {
        for (int y = 0; y < grid.ny; ++y)
        {
            for (int x = 0; x < grid.nx; ++x)
            {
                double wx = grid.min_x + x * grid.voxel_size;
                double wy = grid.min_y + y * grid.voxel_size;
                double wz = grid.min_z + z * grid.voxel_size;
                
                double real = grid.sdf_data[grid.idx(x, y, z)];
                double pred = predict_sdf(wx, wy, wz, params);
                
                double err = std::abs(real - pred);

                // --- FILTRADO DE OUTLIERS ---
                // Solo guardamos el error si es menor o igual a 1 metro.
                // Si es mayor, lo ignoramos completamente para la media.
                if (err <= ERROR_THRESHOLD) 
                {
                    errors.push_back(err);
                }
            }
        }
    }

    // Si todos los puntos superan el metro de error (caso muy raro), devolvemos 0 o un indicador
    if (errors.empty())
        return {0.0, 0.0};

    double sum = 0.0;
    for (double e : errors)
        sum += e;
    
    // Ahora la media se divide solo por el número de puntos "buenos"
    double mean = sum / errors.size();

    double var = 0.0;
    for (double e : errors)
        var += (e - mean) * (e - mean);

    return {mean, std::sqrt(var / errors.size())};
}

vtkSmartPointer<vtkPolyData> generate_mesh_from_gaussians(const std::vector<double> &params, const Grid3D &grid)
{
    const int UPSAMPLE = 1;
    double spacing = grid.voxel_size / UPSAMPLE;
    int dimX = (grid.nx - 1) * UPSAMPLE + 1;
    int dimY = (grid.ny - 1) * UPSAMPLE + 1;
    int dimZ = (grid.nz - 1) * UPSAMPLE + 1;

    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(dimX, dimY, dimZ);
    imageData->SetSpacing(spacing, spacing, spacing);
    imageData->SetOrigin(grid.min_x, grid.min_y, grid.min_z);
    imageData->AllocateScalars(VTK_FLOAT, 1);
    float *scalars = static_cast<float *>(imageData->GetScalarPointer());

    for (int z = 0; z < dimZ; ++z)
    {
        for (int y = 0; y < dimY; ++y)
        {
            for (int x = 0; x < dimX; ++x)
            {
                double wx = grid.min_x + x * spacing;
                double wy = grid.min_y + y * spacing;
                double wz = grid.min_z + z * spacing;
                double val = predict_sdf(wx, wy, wz, params);
                scalars[z * dimY * dimX + y * dimX + x] = static_cast<float>(val);
            }
        }
    }
    vtkSmartPointer<vtkMarchingCubes> mc = vtkSmartPointer<vtkMarchingCubes>::New();
    mc->SetInputData(imageData);
    mc->SetValue(0, 0.0); // Iso-surface 0
    mc->Update();
    return mc->GetOutput();
}

void print_debug_slice(const Grid3D &grid, int z_slice)
{
    if (z_slice < 0 || z_slice >= grid.nz)
    {
        std::cerr << "Slice Z fuera de rango.\n";
        return;
    }

    std::cout << "\n--- DEBUG SDF SLICE Z=" << z_slice << " (" << grid.nx << "x" << grid.ny << ") ---\n";
    std::cout << "Leyenda: " << "\033[1;31m" << "Negativo (Interior)" << "\033[0m"
              << " | " << "\033[1;37m" << "Positivo (Aire)" << "\033[0m\n\n";

    for (int y = 0; y < grid.ny; ++y)
    {
        std::cout << std::setw(3) << y << " | "; // Indice Y
        for (int x = 0; x < grid.nx; ++x)
        {
            float val = grid.sdf_data[grid.idx(x, y, z_slice)];

            // COLORES ANSI
            if (val < 0)
            {
                // ROJO para interior (obstáculo)
                std::cout << "\033[1;31m";
            }
            else if (val == 0)
            {
                // VERDE para superficie exacta
                std::cout << "\033[1;32m";
            }
            else
            {
                // GRIS/BLANCO para aire
                // Si está muy cerca de 0, lo pintamos brillante, si no, oscuro
                if (val < 0.1)
                    std::cout << "\033[1;37m"; // Blanco brillante
                else
                    std::cout << "\033[0;37m"; // Gris
            }

            // Imprimir valor formateado
            // Usamos precision baja para que quepa en pantalla
            printf("%5.2f ", val);

            std::cout << "\033[0m"; // Reset color
        }
        std::cout << "\n";
    }
    std::cout << "------------------------------------------\n\n";
}

int main(int argc, char **argv)
{
    std::string folder_path = "/home/ros/ros2_ws/tests_results/csv/";
    std::string output_mesh_file = "combined_mesh.stl";

    // --- 1. Obtener Archivos ---
    std::vector<std::string> csv_files;
    try
    {
        for (const auto &entry : fs::directory_iterator(folder_path))
            if (entry.path().extension() == ".csv")
                csv_files.push_back(entry.path().string());
        std::sort(csv_files.begin(), csv_files.end());
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }

    if (csv_files.empty())
        return 1;

    vtkSmartPointer<vtkAppendPolyData> appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
    double sum_mae = 0.0;
    int processed_count = 0;

    // --- COLORES ANSI ---
    const std::string C_RESET = "\033[0m";
    const std::string C_GREEN = "\033[32m";  // MAE Bajo
    const std::string C_YELLOW = "\033[33m"; // MAE Medio
    const std::string C_RED = "\033[31m";    // MAE Alto / Error
    const std::string C_CYAN = "\033[36m";   // Info

    std::cout << "\n"
              << C_CYAN << "=== INICIANDO PROCESAMIENTO DE " << csv_files.size() << " CUBOS ===" << C_RESET << "\n\n";

    auto t_global_start = std::chrono::high_resolution_clock::now();
 
    // --- PARALELISMO ENTRE CUBOS (Coarse-Grained) ---
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < csv_files.size(); ++i)
    {
        // 1. VARIABLES LOCALES DEL HILO (Sin cambios en la lógica, pero se ejecutan en paralelo)
        auto t_start = std::chrono::high_resolution_clock::now();

        Grid3D grid = load_perfect_grid(csv_files[i]);
        auto t_load = std::chrono::high_resolution_clock::now();

        std::vector<double> params = generate_initial_params(grid);
        auto t_init = std::chrono::high_resolution_clock::now();

        std::vector<GMMData> data = sample_points_for_gd(grid, 500, 0.0, 0.0);
        auto t_sample = std::chrono::high_resolution_clock::now();

        // IMPORTANTE: GMMSolver debe tener num_threads = 1 internamente
        GMMSolver solver;
        // solver.setMaxNumIterations(30); 
        solver.solve(data, params);
        auto t_opt = std::chrono::high_resolution_clock::now();

        MetricsResult res = calculate_metrics(grid, params);

        // Generar malla localmente (esto consume CPU, así que es bueno que esté paralelo)
        vtkSmartPointer<vtkPolyData> mesh = generate_mesh_from_gaussians(params, grid);
        auto t_mesh = std::chrono::high_resolution_clock::now();

        auto d = [&](auto a, auto b) { return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count(); };
        long dur_total = d(t_start, t_mesh);
        long dur_load = d(t_start, t_load);
        long dur_init = d(t_load, t_init);
        long dur_sample = d(t_init, t_sample);
        long dur_opt = d(t_sample, t_opt);
        long dur_mesh = d(t_opt, t_mesh);

        // 2. SECCIÓN CRÍTICA (Escritura segura en variables compartidas y consola)
        #pragma omp critical
        {
            // Acumular resultados globales
            if (mesh->GetNumberOfPoints() > 0)
            {
                appendFilter->AddInputData(mesh);
                processed_count++;
            }
            sum_mae += res.mae;

            // Imprimir log (para que no se mezclen las letras de distintos hilos)
            std::string color = C_GREEN;
            if (res.mae > 0.05) color = C_RED;
            else if (res.mae > 0.02) color = C_YELLOW;

            std::string fname = fs::path(csv_files[i]).filename().string();

            std::cout << "[" << std::setw(3) << (i + 1) << "/" << csv_files.size() << "] "
                      << std::left << std::setw(20) << fname << std::right << " | "
                      << color << "MAE: " << std::fixed << std::setprecision(4) << res.mae << "m" << C_RESET << " | "
                      << "T: " << std::setw(3) << dur_total << "ms "
                      << "(L:" << dur_load 
                      << " I:" << dur_init 
                      << " S:" << dur_sample  
                      << " O:" << dur_opt     
                      << " M:" << dur_mesh << ")"
                      << std::endl;
        }
    }

    auto t_global_end = std::chrono::high_resolution_clock::now();
    
    // Cálculo de tiempos
    long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_global_end - t_global_start).count();
    double total_seconds = total_ms / 1000.0; 
    
    double avg_ms_per_cube = (csv_files.size() > 0) ? (double)total_ms / csv_files.size() : 0.0;
    double avg_mae = (processed_count > 0) ? (sum_mae / processed_count) : 0.0;
    double cubes_per_sec = (total_seconds > 0) ? (csv_files.size() / total_seconds) : 0.0;

    std::cout << "\n" << C_CYAN << "=== GUARDANDO MALLA FINAL ===" << C_RESET << "\n";
    
    if (processed_count > 0)
    {
        appendFilter->Update();
        vtkSmartPointer<vtkSTLWriter> writer = vtkSmartPointer<vtkSTLWriter>::New();
        writer->SetFileName(output_mesh_file.c_str());
        writer->SetInputData(appendFilter->GetOutput());
        writer->SetFileTypeToBinary();
        writer->Write();
        
        std::cout << " > Guardado en:      " << output_mesh_file << "\n";
        std::cout << " > -----------------------------------\n";
        std::cout << " > Cubos Procesados: " << processed_count << "/" << csv_files.size() << "\n";
        std::cout << " > MAE Promedio:     " << std::fixed << std::setprecision(4) << avg_mae << " m\n";
        std::cout << " > Tiempo TOTAL:     " << std::fixed << std::setprecision(2) << total_seconds << " s\n";
        std::cout << " > Tiempo MEDIO:     " << std::setprecision(1) << avg_ms_per_cube << " ms/cubo\n";
        std::cout << " > Velocidad:        " << std::setprecision(2) << cubes_per_sec << " cubos/s\n";
        std::cout << " > -----------------------------------\n";
    }
    else
    {
        std::cout << C_RED << " > Error: No se generó geometría válida." << C_RESET << "\n";
    }

    return 0;
}