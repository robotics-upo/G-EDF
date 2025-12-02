#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <tuple>
#include <limits> 
#include <omp.h>

// VTK Includes
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkMarchingCubes.h>
#include <vtkAppendPolyData.h>
#include <vtkSTLWriter.h>
#include <vtkPolyData.h>

// RESOLUCIÓN (Metros)
const float TARGET_RESOLUTION = 0.05f; 

// ZONA DE INTERÉS (CROP)
const float MIN_X = -99099.0f; 
const float MAX_X =  99099.0f; 

const float MIN_Y = -99999.0f;
const float MAX_Y =  99909.0f;

const float MIN_Z = -99999.0f; 
const float MAX_Z =  99999.0f; 

const float BLOCK_SIZE = 1.0f; 

struct GaussianRaw {
    double rel_mx, rel_my, rel_mz;
    double p_sx, p_sy, p_sz; 
    double weight;
};

struct CubeData {
    double x, y, z; 
    std::vector<GaussianRaw> gaussians;
};

using CubeKey = std::tuple<int, int, int>;

double predict_sdf_exact(double x, double y, double z, const CubeData& cube) {
    double sdf_pred = 0.0;
    double eps = 1e-6;

    for (const auto& g : cube.gaussians) {
        double mx = cube.x + g.rel_mx;
        double my = cube.y + g.rel_my;
        double mz = cube.z + g.rel_mz;

        double l00 = g.p_sx * g.p_sx + eps;
        double l11 = g.p_sy * g.p_sy + eps;
        double l22 = g.p_sz * g.p_sz + eps;

        double w = g.weight;

        double vx = x - mx;
        double vy = y - my;
        double vz = z - mz;

        double z0 = vx / l00;
        double z1 = vy / l11;
        double z2 = vz / l22;

        double dist_sq = z0 * z0 + z1 * z1 + z2 * z2;
        
        sdf_pred += w * std::exp(-0.5 * dist_sq);
    }
    return sdf_pred;
}

int main(int argc, char** argv) {
    std::string csv_path = "/home/ros/ros2_ws/gaussian_mesh.csv";
    std::string out_path = "/home/ros/ros2_ws/reconstructed_mesh.stl";
    
    const int NX = static_cast<int>(std::round(BLOCK_SIZE / TARGET_RESOLUTION));
    const int NY = NX; 
    const int NZ = NX;
    
    const int DIM_X = NX + 1;
    const int DIM_Y = NY + 1;
    const int DIM_Z = NZ + 1;

    const double SPACING = TARGET_RESOLUTION;
    const double ORIGIN_OFFSET = -0.5 * TARGET_RESOLUTION;

    std::cout << "=== RECONSTRUCCION CON RESOLUCION PERSONALIZADA ===" << std::endl;
    std::cout << "Resolucion: " << TARGET_RESOLUTION << " m" << std::endl;
    std::cout << "Grid por Cubo: " << DIM_X << "x" << DIM_Y << "x" << DIM_Z << " puntos" << std::endl;
    std::cout << "Zona X: [" << MIN_X << ", " << MAX_X << "]" << std::endl;

    std::map<CubeKey, CubeData> grid_map;
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir " << csv_path << std::endl;
        return 1;
    }

    std::string line;
    std::getline(file, line); 

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> row;
        while (std::getline(ss, token, ',')) {
            row.push_back(token);
        }
        if (row.size() < 13) continue;

        float cx = std::stof(row[1]); 
        float cy = std::stof(row[2]); 
        float cz = std::stof(row[3]); 

        // Filtro de Zona
        if (cx < MIN_X || cx > MAX_X) continue;
        if (cy < MIN_Y || cy > MAX_Y) continue;
        if (cz < MIN_Z || cz > MAX_Z) continue;

        GaussianRaw g;
        g.rel_mx = std::stod(row[6]); 
        g.rel_my = std::stod(row[7]);
        g.rel_mz = std::stod(row[8]);
        g.p_sx   = std::stod(row[9]); 
        g.p_sy   = std::stod(row[10]);
        g.p_sz   = std::stod(row[11]);
        g.weight = std::stod(row[12]); 

        CubeKey key = std::make_tuple((int)cx, (int)cy, (int)cz);
        grid_map[key].x = cx;
        grid_map[key].y = cy;
        grid_map[key].z = cz;
        grid_map[key].gaussians.push_back(g);
    }
    file.close();
    
    std::cout << "Cubos a procesar: " << grid_map.size() << std::endl;

    if (grid_map.empty()) {
        std::cerr << "AVISO: Zona vacia." << std::endl;
        return 0;
    }

    std::vector<CubeData> cubes;
    cubes.reserve(grid_map.size());
    for(const auto& pair : grid_map) cubes.push_back(pair.second);

    auto global_appender = vtkSmartPointer<vtkAppendPolyData>::New();
    std::vector<vtkSmartPointer<vtkPolyData>> thread_meshes(cubes.size());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < cubes.size(); ++i) {
        const auto& cube = cubes[i];

        auto image = vtkSmartPointer<vtkImageData>::New();
        image->SetDimensions(DIM_X, DIM_Y, DIM_Z); 
        image->SetSpacing(SPACING, SPACING, SPACING);
        image->SetOrigin(cube.x + ORIGIN_OFFSET, 
                         cube.y + ORIGIN_OFFSET, 
                         cube.z + ORIGIN_OFFSET);
        
        image->AllocateScalars(VTK_FLOAT, 1);
        float* scalar_ptr = static_cast<float*>(image->GetScalarPointer());

        for (int z = 0; z < DIM_Z; ++z) {
            for (int y = 0; y < DIM_Y; ++y) {
                for (int x = 0; x < DIM_X; ++x) {
                    double wx = (cube.x + ORIGIN_OFFSET) + x * SPACING;
                    double wy = (cube.y + ORIGIN_OFFSET) + y * SPACING;
                    double wz = (cube.z + ORIGIN_OFFSET) + z * SPACING;

                    double val = predict_sdf_exact(wx, wy, wz, cube);
                    scalar_ptr[z * DIM_Y * DIM_X + y * DIM_X + x] = static_cast<float>(val);
                }
            }
        }

        auto mc = vtkSmartPointer<vtkMarchingCubes>::New();
        mc->SetInputData(image);
        mc->SetValue(0, 0.0); 
        mc->ComputeNormalsOn();
        mc->Update();

        if (mc->GetOutput()->GetNumberOfPoints() > 0) {
            auto poly = vtkSmartPointer<vtkPolyData>::New();
            poly->DeepCopy(mc->GetOutput());
            thread_meshes[i] = poly;
        }
    }

    std::cout << "Uniendo..." << std::endl;
    for (const auto& mesh : thread_meshes) {
        if (mesh) global_appender->AddInputData(mesh);
    }
    
    if (global_appender->GetNumberOfInputConnections(0) == 0) {
         std::cout << "Sin geometria resultante." << std::endl;
         return 0;
    }

    global_appender->Update();

    auto writer = vtkSmartPointer<vtkSTLWriter>::New();
    writer->SetFileName(out_path.c_str());
    writer->SetInputData(global_appender->GetOutput());
    writer->SetFileTypeToBinary();
    writer->Write();

    std::cout << "Archivo guardado: " << out_path << std::endl;
    return 0;
}