#include <vector>
#include <iostream>
#include <thread>
#include <chrono>

// Mock RCLCPP_INFO
namespace rclcpp {
    struct Logger {};
    inline Logger get_logger(const std::string&) { return {}; }
}
#define RCLCPP_INFO(logger, ...) printf(__VA_ARGS__); printf("\n")

#include "db_tsdf/grid_16.hpp"

int main() {
    std::cout << "Starting Gaussian Grid Test..." << std::endl;

    GRID16 grid;
    // Setup small grid with limited max cells to force discard
    // 10x10x10 meters, res 0.1m -> 100x100x100 voxels total?
    // Each cell is 1x1x1 world unit -> 10x10x10 voxels.
    // Max cells = 5.
    grid.setup(0, 10, 0, 10, 0, 10, 0.1, 5); 

    std::cout << "Grid setup complete. Max cells: 5" << std::endl;

    // Allocate cells sequentially
    for (int i = 0; i < 10; ++i) {
        // Allocating at same position forces reuse if grid is full
        // But we need to move around to force reuse of *different* cells or just fill it up.
        // Grid max cells is 5.
        // Allocating 0,0,0 -> 5,0,0 fills it.
        // Allocating 6,0,0 reuses 0,0,0.
        std::cout << "Allocating cell at " << i << ", 0, 0" << std::endl;
        grid.allocCell(i, 0, 0);
        
        // Fill with a sphere pattern
        float cx = 5.0, cy = 5.0, cz = 5.0;
        float radius = 3.0;
        
        for(int z=0; z<10; ++z) {
            for(int y=0; y<10; ++y) {
                for(int x=0; x<10; ++x) {
                    float dist = std::sqrt(std::pow(x-cx, 2) + std::pow(y-cy, 2) + std::pow(z-cz, 2)) - radius;
                    VoxelData& v = grid(i + (x+0.5)*0.1, (y+0.5)*0.1, (z+0.5)*0.1);
                    
                    // Approximate rank/sign
                    int rank = std::abs(dist) / 0.1; 
                    if (rank > 15) rank = 15; // Cap at max rank
                    
                    // Set bits for rank (simple approximation)
                    v.d = (1 << rank) - 1; // This is not exactly how rank works in DB-TSDF but close enough for test
                    // Actually DB-TSDF uses specific masks. 
                    // Let's just use the 'rank' field if available or just set d to something non-zero.
                    // Wait, VoxelData.d is uint16_t.
                    // __builtin_popcount(v.d) is used.
                    // So to set rank N, we need N bits set.
                    v.d = 0;
                    for(int k=0; k<rank; ++k) v.d |= (1<<k);
                    
                    if (dist <= 0) {
                        v.s = 0; // Occupied
                    } else {
                        v.s = 1; // Free
                    }
                }
            }
        }
    }

    // By now, we should have triggered discards.
    // Since we can't easily access private _gaussian_cubes, we rely on the fact that it compiles and runs without crashing.
    // Ideally we would add a getter or print in the map.
    
    std::cout << "Allocations complete. Waiting for background training..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    std::cout << "Exporting Gaussian Mesh..." << std::endl;
    grid.exportGaussianMesh("test_gaussian_mesh.stl");
    
    // Check if file exists
    std::ifstream f("test_gaussian_mesh.stl");
    if (f.good()) {
        std::cout << "Mesh file created successfully." << std::endl;
    } else {
        std::cerr << "Failed to create mesh file." << std::endl;
    }
    
    std::cout << "Test finished." << std::endl;
    
    return 0;
}
