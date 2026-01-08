#ifndef __GRID_16_HPP__
#define __GRID_16_HPP__

#include <algorithm>  
#include <bitset>
#include <stdint.h>
#include <cmath>
#include <omp.h>
#include <vector>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <clocale>

// Mesh
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkMarchingCubes.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkSTLWriter.h>
#include <vtkAppendPolyData.h>
#include <vtkImageGaussianSmooth.h>
#include <vtkPolyData.h>
#include <vtkWindowedSincPolyDataFilter.h>

// DB-TSDF
#include "db_tsdf/voxel_cube.hpp"
#include "db_tsdf/gaussian_cube.hpp"

// CSV
#include <fstream> 
#include <iomanip>


class GRID16
{
    public:

	struct Iterator
    {
        Iterator(GRID16* parent, VoxelData **grid, uint32_t i, uint32_t base, uint32_t j, uint32_t cellSizeX) 
        { 
            _parent = parent;
            _grid = grid; 
            _i = i;
            _j = j;
            _base = base;
            _cellSizeX = cellSizeX;
            _curr = _grid[_i];
        }

        Iterator& operator=(const Iterator &it)
        { 
            _grid = it._grid; 
            _i = it._i;
            _j = it._j;
            _base = it._base;
            _cellSizeX = it._cellSizeX;
            _curr = it._curr;
            return *this;
        }

        VoxelData &operator*() { 
            if (_curr == _parent->_dummy) {
                return _parent->_garbage; 
            }
            return _curr[_j+_base]; 
        }

        VoxelData *operator->() { 
            if (_curr == _parent->_dummy) {
                return &(_parent->_garbage);
            }
            return _curr + _j + _base; 
        }

        Iterator& operator++() 
        { 
            _j++;
            if(_j >= _cellSizeX)
            {
                _j = 0;
                _i++;
                _curr = _grid[_i];
            }
            return *this; 
        }  

        protected:
        GRID16* _parent;
        VoxelData **_grid;
        VoxelData *_curr;
        uint32_t _i, _j, _base, _cellSizeX;
    };


    GRID16(void)
	{
		_grid = NULL;
        _buffer = NULL; // Circular buffer to store all cell masks
		_garbage = VoxelData{0xFFFFu, 0xFF, 0xFF};
		_dummy = NULL;
        _reset_template = NULL;
    }

    void setup(float minX, float maxX, float minY, float maxY, float minZ, float maxZ, float cellRes = 0.05, int maxCells = 100000)
	{
		if(_grid != NULL)
			free(_grid);

        if(_buffer != NULL)
			free(_buffer);

		if(_dummy != NULL)  
			free(_dummy);

        if(_reset_template != NULL)
            free(_reset_template);

		_maxX = (int)ceil(maxX);
		_maxY = (int)ceil(maxY);
		_maxZ = (int)ceil(maxZ);
		_minX = (int)floor(minX);
		_minY = (int)floor(minY);
		_minZ = (int)floor(minZ);
        
		// Memory allocation for the world grid. It is initialized to 0 (NULL)
		_gridSizeX = abs(_maxX-_minX);
		_gridSizeY = abs(_maxY-_minY);
		_gridSizeZ = abs(_maxZ-_minZ);
		_gridStepY = _gridSizeX;
		_gridStepZ = _gridSizeX*_gridSizeY;
		_gridSize = _gridSizeX*_gridSizeY*_gridSizeZ;
        
        // Memory allocation for maxCells
        _maxCells = (uint64_t)maxCells;
        _cellRes = cellRes;
        _oneDivRes = 1.0/_cellRes;
        _cellSizeX = (uint32_t)_oneDivRes;
        _cellSizeY = (uint32_t)_oneDivRes;
        _cellSizeZ = (uint32_t)_oneDivRes;
        _cellStepY = _cellSizeX;
        _cellStepZ = _cellSizeX*_cellSizeY;
        _cellSize = 1 + static_cast<uint64_t>(_cellSizeX)*_cellSizeY*_cellSizeZ;  // The 1 is to store control information of the cell
        _buffer = (VoxelData *)malloc(_maxCells*_cellSize*sizeof(VoxelData)); // Circular buffer to store all cell masks
        std::memset(_buffer, -1, _maxCells*_cellSize*sizeof(VoxelData));     // Init the buffer to longest distance
        for(int i=0; i<_maxCells; i++)
        {
            uint32_t* control_index_ptr = reinterpret_cast<uint32_t*>(&_buffer[i*_cellSize]);
            *control_index_ptr = _gridSize;
        }
        _cellIndex = 0;

		_dummy = (VoxelData*)malloc(_cellSize * sizeof(VoxelData));
		std::memset(_dummy, -1, _cellSize * sizeof(VoxelData));
        uint32_t* dummy_index_ptr = reinterpret_cast<uint32_t*>(&_dummy[0]);
        *dummy_index_ptr = _gridSize;
        _grid = (VoxelData**)malloc(_gridSize * sizeof(VoxelData*));
        for (uint32_t k = 0; k < _gridSize; ++k) _grid[k] = _dummy;
        
        // Initialize reset template for fast memcpy
        _reset_template = (VoxelData*)malloc(_cellSize * sizeof(VoxelData));
        // We only care about indices 1.._cellSize-1
        for(uint64_t j=1; j<_cellSize; ++j) {
            _reset_template[j].d = 0xFFFFu;
            _reset_template[j].s = 1u;
            _reset_template[j].hits = 0u;

            // _reset_template[j].offX = 0;
            // _reset_template[j].offY = 0;
            // _reset_template[j].offZ = 0;
            // _reset_template[j].posHits = 0;
        }

        _last_update_frame.resize(_gridSize, 0);
        _current_frame = 0;
    }    

    ~GRID16(void)
	{
		if(_grid != NULL)
			free(_grid);

        if(_buffer != NULL)
			free(_buffer);
		if(_dummy != NULL)  
			free(_dummy); 
        if(_reset_template != NULL)
            free(_reset_template); 
	
    }

	void clear(void)
	{
		for (uint32_t k = 0; k < _gridSize; ++k) _grid[k] = _dummy; // Set pointers to dummy
		std::memset(_buffer, -1, _maxCells*_cellSize*sizeof(VoxelData));     // Init the buffer to longest distance
        for(uint64_t i=0; i<_maxCells; i++)
        {
            uint32_t* control_index_ptr = reinterpret_cast<uint32_t*>(&_buffer[i*_cellSize]);
            *control_index_ptr = _gridSize;
        }
        _cellIndex = 0;
        std::fill(_last_update_frame.begin(), _last_update_frame.end(), 0);
        _current_frame = 0;
	}

	void allocCell(float x, float y, float z)
	{
		x -= _minX;
		y -= _minY;
		z -= _minZ;
		uint32_t int_x = (uint32_t)x, int_y = (uint32_t)y, int_z = (uint32_t)z;
        uint32_t i = int_x + int_y*_gridStepY + int_z*_gridStepZ;
        
        // Update timestamp
        if (i < _gridSize) {
            _last_update_frame[i] = _current_frame;
        }

		if( _grid[i] == _dummy)  
		{
			_grid[i] = _buffer + (_cellIndex % _maxCells)*_cellSize;
            // --- DISCARDING / REUSE LOGIC ---
            uint32_t* old_index_ptr = reinterpret_cast<uint32_t*>(&_grid[i][0]);

            // If the cell was previously allocated (old_index != _gridSize), we are discarding it.
            // We should train a GaussianCube from the old data before overwriting it.
            if (*old_index_ptr != _gridSize) {
                uint32_t prev_owner_idx = *old_index_ptr;
                
                if (prev_owner_idx != _gridSize) {
                    // This buffer block was used by `prev_owner_idx` in the grid.
                    // We need to invalidate the pointer in the grid for that old index.
                    _grid[prev_owner_idx] = _dummy;
                }
            }
            
            
            VoxelData* cell = _grid[i];
            // Optimized reset using memcpy
            std::memcpy(cell + 1, _reset_template + 1, (_cellSize - 1) * sizeof(VoxelData));

            uint32_t* control_index_ptr = reinterpret_cast<uint32_t*>(&cell[0]);
            *control_index_ptr = i;

			_cellIndex++;
		}
	}

	VoxelData &operator()(float x, float y, float z)
	{
		x -= _minX;
		y -= _minY;
		z -= _minZ;
		uint32_t int_x = (uint32_t)x, int_y = (uint32_t)y, int_z = (uint32_t)z;
        uint32_t i = int_x + int_y*_gridStepY + int_z*_gridStepZ;
        
        // Update timestamp for this cell
        if (i < _gridSize) {
            _last_update_frame[i] = _current_frame;
        }
        
		if(_grid[i] == _dummy) { return _garbage; }
		uint32_t j = 1 + (uint32_t)((x-int_x)*_oneDivRes) + (uint32_t)((y-int_y)*_oneDivRes)*_cellStepY + (uint32_t)((z-int_z)*_oneDivRes)*_cellStepZ;

        return _grid[i][j];
	}

	VoxelData read(float x, float y, float z)
	{
		x -= _minX;
		y -= _minY;
		z -= _minZ;
		uint32_t int_x = (uint32_t)x, int_y = (uint32_t)y, int_z = (uint32_t)z;
        uint32_t i = int_x + int_y*_gridStepY + int_z*_gridStepZ;
		if(_grid[i] == _dummy) { return _garbage; }

        uint32_t j = 1 + (uint32_t)((x-int_x)*_oneDivRes) + (uint32_t)((y-int_y)*_oneDivRes)*_cellStepY + (uint32_t)((z-int_z)*_oneDivRes)*_cellStepZ;

        return _grid[i][j];
	}

	Iterator getIterator(float x, float y, float z)
	{
		x -= _minX;
		y -= _minY;
		z -= _minZ;
		uint32_t int_x = (uint32_t)x, int_y = (uint32_t)y, int_z = (uint32_t)z;
        uint32_t i = int_x + int_y*_gridStepY + int_z*_gridStepZ;

		return Iterator(this, _grid, i, 1 + (uint32_t)((y-int_y)*_oneDivRes)*_cellStepY + (uint32_t)((z-int_z)*_oneDivRes)*_cellStepZ, (uint32_t)((x-int_x)*_oneDivRes), _cellSizeX);
	}
    
    void exportGridToPCD(const std::string& filename, int subsampling_factor)
        {
        using PointT = pcl::PointXYZ;
        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
        const uint32_t step = std::max(1, subsampling_factor);

        for (uint32_t cz = 0; cz < _gridSizeZ; ++cz)
        {
            const float z0 = _minZ + static_cast<float>(cz);
            for (uint32_t cy = 0; cy < _gridSizeY; ++cy)
            {
                const float y0 = _minY + static_cast<float>(cy);
                for (uint32_t cx = 0; cx < _gridSizeX; ++cx)
                {
                    const float x0 = _minX + static_cast<float>(cx);
                    const uint32_t i = cx + cy * _gridStepY + cz * _gridStepZ;
                    VoxelData* cell = _grid[i];
                    if (cell == _dummy) continue;
                    for (uint32_t vz = 0; vz < _cellSizeZ; vz += step) {
                        for (uint32_t vy = 0; vy < _cellSizeY; vy += step) {
                            for (uint32_t vx = 0; vx < _cellSizeX; vx += step) {
                                const uint32_t j = 1u + vx + vy * _cellStepY + vz * _cellStepZ;

                               const uint64_t dist = __builtin_popcount(cell[j].d);   
                                if (dist > 1u)                 continue;            
                                if ((cell[j].s & 0x01u) != 0u)  continue;    

                                PointT pt;
                                pt.x = x0 + (vx + 0.5f) * _cellRes;
                                pt.y = y0 + (vy + 0.5f) * _cellRes;
                                pt.z = z0 + (vz + 0.5f) * _cellRes;
                                cloud->push_back(pt);
                            }
                        }
                    }
                }
            }
        }
        if (cloud->empty())
        {
            std::cerr << "[GRID16] Warning: Empty Cloud (no mask==0 found).\n";
            return;
        }
        std::cout << "[GRID16] Total points (mask==0): " << cloud->size() << "\n";
        pcl::io::savePCDFileBinary(filename, *cloud);
        std::cout << "[GRID16] PCD exported: " << filename << "\n";
    }
    
    // void exportGridToPCD(const std::string& filename, int subsampling_factor)
    // {
    //     using PointT = pcl::PointXYZ;
    //     pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    //     const uint32_t step = std::max(1, subsampling_factor);

    //     // Factor de escala para convertir int8 (-128..127) a distancia física
    //     // El rango total del byte (254 pasos utiles) cubre el tamaño del voxel (_cellRes)
    //     const float offset_scale = _cellRes / 254.0f;

    //     for (uint32_t cz = 0; cz < _gridSizeZ; ++cz)
    //     {
    //         const float z0 = _minZ + static_cast<float>(cz);
    //         for (uint32_t cy = 0; cy < _gridSizeY; ++cy)
    //         {
    //             const float y0 = _minY + static_cast<float>(cy);
    //             for (uint32_t cx = 0; cx < _gridSizeX; ++cx)
    //             {
    //                 const float x0 = _minX + static_cast<float>(cx);
    //                 const uint32_t i = cx + cy * _gridStepY + cz * _gridStepZ;
    //                 VoxelData* cell = _grid[i];
    //                 if (cell == _dummy) continue;
                    
    //                 for (uint32_t vz = 0; vz < _cellSizeZ; vz += step) {
    //                     for (uint32_t vy = 0; vy < _cellSizeY; vy += step) {
    //                         for (uint32_t vx = 0; vx < _cellSizeX; vx += step) {
    //                             const uint32_t j = 1u + vx + vy * _cellStepY + vz * _cellStepZ;

    //                             const uint64_t dist = __builtin_popcount(cell[j].d);   
    //                             if (dist > 1u)                 continue;            
    //                             if ((cell[j].s & 0x01u) != 0u)  continue;    

    //                             PointT pt;
                                
    //                             // 1. Calcular el centro geométrico del vóxel
    //                             float center_x = x0 + (vx + 0.5f) * _cellRes;
    //                             float center_y = y0 + (vy + 0.5f) * _cellRes;
    //                             float center_z = z0 + (vz + 0.5f) * _cellRes;

    //                             // 2. Sumar el offset real (descomprimido)
    //                             pt.x = center_x + (static_cast<float>(cell[j].offX) * offset_scale);
    //                             pt.y = center_y + (static_cast<float>(cell[j].offY) * offset_scale);
    //                             pt.z = center_z + (static_cast<float>(cell[j].offZ) * offset_scale);
                                
    //                             cloud->push_back(pt);
    //                         }
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     if (cloud->empty())
    //     {
    //         std::cerr << "[GRID16] Warning: Empty Cloud (no mask==0 found).\n";
    //         return;
    //     }
    //     std::cout << "[GRID16] Total points (mask==0): " << cloud->size() << "\n";
    //     pcl::io::savePCDFileBinary(filename, *cloud);
    //     std::cout << "[GRID16] PCD exported: " << filename << "\n";
    // }

    void exportGridToPLY(const std::string& filename, int subsampling_factor)
        {
        using PointT = pcl::PointXYZ;
        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
        const uint32_t step = std::max(1, subsampling_factor);

        for (uint32_t cz = 0; cz < _gridSizeZ; ++cz)
        {
            const float z0 = _minZ + static_cast<float>(cz);
            for (uint32_t cy = 0; cy < _gridSizeY; ++cy)
            {
                const float y0 = _minY + static_cast<float>(cy);
                for (uint32_t cx = 0; cx < _gridSizeX; ++cx)
                {
                    const float x0 = _minX + static_cast<float>(cx);
                    const uint32_t i = cx + cy * _gridStepY + cz * _gridStepZ;
                    VoxelData* cell = _grid[i];
                    if (cell == _dummy) continue;
                    for (uint32_t vz = 0; vz < _cellSizeZ; vz += step) {
                        for (uint32_t vy = 0; vy < _cellSizeY; vy += step) {
                            for (uint32_t vx = 0; vx < _cellSizeX; vx += step) {
                                const uint32_t j = 1u + vx + vy * _cellStepY + vz * _cellStepZ;

                               const uint64_t dist = __builtin_popcount(cell[j].d); 
                                if (dist > 1u)                 continue;            
                                if ((cell[j].s & 0x01u) != 0u)  continue;            

                                PointT pt;
                                pt.x = x0 + (vx + 0.5f) * _cellRes;
                                pt.y = y0 + (vy + 0.5f) * _cellRes;
                                pt.z = z0 + (vz + 0.5f) * _cellRes;
                                cloud->push_back(pt);
                            }
                        }
                    }
                }
            }
        }
        if (cloud->empty())
        {
            std::cerr << "[GRID16] Warning: Empty Cloud (no mask==0 found).\n";
            return;
        }
        std::cout << "[GRID16] Total points (mask==0): " << cloud->size() << "\n";
        pcl::io::savePLYFileBinary(filename, *cloud);
        std::cout << "[GRID16] PLY exported: " << filename << "\n";
    }

    void exportSubgridToCSV(const std::string& filename, int subsampling_factor)
        {
        (void)filename;
        (void)subsampling_factor;

        const std::string out_dir = "/home/ros/ros2_ws/tests_results/csv";
        // std::filesystem::create_directories(out_dir);

        for (uint32_t cz = 0; cz < _gridSizeZ; ++cz)
        {
            const float z0 = _minZ + static_cast<float>(cz);
            for (uint32_t cy = 0; cy < _gridSizeY; ++cy)
            {
                const float y0 = _minY + static_cast<float>(cy);
                for (uint32_t cx = 0; cx < _gridSizeX; ++cx)
                {
                    const float x0 = _minX + static_cast<float>(cx);

                    const uint32_t i = cx + cy * _gridStepY + cz * _gridStepZ;
                    VoxelData* cell = _grid[i];
                    if (cell == _dummy) continue;

                    bool has_occupied = false;
                    for (uint32_t vz = 0; vz < _cellSizeZ && !has_occupied; ++vz)
                    for (uint32_t vy = 0; vy < _cellSizeY && !has_occupied; ++vy)
                    for (uint32_t vx = 0; vx < _cellSizeX; ++vx)
                    {
                        const uint32_t j = 1u + vx + vy * _cellStepY + vz * _cellStepZ;
                        if ( (cell[j].s & 0x01u) == 0u ) { has_occupied = true; break; }
                    }
                    if (!has_occupied) continue;

                    const int X = static_cast<int>(std::floor(x0 + 1e-6f));
                    const int Y = static_cast<int>(std::floor(y0 + 1e-6f));
                    const int Z = static_cast<int>(std::floor(z0 + 1e-6f));

                    std::ostringstream base;
                    base << out_dir << "/" << X << "_" << Y << "_" << Z;

                    // --- CSV ---
                    std::ofstream f_csv(base.str() + ".csv");
                    if (!f_csv.is_open()) {
                        std::cerr << "[GRID] No se pudo abrir " << (base.str()+".csv") << "\n";
                        continue;
                    }
                    f_csv << "x,y,z,d_manhattan,s,hits\n";
                    f_csv << std::fixed << std::setprecision(6);

                    // --- PLY de puntos en centros ocupados ---
                    std::vector<std::array<float,3>> pts; pts.reserve(1024);

                    const float half = 0.5f * _cellRes;
                    for (uint32_t vz = 0; vz < _cellSizeZ; ++vz)
                    {
                        const float z_voxel_min = z0 + static_cast<float>(vz) * _cellRes;
                        for (uint32_t vy = 0; vy < _cellSizeY; ++vy)
                        {
                            const float y_voxel_min = y0 + static_cast<float>(vy) * _cellRes;
                            for (uint32_t vx = 0; vx < _cellSizeX; ++vx)
                            {
                                const float x_voxel_min = x0 + static_cast<float>(vx) * _cellRes;
                                const uint32_t j = 1u + vx + vy * _cellStepY + vz * _cellStepZ;

                                const uint64_t d_manhattan = __builtin_popcount(cell[j].d);
                                // const uint32_t s    = static_cast<uint32_t>(cell[j].s);
                                const uint32_t s = (cell[j].s & 0x01u);
                                const uint32_t hits = static_cast<uint32_t>(cell[j].hits);

                                // >>> CAMBIO: escribir el CENTRO del vóxel en el CSV <<<
                                const float cx = x_voxel_min + half;
                                const float cy = y_voxel_min + half;
                                const float cz = z_voxel_min + half;
                                f_csv << cx << "," << cy << "," << cz << ","
                                    << d_manhattan << "," << s << "," << hits << "\n";

                                // Centro del voxel si está ocupado (bit 0 == 0)
                                if ( (s & 0x01u) == 0u ) {
                                    pts.push_back( { cx, cy, cz } );
                                }
                            }
                        }
                    }
                    f_csv.close();

                    // Escribir PLY ASCII solo con vértices
                    std::ofstream f_ply(base.str() + ".ply");
                    if (!f_ply.is_open()) {
                        std::cerr << "[GRID] No se pudo abrir " << (base.str()+".ply") << "\n";
                        continue;
                    }
                    f_ply << "ply\nformat ascii 1.0\n";
                    f_ply << "element vertex " << pts.size() << "\n";
                    f_ply << "property float x\nproperty float y\nproperty float z\n";
                    f_ply << "end_header\n";
                    f_ply << std::fixed << std::setprecision(6);
                    for (const auto& p : pts) {
                        f_ply << p[0] << " " << p[1] << " " << p[2] << "\n";
                    }
                    f_ply.close();
                }
            }
        }
    }

    void exportMesh(const std::string& filename, float iso_level, int occ_min_hits){
        // RCLCPP_INFO(rclcpp::get_logger("GRID16_Mesh"), "Starting mesh extraction..."); // Commented out as RCLCPP is not defined here

        vtkSmartPointer<vtkAppendPolyData> appender = 
            vtkSmartPointer<vtkAppendPolyData>::New();

        const float BAND = 0.1f * _cellRes;

        for (uint32_t cz = 0; cz < _gridSizeZ; ++cz)
        for (uint32_t cy = 0; cy < _gridSizeY; ++cy)
        for (uint32_t cx = 0; cx < _gridSizeX; ++cx)
        {
            const uint32_t i = cx + cy * _gridStepY + cz * _gridStepZ;
            VoxelData* cell = _grid[i];

            if (cell == _dummy) continue;

            vtkSmartPointer<vtkImageData> image = vtkSmartPointer<vtkImageData>::New();
            image->SetDimensions(_cellSizeX + 1, _cellSizeY + 1, _cellSizeZ + 1); 
            image->SetSpacing(_cellRes, _cellRes, _cellRes);
            const float x0 = _minX + static_cast<float>(cx);
            const float y0 = _minY + static_cast<float>(cy);
            const float z0 = _minZ + static_cast<float>(cz);
            image->SetOrigin(x0, y0, z0);
            image->AllocateScalars(VTK_FLOAT, 1);
            
            float *dest = static_cast<float*>(image->GetScalarPointer());
            bool has_occupied_voxels = false;

            for (uint32_t vz = 0; vz < _cellSizeZ + 1; ++vz)
            for (uint32_t vy = 0; vy < _cellSizeY + 1; ++vy)
            for (uint32_t vx = 0; vx < _cellSizeX + 1; ++vx)
            {
                VoxelData vox = this->read(x0 + vx * _cellRes, 
                                           y0 + vy * _cellRes, 
                                           z0 + vz * _cellRes);

                const uint64_t dist_rank = __builtin_popcount(vox.d);
                const bool enough_hits = (vox.hits >= occ_min_hits);
                const bool occupied = ((vox.s & 0x01u) == 0);
                const bool is_surface = (dist_rank <= 1); 

                float sdf_value;
                if (!enough_hits) {
                    sdf_value = +BAND; 
                } else if (occupied) {
                    sdf_value = -BAND; 
                    has_occupied_voxels = true;
                } else {
                    sdf_value = +BAND; 
                }
                
                dest[vx + vy * (_cellSizeX + 1) + vz * (_cellSizeX + 1) * (_cellSizeY + 1)] = sdf_value;
            }

            if (has_occupied_voxels)
            {
                auto smoother = vtkSmartPointer<vtkImageGaussianSmooth>::New();
                smoother->SetInputData(image);
                smoother->SetStandardDeviation(1.0);
                smoother->Update();

                auto mc = vtkSmartPointer<vtkMarchingCubes>::New();
                mc->SetInputConnection(smoother->GetOutputPort());
                mc->SetValue(0, iso_level); 
                mc->Update();

                appender->AddInputData(mc->GetOutput());
            }
        } 

        // RCLCPP_INFO(rclcpp::get_logger("GRID16_Mesh"), "Joining cell meshes..."); // Commented out
        appender->Update();

        auto ext_pos = filename.find_last_of('.');
        std::string ext = (ext_pos==std::string::npos) ? "" : filename.substr(ext_pos+1);

        if (ext == "stl") {
            auto writer = vtkSmartPointer<vtkSTLWriter>::New();
            writer->SetFileName(filename.c_str());
            writer->SetInputData(appender->GetOutput());
            writer->SetFileTypeToBinary();  
            if (!writer->Write()) {
              throw std::runtime_error("VTK STL writer failed to write mesh.");
            }
        }
        else if (ext == "vtp") {
            auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
            writer->SetFileName(filename.c_str());
            writer->SetInputData(appender->GetOutput());
            if (!writer->Write()) {
              throw std::runtime_error("VTK XML writer failed to write mesh.");
            }
        }
        else {
            throw std::invalid_argument("Unsupported file extension: " + ext);
        }
        
        // RCLCPP_INFO(rclcpp::get_logger("GRID16_Mesh"), "Mesh saved to %s", filename.c_str()); // Commented out
    }

    // void exportMesh(const std::string& filename, float iso_level, int occ_min_hits) {
    //     vtkSmartPointer<vtkAppendPolyData> appender = vtkSmartPointer<vtkAppendPolyData>::New();

    //     // Configuración de suavizado y búsqueda
    //     const float offset_scale = _cellRes / 254.0f;
    //     const float MAX_SEARCH_DIST = 1.5f * _cellRes; // Buscar hasta 1.5 voxels de distancia
    //     const float TRUNCATION = 2.0f * _cellRes;      // Valor de saturación

    //     for (uint32_t cz = 0; cz < _gridSizeZ; ++cz)
    //     for (uint32_t cy = 0; cy < _gridSizeY; ++cy)
    //     for (uint32_t cx = 0; cx < _gridSizeX; ++cx)
    //     {
    //         const uint32_t i = cx + cy * _gridStepY + cz * _gridStepZ;
    //         VoxelData* cell = _grid[i];

    //         if (cell == _dummy) continue;

    //         // Creamos una imagen VTK con bordes (+1) para cerrar la malla correctamente
    //         vtkSmartPointer<vtkImageData> image = vtkSmartPointer<vtkImageData>::New();
    //         int dimX = _cellSizeX + 1;
    //         int dimY = _cellSizeY + 1;
    //         int dimZ = _cellSizeZ + 1;
            
    //         image->SetDimensions(dimX, dimY, dimZ);
    //         image->SetSpacing(_cellRes, _cellRes, _cellRes);
            
    //         const float x0 = _minX + static_cast<float>(cx);
    //         const float y0 = _minY + static_cast<float>(cy);
    //         const float z0 = _minZ + static_cast<float>(cz);
    //         image->SetOrigin(x0, y0, z0);
    //         image->AllocateScalars(VTK_FLOAT, 1);
            
    //         float *dest = static_cast<float*>(image->GetScalarPointer());
    //         bool has_surface = false;

    //         // Recorremos la rejilla local para generar el campo escalar
    //         for (int vz = 0; vz < dimZ; ++vz)
    //         for (int vy = 0; vy < dimY; ++vy)
    //         for (int vx = 0; vx < dimX; ++vx)
    //         {
    //             // Coordenadas locales del grid point actual
    //             // Nota: Usamos índices int para buscar vecinos de forma segura
                
    //             // 1. Determinar el estado base del vóxel actual (para el signo)
    //             // (Clamp índices para no salirnos al leer el estado propio)
    //             int r_vx = std::min(vx, (int)_cellSizeX - 1);
    //             int r_vy = std::min(vy, (int)_cellSizeY - 1);
    //             int r_vz = std::min(vz, (int)_cellSizeZ - 1);
                
    //             VoxelData center_vox = this->read(x0 + r_vx*_cellRes, y0 + r_vy*_cellRes, z0 + r_vz*_cellRes);
    //             bool is_occupied = ((center_vox.s & 0x01u) == 0) && (center_vox.hits >= occ_min_hits);

    //             // 2. BUSCAR EL PUNTO EXACTO MÁS CERCANO (Nearest Neighbor Search)
    //             // Buscamos en un entorno 3x3 alrededor de este punto de la rejilla
    //             float min_dist_sq = 1e9f;
                
    //             for(int dz = -1; dz <= 1; ++dz)
    //             for(int dy = -1; dy <= 1; ++dy)
    //             for(int dx = -1; dx <= 1; ++dx)
    //             {
    //                 // Coordenada del vecino a consultar
    //                 int nx = r_vx + dx;
    //                 int ny = r_vy + dy;
    //                 int nz = r_vz + dz;

    //                 // Bounds check (dentro del subgrid local + padding virtual)
    //                 // Usamos 'read' que ya gestiona coordenadas globales, es más seguro
    //                 float world_nx = x0 + nx * _cellRes;
    //                 float world_ny = y0 + ny * _cellRes;
    //                 float world_nz = z0 + nz * _cellRes;

    //                 VoxelData n_vox = this->read(world_nx, world_ny, world_nz);

    //                 // Si el vecino tiene una superficie válida (hits suficientes y ocupado)
    //                 if ((n_vox.s & 0x01u) == 0 && n_vox.hits >= occ_min_hits)
    //                 {
    //                     // Posición absoluta del punto exacto dentro de ese vecino
    //                     float exact_nx = world_nx + 0.5f*_cellRes + (static_cast<float>(n_vox.offX) * offset_scale);
    //                     float exact_ny = world_ny + 0.5f*_cellRes + (static_cast<float>(n_vox.offY) * offset_scale);
    //                     float exact_nz = world_nz + 0.5f*_cellRes + (static_cast<float>(n_vox.offZ) * offset_scale);

    //                     // Posición actual de mi grid point
    //                     float current_x = x0 + vx * _cellRes;
    //                     float current_y = y0 + vy * _cellRes;
    //                     float current_z = z0 + vz * _cellRes;

    //                     float dist_sq = std::pow(current_x - exact_nx, 2) + 
    //                                     std::pow(current_y - exact_ny, 2) + 
    //                                     std::pow(current_z - exact_nz, 2);

    //                     if (dist_sq < min_dist_sq) {
    //                         min_dist_sq = dist_sq;
    //                     }
    //                 }
    //             }

    //             // 3. Calcular valor SDF final
    //             float dist = std::sqrt(min_dist_sq);
    //             float sdf_value;

    //             // Si encontramos un vecino cercano (dentro del rango de influencia)
    //             if (dist < MAX_SEARCH_DIST) {
    //                 // El signo depende de si YO (el vóxel central) estoy ocupado o no
    //                 if (is_occupied) sdf_value = -dist;
    //                 else             sdf_value =  dist;
                    
    //                 has_surface = true;
    //             } else {
    //                 // Lejos de la superficie -> Truncar
    //                 if (is_occupied) sdf_value = -TRUNCATION;
    //                 else             sdf_value =  TRUNCATION;
    //             }

    //             dest[vx + vy * dimX + vz * dimX * dimY] = sdf_value;
    //         }

    //         if (has_surface)
    //         {
    //             // 1. Suavizado de Imagen (IMPRESCINDIBLE para quitar granulación)
    //             // Un valor de 0.6 - 0.8 es ideal para fusionar los voxeles sin perder detalle.
    //             auto smoother = vtkSmartPointer<vtkImageGaussianSmooth>::New();
    //             smoother->SetInputData(image);
    //             smoother->SetStandardDeviation(1.0); 
    //             smoother->Update();

    //             // 2. Marching Cubes
    //             auto mc = vtkSmartPointer<vtkMarchingCubes>::New();
    //             mc->SetInputConnection(smoother->GetOutputPort());
    //             mc->SetValue(0, iso_level); 
    //             mc->Update();

    //             // Solo añadimos si hay geometría real
    //             if (mc->GetOutput()->GetNumberOfPoints() > 0) {
    //                 appender->AddInputData(mc->GetOutput());
    //             }
    //         }
    //     } 

    //     appender->Update();

    //     // Guardado (Igual que siempre)
    //     auto ext_pos = filename.find_last_of('.');
    //     std::string ext = (ext_pos==std::string::npos) ? "" : filename.substr(ext_pos+1);

    //     if (ext == "stl") {
    //         auto writer = vtkSmartPointer<vtkSTLWriter>::New();
    //         writer->SetFileName(filename.c_str());
    //         writer->SetInputData(appender->GetOutput());
    //         writer->SetFileTypeToBinary();  
    //         if (!writer->Write()) throw std::runtime_error("VTK STL writer failed.");
    //     }
    //     else if (ext == "vtp") {
    //         auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();
    //         writer->SetFileName(filename.c_str());
    //         writer->SetInputData(appender->GetOutput());
    //         if (!writer->Write()) throw std::runtime_error("VTK XML writer failed.");
    //     }
    //     else {
    //         throw std::invalid_argument("Unsupported file extension: " + ext);
    //     }
    // }


protected:

    VoxelData **_grid;
    float _maxX, _maxY, _maxZ, _minX, _minY, _minZ;
	uint32_t _gridSizeX, _gridSizeY, _gridSizeZ, _gridStepY, _gridStepZ, _gridSize;
    float _cellRes, _oneDivRes;
    uint32_t _cellSizeX, _cellSizeY, _cellSizeZ, _cellStepY, _cellStepZ;
    uint64_t _maxCells, _cellSize, _cellIndex;
    VoxelData *_buffer;
	VoxelData *_dummy;
    VoxelData *_reset_template;
	VoxelData _garbage;
    std::vector<uint32_t> _last_update_frame;
    uint32_t _current_frame;
    float _grid_mesh_iso = 0.0f;
    int _training_points = 500;
    bool _debug_mode = true;
    
public:

    void setTrainingPoints(int n) {
        _training_points = n;
    }

    void setCurrentFrame(uint32_t frame) {
        _current_frame = frame;
    }

    void setDebugMode(bool enabled) {
        _debug_mode = enabled;
    }
    
    void exportGaussianMesh(const std::string& filename) {
        // std::cout << "[GRID16] Exporting Gaussian Mesh to " << filename << "..." << std::endl;
        
        // vtkSmartPointer<vtkAppendPolyData> appender = vtkSmartPointer<vtkAppendPolyData>::New();
        
        // std::vector<vtkSmartPointer<vtkPolyData>> meshes;
        std::vector<CubeExportData> all_cube_data;

        const std::string C_RESET = "\033[0m";
        const std::string C_GREEN = "\033[32m";  
        const std::string C_YELLOW = "\033[33m"; 
        const std::string C_RED = "\033[31m";    
        const std::string C_CYAN = "\033[36m";   
        const std::string C_MAGENTA = "\033[35m"; 

        std::cout << "\n" << C_CYAN << "=== STARTING BATCH PROCESSING ===" << C_RESET << "\n\n";

        // 1. Filtrado previo
        std::vector<uint32_t> active_indices;
        active_indices.reserve(_gridSize / 10); 

        for(uint32_t i=0; i<_gridSize; ++i) {
            if (_grid[i] != _dummy) {
                bool has_occupied = false;
                bool has_free = false;
                for (uint16_t j = 1; j < _cellSize; ++j) {
                    if ((_grid[i][j].s & 0x01u) == 0) has_occupied = true;
                    else has_free = true;
                    if (has_occupied && has_free) break;
                }
                if (has_occupied && has_free) { 
                    active_indices.push_back(i);
                }
            }
        }
        
        std::cout << "[GRID16] Processing " << active_indices.size() << " surface cells (skipped solid/empty)..." << std::endl;
        
        #ifdef _OPENMP
        std::cout << "[GRID16] OpenMP Max Threads: " << omp_get_max_threads() << std::endl;
        #endif

        auto t_global_start = std::chrono::high_resolution_clock::now();
        
        int processed_count = 0;    
        int kept_count = 0;         
        int lost_count = 0;         
        double total_mae_kept = 0.0; 
        
        const double LOST_THRESHOLD = 0.5; 

        #pragma omp parallel
        {
            #pragma omp single
            {
                #ifdef _OPENMP
                std::cout << "[GRID16] Running with " << omp_get_num_threads() << " threads." << std::endl;
                #endif
            }

            Workspace ws;

            #pragma omp for schedule(dynamic)
            for(size_t k=0; k<active_indices.size(); ++k) {
                uint32_t i = active_indices[k];
                auto t_start = std::chrono::high_resolution_clock::now();

                uint32_t p_z = i / _gridStepZ;
                uint32_t rem = i % _gridStepZ;
                uint32_t p_y = rem / _gridStepY;
                uint32_t p_x = rem % _gridStepY;
                float p_fx = _minX + p_x;
                float p_fy = _minY + p_y;
                float p_fz = _minZ + p_z;
                
                auto t_load = std::chrono::high_resolution_clock::now();

                VoxelCube v_cube(_grid[i], _cellSizeX, _cellSizeY, _cellSizeZ, _cellRes, p_fx, p_fy, p_fz);
                auto t_init = std::chrono::high_resolution_clock::now(); 

                GaussianCube g_cube;
                g_cube.train(v_cube, ws, _training_points);
                auto t_opt = std::chrono::high_resolution_clock::now();
                
                // vtkSmartPointer<vtkPolyData> mesh = g_cube.get_mesh(p_fx, p_fy, p_fz, _cellRes, _cellSizeX, _cellSizeY, _cellSizeZ, ws);
                auto t_mesh = std::chrono::high_resolution_clock::now();
                
                auto d = [&](auto a, auto b) { return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count(); };
                long dur_total = d(t_start, t_mesh);
                long dur_load = d(t_start, t_load);
                long dur_init = d(t_load, t_init);
                long dur_opt = d(t_init, t_opt);
                long dur_mesh = d(t_opt, t_mesh);

                // if (mesh) {
                    #pragma omp critical
                    {
                        processed_count++;
                        
                        if (g_cube.mae > LOST_THRESHOLD) {
                            lost_count++;
                            std::stringstream ss;
                            ss << "[" << std::setw(3) << processed_count << "/" << active_indices.size() << "] "
                               << "Idx: " << std::setw(6) << i << " | "
                               << C_MAGENTA << "LOST (MAE: " << std::fixed << std::setprecision(4) << g_cube.mae << "m)" << C_RESET
                               << " | T: " << std::setw(3) << dur_total << "ms";
                            std::cout << ss.str() << std::endl;
                        } 
                        else {
                            kept_count++;
                            // meshes.push_back(mesh);
                            total_mae_kept += g_cube.mae;

                            CubeExportData cdata;
                            cdata.index = i;
                            cdata.x = p_fx; cdata.y = p_fy; cdata.z = p_fz;
                            cdata.mae = g_cube.mae;
                            cdata.params = g_cube.params; 
                            all_cube_data.push_back(cdata);
                            
                            if (_debug_mode) {
                                if (kept_count % 1 == 0 || g_cube.mae > 0.05) {
                                    std::stringstream ss;
                                    // Ancho fijo para contadores [  123/10000]
                                    ss << "[" << std::setw(5) << processed_count << "/" << std::setw(5) << active_indices.size() << "] "
                                    << "Idx:" << std::setw(6) << i << " | ";
                                    
                                    if (g_cube.mae > 0.03) ss << C_RED;
                                    else if (g_cube.mae > 0.02) ss << C_YELLOW;
                                    else ss << C_GREEN;
                                    
                                    // Ancho fijo para MAE (evita saltos entre 0.x y 1.x)
                                    ss << "MAE:" << std::fixed << std::setw(6) << std::setprecision(4) << g_cube.mae << "m" << C_RESET;
                                    
                                    // Ancho fijo para cada tiempo parcial
                                    ss << " | T:" << std::setw(4) << dur_total << "ms "
                                    << "(L:" << std::setw(2) << dur_load 
                                    << " I:" << std::setw(2) << dur_init 
                                    << " O:" << std::setw(4) << dur_opt     
                                    << " M:" << std::setw(2) << dur_mesh << ")";
                                    std::cout << ss.str() << std::endl;
                                }
                            }
                        }
                    }
                // }
            }
        } 
        
        auto t_global_end = std::chrono::high_resolution_clock::now();
        
        // --- ESTADÍSTICAS FINALES ---
        long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_global_end - t_global_start).count();
        double total_seconds = total_ms / 1000.0;
        
        double avg_ms_per_cube = (processed_count > 0) ? (double)total_ms / processed_count : 0.0;
        double cubes_per_sec = (total_seconds > 0) ? (processed_count / total_seconds) : 0.0;
        double avg_mae_kept = (kept_count > 0) ? (total_mae_kept / kept_count) : 0.0;
        double lost_percentage = (processed_count > 0) ? ((double)lost_count / processed_count * 100.0) : 0.0;
        double kept_percentage = (processed_count > 0) ? ((double)kept_count / processed_count * 100.0) : 0.0;
        
        std::cout << "\n" << C_CYAN << "=== SAVING FINAL MESH ===" << C_RESET << "\n";
        std::cout << " > Total Time:       " << std::fixed << std::setprecision(2) << total_seconds << " s\n";
        std::cout << " > Processed:        " << processed_count << " / " << active_indices.size() << "\n";        
        std::cout << " > KEPT Cubes:       " << kept_count << " (" << std::fixed << std::setprecision(2) << kept_percentage << "%)\n";
        std::cout << " > LOST Cubes:       " << lost_count << " (" << std::fixed << std::setprecision(2) << lost_percentage << "%)\n";
        std::cout << " > Avg MAE (Kept):   " << std::fixed << std::setprecision(5) << avg_mae_kept << " m\n";
        std::cout << " > Speed:            " << std::fixed << std::setprecision(2) << cubes_per_sec << " cubes/s\n";
        std::cout << " > Avg Time/Cube:    " << std::fixed << std::setprecision(1) << avg_ms_per_cube << " ms\n";
        
        // if (meshes.empty()) {
        //     std::cerr << "[GRID16] Warning: No meshes to save." << std::endl;
        //     return;
        // }

        // for (const auto& mesh : meshes) {
        //     appender->AddInputData(mesh);
        // }

        // appender->Update();
        // vtkSmartPointer<vtkSTLWriter> writer = vtkSmartPointer<vtkSTLWriter>::New();
        // writer->SetFileName(filename.c_str());
        // writer->SetInputData(appender->GetOutput());
        // writer->SetFileTypeToBinary();
        // writer->Write();
        // std::cout << "[GRID16] Saved mesh to " << filename << std::endl;

        // --- EXPORTAR CSV DE GAUSSIANAS ---
        std::string csv_filename = filename.substr(0, filename.find_last_of('.')) + ".csv";
        std::ofstream f(csv_filename);
        
        if (f.is_open()) {
            // Cabecera Exacta: Origen Cubo | Gaussiana Relativa (Mean, Sigma, Weight)
            f << "CubeIdx,CubeX,CubeY,CubeZ,MAE,"
              << "G_ID,RelMeanX,RelMeanY,RelMeanZ,SigmaX,SigmaY,SigmaZ,Weight\n";

            f << std::fixed << std::setprecision(6);

            for(const auto& cdata : all_cube_data) {
                for(int k=0; k<NUM_GAUSSIANS; ++k) {
                    int base = k * PARAMS_PER_GAUSSIAN;
                    
                    // 1. Leer parámetros absolutos del solver
                    double mx_abs = cdata.params[base + 0];
                    double my_abs = cdata.params[base + 1];
                    double mz_abs = cdata.params[base + 2];
                    double sx = cdata.params[base + 3];
                    double sy = cdata.params[base + 4];
                    double sz = cdata.params[base + 5];
                    double w  = cdata.params[base + 6];

                    // 2. Calcular Posición Relativa (CON SIGNO)
                    // Relativa = Absoluta - OrigenCubo
                    // Si la gaussiana está a la izquierda del cubo, saldrá negativo.
                    double rel_x = mx_abs - cdata.x;
                    double rel_y = my_abs - cdata.y;
                    double rel_z = mz_abs - cdata.z;

                    // 3. Escribir fila
                    f << cdata.index << ","
                      << cdata.x << "," << cdata.y << "," << cdata.z << ","
                      << cdata.mae << ","
                      << k << "," 
                      << rel_x << "," << rel_y << "," << rel_z << ","
                      << sx << "," << sy << "," << sz << ","
                      << w << "\n";
                }
            }
            f.close();
            std::cout << "[GRID16] Saved Gaussian CSV to " << csv_filename << std::endl;
            // std::cout << " > Rows: " << (all_cube_data.size() * NUM_GAUSSIANS) << std::endl;
        } else {
            std::cerr << "[GRID16] Error: Could not create CSV file." << std::endl;
        }
    }
};

#endif
