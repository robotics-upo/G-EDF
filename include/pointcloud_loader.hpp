#ifndef POINTCLOUD_LOADER_HPP
#define POINTCLOUD_LOADER_HPP

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <cstring>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>

namespace fs = std::filesystem;

/// PLY property info
struct PlyProperty {
    std::string name;
    std::string type;
    int size;
    int offset;
};

/// Parse PLY header and detect float64 properties
inline bool parsePlyHeader(std::ifstream& file, std::vector<PlyProperty>& props,
                           size_t& vertex_count, size_t& header_end, bool& is_binary,
                           bool& is_little_endian) {
    std::string line;
    int current_offset = 0;
    bool in_vertex_element = false;
    vertex_count = 0;
    is_binary = false;
    is_little_endian = true;

    while (std::getline(file, line)) {
        // Remove carriage return if present
        if (!line.empty() && line.back() == '\r') line.pop_back();

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "format") {
            std::string fmt;
            iss >> fmt;
            is_binary = (fmt != "ascii");
            is_little_endian = (fmt == "binary_little_endian");
        }
        else if (token == "element") {
            std::string elem_name;
            size_t count;
            iss >> elem_name >> count;
            in_vertex_element = (elem_name == "vertex");
            if (in_vertex_element) vertex_count = count;
        }
        else if (token == "property" && in_vertex_element) {
            std::string type, name;
            iss >> type >> name;

            PlyProperty prop;
            prop.name = name;
            prop.type = type;
            prop.offset = current_offset;

            if (type == "float" || type == "float32") prop.size = 4;
            else if (type == "double" || type == "float64") prop.size = 8;
            else if (type == "uchar" || type == "uint8") prop.size = 1;
            else if (type == "ushort" || type == "uint16") prop.size = 2;
            else if (type == "uint" || type == "uint32" || type == "int" || type == "int32") prop.size = 4;
            else prop.size = 4;

            current_offset += prop.size;
            props.push_back(prop);
        }
        else if (token == "end_header") {
            header_end = file.tellg();
            return true;
        }
    }
    return false;
}

/// Find property by name
inline const PlyProperty* findProperty(const std::vector<PlyProperty>& props, const std::string& name) {
    for (const auto& p : props) {
        if (p.name == name) return &p;
    }
    return nullptr;
}

/// Read value from buffer at offset
template<typename T>
inline T readValue(const char* buffer, int offset) {
    T val;
    std::memcpy(&val, buffer + offset, sizeof(T));
    return val;
}

/// Load binary PLY with float64 support
inline pcl::PointCloud<pcl::PointXYZ>::Ptr loadPlyFloat64(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Cannot open: " << filepath << std::endl;
        return nullptr;
    }

    std::vector<PlyProperty> props;
    size_t vertex_count, header_end;
    bool is_binary, is_little_endian;

    if (!parsePlyHeader(file, props, vertex_count, header_end, is_binary, is_little_endian)) {
        std::cerr << "[ERROR] Failed to parse PLY header" << std::endl;
        return nullptr;
    }

    const PlyProperty* px = findProperty(props, "x");
    const PlyProperty* py = findProperty(props, "y");
    const PlyProperty* pz = findProperty(props, "z");

    if (!px || !py || !pz) {
        std::cerr << "[ERROR] PLY missing x/y/z properties" << std::endl;
        return nullptr;
    }

    bool is_double = (px->type == "double" || px->type == "float64");

    // Calculate stride (bytes per point)
    int stride = 0;
    for (const auto& p : props) stride = std::max(stride, p.offset + p.size);

    std::cout << "[INFO] PLY format: " << (is_binary ? "binary" : "ascii")
              << ", coords: " << (is_double ? "float64" : "float32")
              << ", stride: " << stride << " bytes" << std::endl;

    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->reserve(vertex_count);

    if (!is_binary) {
        // ASCII format - not common for large files, fallback to PCL
        file.close();
        pcl::io::loadPLYFile<pcl::PointXYZ>(filepath, *cloud);
        return cloud;
    }

    // Binary format
    file.seekg(header_end);
    std::vector<char> buffer(stride);

    // Progress tracking
    size_t progress_step = vertex_count / 20;
    if (progress_step == 0) progress_step = 1;

    for (size_t i = 0; i < vertex_count; ++i) {
        file.read(buffer.data(), stride);
        if (!file.good()) break;

        pcl::PointXYZ pt;
        if (is_double) {
            pt.x = static_cast<float>(readValue<double>(buffer.data(), px->offset));
            pt.y = static_cast<float>(readValue<double>(buffer.data(), py->offset));
            pt.z = static_cast<float>(readValue<double>(buffer.data(), pz->offset));
        } else {
            pt.x = readValue<float>(buffer.data(), px->offset);
            pt.y = readValue<float>(buffer.data(), py->offset);
            pt.z = readValue<float>(buffer.data(), pz->offset);
        }
        cloud->push_back(pt);

        if ((i + 1) % progress_step == 0) {
            int pct = static_cast<int>((i + 1) * 100 / vertex_count);
            std::cout << "\r[INFO] Loading: " << pct << "%" << std::flush;
        }
    }
    std::cout << std::endl;

    cloud->width = cloud->size();
    cloud->height = 1;
    cloud->is_dense = false;

    return cloud;
}

/// Load PLY or PCD pointcloud file
inline pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloud(const std::string& filepath) {
    if (!fs::exists(filepath)) {
        std::cerr << "[ERROR] File not found: " << filepath << std::endl;
        return nullptr;
    }

    std::string ext = fs::path(filepath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;

    if (ext == ".ply") {
        // Use custom loader for float64 support
        cloud = loadPlyFloat64(filepath);
    } else if (ext == ".pcd") {
        cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(filepath, *cloud) < 0) {
            std::cerr << "[ERROR] Failed to load: " << filepath << std::endl;
            return nullptr;
        }
    } else {
        std::cerr << "[ERROR] Unsupported format: " << ext << " (use .ply or .pcd)" << std::endl;
        return nullptr;
    }

    if (cloud && !cloud->empty()) {
        std::cout << "[INFO] Loaded pointcloud: " << cloud->size() << " points" << std::endl;
    }
    return cloud;
}

/// Get axis-aligned bounding box of pointcloud
inline void getCloudBounds(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                           float& min_x, float& min_y, float& min_z,
                           float& max_x, float& max_y, float& max_z) {
    min_x = min_y = min_z = std::numeric_limits<float>::max();
    max_x = max_y = max_z = std::numeric_limits<float>::lowest();

    for (const auto& pt : cloud->points) {
        if (!std::isfinite(pt.x)) continue;
        min_x = std::min(min_x, pt.x);
        min_y = std::min(min_y, pt.y);
        min_z = std::min(min_z, pt.z);
        max_x = std::max(max_x, pt.x);
        max_y = std::max(max_y, pt.y);
        max_z = std::max(max_z, pt.z);
    }
}

#endif // POINTCLOUD_LOADER_HPP
