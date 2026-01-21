#ifndef CUBE_MANAGER_HPP
#define CUBE_MANAGER_HPP

#include <vector>
#include <cmath>
#include <unordered_map>
#include <iostream>
#include <chrono>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

/// 1m³ cube with associated point indices
struct Cube {
    int ix, iy, iz;
    float origin_x, origin_y, origin_z;
    std::vector<int> point_indices;

    float centerX() const { return origin_x + 0.5f; }
    float centerY() const { return origin_y + 0.5f; }
    float centerZ() const { return origin_z + 0.5f; }
};

/// Lightweight cube info (no indices stored)
struct CubeInfo {
    int ix, iy, iz;
    float origin_x, origin_y, origin_z;
    size_t point_count;

    float centerX() const { return origin_x + 0.5f; }
    float centerY() const { return origin_y + 0.5f; }
    float centerZ() const { return origin_z + 0.5f; }
};

/// Streaming cube manager - memory efficient for large pointclouds
class CubeManager {
public:
    explicit CubeManager(float cube_size = 1.0f) : cube_size_(cube_size), current_idx_(0) {}

    /// First pass: compute bounds and cube metadata (no index storage)
    void computeCubeMetadata(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
        cloud_ = cloud;
        cube_infos_.clear();
        current_idx_ = 0;

        std::cout << "[PHASE 1/3] Building KdTree for " << cloud->size() << " points..." << std::flush;
        auto t0 = std::chrono::high_resolution_clock::now();
        kdtree_.setInputCloud(cloud);
        auto t1 = std::chrono::high_resolution_clock::now();
        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << " done (" << (ms / 1000.0) << "s)" << std::endl;

        // Find bounds
        std::cout << "[PHASE 2/3] Computing bounds..." << std::flush;
        float min_x = 1e9f, min_y = 1e9f, min_z = 1e9f;
        float max_x = -1e9f, max_y = -1e9f, max_z = -1e9f;

        for (size_t i = 0; i < cloud->size(); ++i) {
            const auto& pt = cloud->points[i];
            if (!std::isfinite(pt.x)) continue;
            min_x = std::min(min_x, pt.x);
            min_y = std::min(min_y, pt.y);
            min_z = std::min(min_z, pt.z);
            max_x = std::max(max_x, pt.x);
            max_y = std::max(max_y, pt.y);
            max_z = std::max(max_z, pt.z);
        }

        origin_x_ = std::floor(min_x);
        origin_y_ = std::floor(min_y);
        origin_z_ = std::floor(min_z);

        std::cout << " done" << std::endl;
        std::cout << "[INFO] Cloud bounds: ("
                  << min_x << "," << min_y << "," << min_z << ") - ("
                  << max_x << "," << max_y << "," << max_z << ")" << std::endl;

        // Count points per cube (streaming, no index storage)
        std::cout << "[PHASE 2/3] Counting cubes: " << std::flush;
        std::unordered_map<int64_t, size_t> cube_counts;

        auto key = [&](int ix, int iy, int iz) -> int64_t {
            return int64_t(ix) * 1000000LL * 1000000LL + int64_t(iy) * 1000000LL + int64_t(iz);
        };

        size_t progress_step = cloud->size() / 20;
        if (progress_step == 0) progress_step = 1;

        for (size_t i = 0; i < cloud->size(); ++i) {
            const auto& pt = cloud->points[i];
            if (!std::isfinite(pt.x)) continue;

            int ix = int((pt.x - origin_x_) / cube_size_);
            int iy = int((pt.y - origin_y_) / cube_size_);
            int iz = int((pt.z - origin_z_) / cube_size_);

            cube_counts[key(ix, iy, iz)]++;

            if ((i + 1) % progress_step == 0) {
                int pct = static_cast<int>((i + 1) * 100 / cloud->size());
                std::cout << "\r[PHASE 2/3] Counting cubes: " << pct << "% (" 
                          << cube_counts.size() << " cubes found)" << std::flush;
            }
        }
        std::cout << std::endl;

        // Create cube info list (Dense Grid)
        int grid_nx = int((max_x - origin_x_) / cube_size_) + 1;
        int grid_ny = int((max_y - origin_y_) / cube_size_) + 1;
        int grid_nz = int((max_z - origin_z_) / cube_size_) + 1;

        for (int iz = 0; iz < grid_nz; ++iz) {
            for (int iy = 0; iy < grid_ny; ++iy) {
                for (int ix = 0; ix < grid_nx; ++ix) {
                    int64_t k = key(ix, iy, iz);
                    size_t count = cube_counts.count(k) ? cube_counts[k] : 0;

                    CubeInfo info;
                    info.ix = ix; info.iy = iy; info.iz = iz;
                    info.origin_x = origin_x_ + ix * cube_size_;
                    info.origin_y = origin_y_ + iy * cube_size_;
                    info.origin_z = origin_z_ + iz * cube_size_;
                    info.point_count = count;
                    cube_infos_.push_back(info);
                }
            }
        }

        std::cout << "[INFO] Created " << cube_infos_.size() << " cubes (" << cube_size_ << "m)" << std::endl;
        std::cout << "[PHASE 3/3] Training cubes..." << std::endl;
    }

    /// Get next cube with indices (streaming, extracts on demand)
    bool getNextCube(Cube& cube) {
        if (current_idx_ >= cube_infos_.size()) return false;

        const CubeInfo& info = cube_infos_[current_idx_++];
        cube.ix = info.ix;
        cube.iy = info.iy;
        cube.iz = info.iz;
        cube.origin_x = info.origin_x;
        cube.origin_y = info.origin_y;
        cube.origin_z = info.origin_z;
        cube.point_indices.clear();

        // Extract indices for this cube
        float cx = info.centerX();
        float cy = info.centerY();
        float cz = info.centerZ();

        // Use radius search to find points near cube center
        pcl::PointXYZ center;
        center.x = cx; center.y = cy; center.z = cz;

        float radius = std::sqrt(3.0f) * cube_size_ * 0.6f;
        std::vector<int> indices;
        std::vector<float> dists;
        kdtree_.radiusSearch(center, radius, indices, dists);

        // Filter to only points inside this cube
        for (int idx : indices) {
            const auto& pt = cloud_->points[idx];
            int ix = int((pt.x - origin_x_) / cube_size_);
            int iy = int((pt.y - origin_y_) / cube_size_);
            int iz = int((pt.z - origin_z_) / cube_size_);
            if (ix == info.ix && iy == info.iy && iz == info.iz) {
                cube.point_indices.push_back(idx);
            }
        }

        return true;
    }

    /// Reset iterator for another pass
    void resetIterator() { current_idx_ = 0; }

    /// Get total cube count
    size_t getCubeCount() const { return cube_infos_.size(); }

    /// Get current progress
    size_t getCurrentIndex() const { return current_idx_; }

    /// Get cube info list (read-only)
    const std::vector<CubeInfo>& getCubeInfos() const { return cube_infos_; }

    const pcl::KdTreeFLANN<pcl::PointXYZ>& getKdTree() const { return kdtree_; }
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& getCloud() const { return cloud_; }
    float getCubeSize() const { return cube_size_; }
    float getOriginX() const { return origin_x_; }
    float getOriginY() const { return origin_y_; }
    float getOriginZ() const { return origin_z_; }

private:
    float cube_size_;
    float origin_x_, origin_y_, origin_z_;
    std::vector<CubeInfo> cube_infos_;
    size_t current_idx_;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
};

#endif // CUBE_MANAGER_HPP
