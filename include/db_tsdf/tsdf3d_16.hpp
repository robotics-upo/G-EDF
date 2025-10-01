#ifndef __TSDF3D_16_HPP__
#define __TSDF3D_16_HPP__

#include <algorithm>  
#include <bitset>
#include <db_tsdf/df3d.hpp>
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <boost/thread.hpp>
#include <boost/chrono.hpp>
#include "rclcpp/clock.hpp"
#include <db_tsdf/grid_16.hpp>

#include <Eigen/Dense>	
#include <iostream>



// DIRECTIONAL KERNELS
constexpr int BINS_AZ  = 40;		// azimuth bins               
constexpr int BINS_EL  = 40;		// elevation bins
constexpr int NUM_BINS = BINS_AZ * BINS_EL;  
constexpr int OCC_MIN_HITS = 50;		// hits threshold to mark occupied
constexpr int SHADOW_RADIUS_MD = 5; 	// shadow radius (voxel units)
struct DirectionalKernel
{
    uint16_t distance_masks[11*11*11];		// Manhattan distance masks
    uint8_t  signs[11*11*11];		        // 0 = occ, 1 = free
};


class TSDF3D16
{
	
public:

	TSDF3D16(void) 
	{
		m_maxX = 40;
		m_maxY = 40;
		m_maxZ = 40;
		m_minX = -40;
		m_minY = -40;
		m_minZ = -10;
		m_resolution = 0.05;
		m_oneDivRes = 1/m_resolution;
		
		initDirectionalKernels();
	}

	~TSDF3D16(void)
	{
	}

	void setup(float minX, float maxX, float minY, float maxY, float minZ, float maxZ, float resolution, int maxCells = 50000)
	{
		m_maxX = maxX;
		m_maxY = maxY;
		m_maxZ = maxZ;
		m_minX = minX;
		m_minY = minY;
		m_minZ = minZ;
		m_resolution = resolution;
		m_oneDivRes = 1/m_resolution;

		// Setup grid
		m_grid.setup(m_minX, m_maxX, m_minY, m_maxY, m_minZ, m_maxZ, m_resolution, maxCells);
	}

	void clear(void)
	{
		m_grid.clear();
	}

	void exportGridToPCD(const std::string& filename, int subsampling_factor)
	{

		m_grid.exportGridToPCD(filename, subsampling_factor);
	}

	void exportGridToPLY(const std::string& filename, int subsampling_factor)
	{

		m_grid.exportGridToPLY(filename, subsampling_factor);
	}

	virtual inline bool isIntoGrid(const float &x, const float &y, const float &z)
	{
		return (x > m_minX+1 && y > m_minY+1 && z > m_minZ+1 && x < m_maxX-1 && y < m_maxY-1 && z < m_maxZ-1);
	}


	void loadCloud(std::vector<pcl::PointXYZ> &cloud, float tx, float ty, float tz, float yaw)
	{
		std::vector<pcl::PointXYZ> out;

		float c = cos(yaw);
		float s = sin(yaw);
		out.resize(cloud.size());
		for(uint32_t i=0; i<out.size(); i++)
		{
			out[i].x = c*cloud[i].x - s*cloud[i].y + tx;
			out[i].y = s*cloud[i].x + c*cloud[i].y + ty; 
			out[i].z = cloud[i].z + tz;
		}
		loadCloud(out);
	}

	void loadCloud(std::vector<pcl::PointXYZ> &cloud, float tx, float ty, float tz, float roll, float pitch, float yaw)
	{
		std::vector<pcl::PointXYZ> out;

		// Get rotation matrix
		float cr, sr, cp, sp, cy, sy;
		float r00, r01, r02, r10, r11, r12, r20, r21, r22;
		sr = sin(roll);
		cr = cos(roll);
		sp = sin(pitch);
		cp = cos(pitch);
		sy = sin(yaw);
		cy = cos(yaw);
		r00 = cy*cp; 	r01 = cy*sp*sr-sy*cr; 	r02 = cy*sp*cr+sy*sr;
		r10 = sy*cp; 	r11 = sy*sp*sr+cy*cr;	r12 = sy*sp*cr-cy*sr;
		r20 = -sp;		r21 = cp*sr;			r22 = cp*cr;

		// Tiltcompensate points 
		out.resize(cloud.size());
		for(uint i=0; i<cloud.size(); i++) 
		{
			out[i].x = cloud[i].x*r00 + cloud[i].y*r01 + cloud[i].z*r02 + tx;
			out[i].y = cloud[i].x*r10 + cloud[i].y*r11 + cloud[i].z*r12 + ty;
			out[i].z = cloud[i].x*r20 + cloud[i].y*r21 + cloud[i].z*r22 + tz;	
		}
		loadCloud(out);
	}

	void loadCloudFiltered(std::vector<pcl::PointXYZ> &cloud,
                       float tx, float ty, float tz,
                       float roll, float pitch, float yaw,
                       float maxAbs)
	{
		std::vector<pcl::PointXYZ> out;
		out.reserve(cloud.size()); // evitar realocaciones

		// Matriz de rotación precalculada (radianes)
		const float sr = std::sin(roll),  cr = std::cos(roll);
		const float sp = std::sin(pitch), cp = std::cos(pitch);
		const float sy = std::sin(yaw),   cy = std::cos(yaw);

		const float r00 =  cy*cp;           const float r01 =  cy*sp*sr - sy*cr;  const float r02 =  cy*sp*cr + sy*sr;
		const float r10 =  sy*cp;           const float r11 =  sy*sp*sr + cy*cr;  const float r12 =  sy*sp*cr - cy*sr;
		const float r20 = -sp;              const float r21 =  cp*sr;              const float r22 =  cp*cr;

		for (std::size_t i = 0; i < cloud.size(); ++i)
		{
			const float ix = cloud[i].x;
			const float iy = cloud[i].y;
			const float iz = cloud[i].z;

			// Sólo calcular x', y' para el filtro rápido
			const float xw = ix*r00 + iy*r01 + iz*r02 + tx;
			const float yw = ix*r10 + iy*r11 + iz*r12 + ty;

			// Filtro por ventana rectangular en XY (Chebyshev / AABB)
			if (std::fabs(xw) <= maxAbs && std::fabs(yw) <= maxAbs)
			{
				// Calcula z' sólo si el punto pasa el filtro
				const float zw = ix*r20 + iy*r21 + iz*r22 + tz;

				pcl::PointXYZ p;
				p.x = xw; p.y = yw; p.z = zw;
				out.push_back(p);
			}
		}
		loadCloud(out);
	}


	void loadCloud(std::vector<pcl::PointXYZ> &cloud)
	{	
		// Allocate required submetric cells
		for(uint32_t i=0; i<cloud.size(); i++)
		{
			for(float z=cloud[i].z-1; z<=cloud[i].z+1; z+=1)
				for(float y=cloud[i].y-1; y<=cloud[i].y+1; y+=1)
					for(float x=cloud[i].x-1; x<=cloud[i].x+1; x+=1)
						if(isIntoGrid(x, y, z))
							m_grid.allocCell(x, y, z);
		}

		// Applies the pre-computed kernel to all grid cells centered in the cloud points 
		const float step = 10*m_resolution;
		#pragma omp parallel for num_threads(10) shared(m_dirKernels, m_grid) 
		for(uint32_t i=0; i<cloud.size(); i++)
		{
			
			if(!isIntoGrid(cloud[i].x-step, cloud[i].y-step, cloud[i].z-step) || 
			   !isIntoGrid(cloud[i].x+step, cloud[i].y+step, cloud[i].z+step))
				continue;

			// Select kernel by ray direction
			Eigen::Vector3f dir(cloud[i].x, cloud[i].y, cloud[i].z);
			const DirectionalKernel& DK = m_dirKernels[dirToBin(dir)];

			int xi, yi, zi, k = 0;
			float x, y, z;
			for(zi=0, z=cloud[i].z-step; zi<11; zi++, z+=m_resolution)
				for(yi=0, y=cloud[i].y-step; yi<11; yi++, y+=m_resolution){
						GRID16::Iterator it = m_grid.getIterator(cloud[i].x-step,y,z);
					for(xi=0, x=cloud[i].x-step; xi<11; xi++, x+=m_resolution,++it, ++k){
						// *it &= kernel[k++];
						
						VoxelData &v = *it;
						uint16_t old_mask = v.d;
						uint16_t new_mask = old_mask & DK.distance_masks[k];
						if (new_mask != old_mask) v.d = new_mask;

						if(DK.signs[k] == 0 && v.hits < OCC_MIN_HITS) 
						{
							++v.hits;
							if (v.hits == OCC_MIN_HITS)
								v.s &= uint8_t(~0x01);
						}
					}
				}
		}
		#pragma omp barrier
	}

	inline TrilinearParams computeDistInterpolation(const double &x, const double &y, const double &z)
	{
		TrilinearParams r;

		if(isIntoGrid(x, y, z))
		{

			// Get neightbour values to compute trilinear interpolation
			float c000, c001, c010, c011, c100, c101, c110, c111;
			c000 = std::bitset<16>(m_grid.read(x, y, z).d).count(); 
			c001 = std::bitset<16>(m_grid.read(x, y, z+m_resolution).d).count(); 
			c010 = std::bitset<16>(m_grid.read(x, y+m_resolution, z).d).count(); 
			c011 = std::bitset<16>(m_grid.read(x, y+m_resolution, z+m_resolution).d).count();  
			c100 = std::bitset<16>(m_grid.read(x+m_resolution, y, z).d).count();  
			c101 = std::bitset<16>(m_grid.read(x+m_resolution, y, z+m_resolution).d).count();  
			c110 = std::bitset<16>(m_grid.read(x+m_resolution, y+m_resolution, z).d).count();  
			c111 = std::bitset<16>(m_grid.read(x+m_resolution, y+m_resolution, z+m_resolution).d).count(); 

			// Compute trilinear parameters
			const float div = -m_oneDivRes*m_oneDivRes*m_oneDivRes;
			float x0, y0, z0, x1, y1, z1;
			x0 = ((int)(x*m_oneDivRes))*m_resolution;
			if(x0<0)
				x0 -= m_resolution; 
			x1 = x0+m_resolution;
			y0 = ((int)(y*m_oneDivRes))*m_resolution;
			if(y0<0)
				y0 -= m_resolution;
			y1 = y0+m_resolution;
			z0 = ((int)(z*m_oneDivRes))*m_resolution;
			if(z0<0)
				z0 -= m_resolution;
			z1 = z0+m_resolution;
			r.a0 = (-c000*x1*y1*z1 + c001*x1*y1*z0 + c010*x1*y0*z1 - c011*x1*y0*z0 
			+ c100*x0*y1*z1 - c101*x0*y1*z0 - c110*x0*y0*z1 + c111*x0*y0*z0)*div;
			r.a1 = (c000*y1*z1 - c001*y1*z0 - c010*y0*z1 + c011*y0*z0
			- c100*y1*z1 + c101*y1*z0 + c110*y0*z1 - c111*y0*z0)*div;
			r.a2 = (c000*x1*z1 - c001*x1*z0 - c010*x1*z1 + c011*x1*z0 
			- c100*x0*z1 + c101*x0*z0 + c110*x0*z1 - c111*x0*z0)*div;
			r.a3 = (c000*x1*y1 - c001*x1*y1 - c010*x1*y0 + c011*x1*y0 
			- c100*x0*y1 + c101*x0*y1 + c110*x0*y0 - c111*x0*y0)*div;
			r.a4 = (-c000*z1 + c001*z0 + c010*z1 - c011*z0 + c100*z1 
			- c101*z0 - c110*z1 + c111*z0)*div;
			r.a5 = (-c000*y1 + c001*y1 + c010*y0 - c011*y0 + c100*y1 
			- c101*y1 - c110*y0 + c111*y0)*div;
			r.a6 = (-c000*x1 + c001*x1 + c010*x1 - c011*x1 + c100*x0 
			- c101*x0 - c110*x0 + c111*x0)*div;
			r.a7 = (c000 - c001 - c010 + c011 - c100
			+ c101 + c110 - c111)*div;
		}

		return r;
	}


protected:

	// Grid parameters
	float m_maxX, m_maxY, m_maxZ;
	float m_minX, m_minY, m_minZ;
	float m_resolution, m_oneDivRes;	
	// uint64_t kernel[41*41*41];
	GRID16 m_grid;

	// Directional kernels
	std::vector<DirectionalKernel> m_dirKernels;
	inline int dirToBin(const Eigen::Vector3f &v) const;
	void initDirectionalKernels();

};



// Map direction to bin index (azimuth × elevation)
inline int TSDF3D16::dirToBin(const Eigen::Vector3f &v) const
{
	float az = std::atan2(v.y(), v.x());
	if (az < 0) 
	{
		az += 2.0f*M_PI;
	}

	float el  = std::asin(v.z() / v.norm());
	int   baz = int(az * BINS_AZ / (2.0f*M_PI));
	int   bel = int((el + M_PI/2) * BINS_EL / M_PI);

	bel = std::clamp(bel, 0, BINS_EL-1);
	baz = std::clamp(baz, 0, BINS_AZ-1);

	return bel*BINS_AZ + baz;          
}


// Build 40×40 directional kernels (21³ window per bin)
inline void TSDF3D16::initDirectionalKernels()
{
	m_dirKernels.resize(NUM_BINS);

	auto binToDir = [](int az,int el)
	{			 
		float azr = (az+0.5f)*2.0f*M_PI/BINS_AZ;
		float elr = (-M_PI/2)+ (el+0.5f)*M_PI/BINS_EL;
		float c   = cosf(elr);
		return Eigen::Vector3f(c*cosf(azr), c*sinf(azr), sinf(elr));
	};

	for(int el=0; el<BINS_EL; ++el)
	for(int az=0; az<BINS_AZ; ++az)
	{
		DirectionalKernel& DK = m_dirKernels[el*BINS_AZ+az];
		Eigen::Vector3f dir = binToDir(az,el).normalized();

		int k=0;
		for(int z=-5;z<=5;++z)
		for(int y=-5;y<=5;++y)
		for(int x=-5;x<=5;++x,++k)
		{       
			constexpr float R_E = float(SHADOW_RADIUS_MD); 

			float re2 = float(x*x + y*y + z*z);            
			float re  = std::sqrt(re2);
			int   rd  = std::min(16, int(std::ceil(re)));   

			uint16_t mask = (rd == 0) ? 0u : (0xFFFFu >> (16 - rd));
			DK.distance_masks[k] = mask;

			bool behind   = dir.dot(Eigen::Vector3f(x,y,z)) > 0.0f;     
			const bool at_hit = (x == 0 && y == 0 && z == 0);
			bool inShadow = behind && (re2 <= R_E*R_E);                 
			DK.signs[k]   = (at_hit || inShadow) ? 0 : 1; 
		}
	}
}

#endif
