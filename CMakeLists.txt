cmake_minimum_required(VERSION 3.8)
project(db_tsdf)

cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0069 NEW) 

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckIPOSupported)
check_ipo_supported(RESULT result OUTPUT output)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    -Ofast
    -ffast-math              
    -march=native       
    -DNDEBUG            
    -fno-math-errno     
    -funroll-loops      
    -fopenmp           
  )
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters)
find_package(Boost REQUIRED COMPONENTS thread chrono)
find_package(Ceres REQUIRED)
find_package(OpenMP REQUIRED)
find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(std_srvs REQUIRED)
find_library(ANN_LIB ANN PATHS /usr/lib /usr/local/lib)

find_package(VTK REQUIRED COMPONENTS 
  CommonCore
  CommonDataModel
  FiltersCore
  IOXML
  IOGeometry
  ImagingCore
  ImagingGeneral
  ImagingMorphological
)

add_executable(db_tsdf_node  src/db_tsdf_node.cpp)

ament_target_dependencies(db_tsdf_node
  rclcpp
  tf2            tf2_ros
  tf2_geometry_msgs
  sensor_msgs    nav_msgs
  pcl_ros        pcl_conversions
  std_srvs
  Boost
)

target_link_libraries(db_tsdf_node
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  OpenMP::OpenMP_CXX
  VTK::CommonCore
  VTK::CommonDataModel
  VTK::FiltersCore
  VTK::IOXML
  VTK::IOGeometry
  VTK::ImagingCore
  VTK::ImagingGeneral
  Ceres::ceres
  glog::glog
)

target_include_directories(db_tsdf_node PRIVATE
  include
  ${Boost_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
)

install(DIRECTORY 
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

add_executable(gmm_solver src/main.cpp)

target_include_directories(gmm_solver PRIVATE 
  include
  ${EIGEN3_INCLUDE_DIR} 
  ${VTK_INCLUDE_DIRS}
)

target_link_libraries(gmm_solver 
  Ceres::ceres 
  glog::glog
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  VTK::CommonCore       
  VTK::CommonDataModel 
  VTK::FiltersCore     
  VTK::IOXML            
  VTK::IOGeometry
)

add_executable(reconstruct_mesh src/reconstruct.cpp)
target_link_libraries(reconstruct_mesh ${VTK_LIBRARIES} OpenMP::OpenMP_CXX)

# Activar Link Time Optimization (LTO) para el solver
if(result)
  set_property(TARGET gmm_solver PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(TARGETS
  db_tsdf_node
  gmm_solver
  reconstruct_mesh
  DESTINATION lib/${PROJECT_NAME}
)


ament_package()