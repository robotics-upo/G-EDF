cmake_minimum_required(VERSION 3.14)
project(gaussian_trainer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Opciones de optimización
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# =====================================================
# Dependencias
# =====================================================

# PCL (Point Cloud Library)
find_package(PCL REQUIRED COMPONENTS common io kdtree filters)

# Ceres Solver (para optimización de Gaussianas)
find_package(Ceres REQUIRED)

# OpenMP (para paralelismo)
find_package(OpenMP REQUIRED)

# =====================================================
# Ejecutable principal
# =====================================================

add_executable(gaussian_trainer 
    src/main.cpp
)

add_executable(gaussian_to_ply
    src/gaussian_to_ply.cpp
)

target_include_directories(gaussian_trainer PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(gaussian_trainer 
    ${PCL_LIBRARIES}
    Ceres::ceres
    OpenMP::OpenMP_CXX
)

# gaussian_to_ply only needs standard lib
target_include_directories(gaussian_to_ply PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PCL_INCLUDE_DIRS}
)

target_link_libraries(gaussian_to_ply
    ${PCL_LIBRARIES}
)

# =====================================================
# Mensaje de configuración
# =====================================================

message(STATUS "")
message(STATUS "=======================================")
message(STATUS " GAUSSIAN TRAINER - Configuración")
message(STATUS "=======================================")
message(STATUS " PCL Version: ${PCL_VERSION}")
message(STATUS " Ceres Found: ${Ceres_FOUND}")
message(STATUS " OpenMP Found: ${OpenMP_FOUND}")
message(STATUS "=======================================")
message(STATUS "")