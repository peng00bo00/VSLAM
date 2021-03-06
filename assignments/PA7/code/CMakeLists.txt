cmake_minimum_required(VERSION 2.8)

project(EX7)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

Find_Package(G2O REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Sophus REQUIRED)
Find_Package(CSparse REQUIRED)
Find_Package(OpenCV REQUIRED)
Find_Package(Pangolin REQUIRED)

SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse g2o_types_sba)

include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR} ${Pangolin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_library(bal_common common.cpp)
add_executable(bundle_adjustment_g2o bundle_adjustment_g2o.cpp)
add_executable(directBA directBA.cpp)

target_link_libraries(bundle_adjustment_g2o ${G2O_LIBS} bal_common)
target_link_libraries(directBA ${G2O_LIBS} ${Pangolin_LIBRARIES} ${OpenCV_LIBS})
