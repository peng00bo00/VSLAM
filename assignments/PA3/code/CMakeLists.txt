cmake_minimum_required(VERSION 2.8)

project(EX3)

# 为使用 sophus，需要使用find_package命令找到它
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# Eigen
include_directories("/usr/include/eigen3")
add_executable(drawTrajectory draw_trajectory.cpp)
target_link_libraries(drawTrajectory Sophus::Sophus ${Pangolin_LIBRARIES})
