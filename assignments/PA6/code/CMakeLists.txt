cmake_minimum_required(VERSION 2.8)

project(EX6)

find_package( Pangolin REQUIRED )
find_package( OpenCV REQUIRED )
find_package( Sophus REQUIRED )

include_directories("/usr/include/eigen3" ${Pangolin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS})

add_executable(optical_flow optical_flow.cpp)
add_executable(direct_method direct_method.cpp)
add_executable(disparity disparity.cpp)

target_link_libraries( optical_flow ${OpenCV_LIBS} )
target_link_libraries( direct_method ${OpenCV_LIBS} ${Sophus_INCLUDE_DIRS} ${Pangolin_LIBRARIES})
target_link_libraries( disparity ${OpenCV_LIBS} )
