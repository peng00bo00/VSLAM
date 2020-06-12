cmake_minimum_required(VERSION 2.8)

project(EX5)

find_package( Pangolin REQUIRED )
find_package( OpenCV REQUIRED )
find_package( Sophus REQUIRED )

include_directories("/usr/include/eigen3" ${Pangolin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS})
# include_directories("/usr/include/eigen3")

add_executable(computeORB computeORB.cpp)
add_executable(E2Rt E2Rt.cpp)
add_executable(GN-BA GN-BA.cpp)
add_executable(ICP ICP.cpp)

target_link_libraries( computeORB ${OpenCV_LIBS} )
target_link_libraries( E2Rt ${Sophus_INCLUDE_DIRS} )
target_link_libraries( GN-BA ${Sophus_INCLUDE_DIRS} )
target_link_libraries( ICP ${Sophus_INCLUDE_DIRS} ${Pangolin_LIBRARIES})