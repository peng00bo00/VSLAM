cmake_minimum_required(VERSION 2.8)

project(EX4)

find_package( Pangolin REQUIRED )
find_package( OpenCV REQUIRED )

include_directories("/usr/include/eigen3" ${Pangolin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
# include_directories("/usr/include/eigen3")

add_executable(undistort_image undistort_image.cpp)
add_executable(disparity disparity.cpp)
add_executable(gaussnewton gaussnewton.cpp)

target_link_libraries( undistort_image ${OpenCV_LIBS} ${Pangolin_LIBRARIES} )
target_link_libraries( disparity ${OpenCV_LIBS} ${Pangolin_LIBRARIES} )
target_link_libraries( gaussnewton ${OpenCV_LIBS} )