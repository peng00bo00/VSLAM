INCLUDE_DIRECTORIES(../include)

ADD_LIBRARY(hello SHARED hello.cpp)
SET(CMAKE_BUILD_TYPE "Release")

ADD_EXECUTABLE(sayhello useHello.cpp)

TARGET_LINK_LIBRARIES(sayhello hello)

INSTALL(TARGETS hello LIBRARY DESTINATION /usr/local/lib)