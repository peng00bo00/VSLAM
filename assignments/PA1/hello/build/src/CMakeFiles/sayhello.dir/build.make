# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pengbo/VSLAM/assignments/PA1/hello

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pengbo/VSLAM/assignments/PA1/hello/build

# Include any dependencies generated for this target.
include src/CMakeFiles/sayhello.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/sayhello.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/sayhello.dir/flags.make

src/CMakeFiles/sayhello.dir/useHello.cpp.o: src/CMakeFiles/sayhello.dir/flags.make
src/CMakeFiles/sayhello.dir/useHello.cpp.o: ../src/useHello.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pengbo/VSLAM/assignments/PA1/hello/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/sayhello.dir/useHello.cpp.o"
	cd /home/pengbo/VSLAM/assignments/PA1/hello/build/src && /home/pengbo/anaconda3/bin/x86_64-conda_cos6-linux-gnu-c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sayhello.dir/useHello.cpp.o -c /home/pengbo/VSLAM/assignments/PA1/hello/src/useHello.cpp

src/CMakeFiles/sayhello.dir/useHello.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sayhello.dir/useHello.cpp.i"
	cd /home/pengbo/VSLAM/assignments/PA1/hello/build/src && /home/pengbo/anaconda3/bin/x86_64-conda_cos6-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pengbo/VSLAM/assignments/PA1/hello/src/useHello.cpp > CMakeFiles/sayhello.dir/useHello.cpp.i

src/CMakeFiles/sayhello.dir/useHello.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sayhello.dir/useHello.cpp.s"
	cd /home/pengbo/VSLAM/assignments/PA1/hello/build/src && /home/pengbo/anaconda3/bin/x86_64-conda_cos6-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pengbo/VSLAM/assignments/PA1/hello/src/useHello.cpp -o CMakeFiles/sayhello.dir/useHello.cpp.s

src/CMakeFiles/sayhello.dir/useHello.cpp.o.requires:

.PHONY : src/CMakeFiles/sayhello.dir/useHello.cpp.o.requires

src/CMakeFiles/sayhello.dir/useHello.cpp.o.provides: src/CMakeFiles/sayhello.dir/useHello.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/sayhello.dir/build.make src/CMakeFiles/sayhello.dir/useHello.cpp.o.provides.build
.PHONY : src/CMakeFiles/sayhello.dir/useHello.cpp.o.provides

src/CMakeFiles/sayhello.dir/useHello.cpp.o.provides.build: src/CMakeFiles/sayhello.dir/useHello.cpp.o


# Object files for target sayhello
sayhello_OBJECTS = \
"CMakeFiles/sayhello.dir/useHello.cpp.o"

# External object files for target sayhello
sayhello_EXTERNAL_OBJECTS =

src/sayhello: src/CMakeFiles/sayhello.dir/useHello.cpp.o
src/sayhello: src/CMakeFiles/sayhello.dir/build.make
src/sayhello: src/libhello.so
src/sayhello: src/CMakeFiles/sayhello.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pengbo/VSLAM/assignments/PA1/hello/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sayhello"
	cd /home/pengbo/VSLAM/assignments/PA1/hello/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sayhello.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/sayhello.dir/build: src/sayhello

.PHONY : src/CMakeFiles/sayhello.dir/build

src/CMakeFiles/sayhello.dir/requires: src/CMakeFiles/sayhello.dir/useHello.cpp.o.requires

.PHONY : src/CMakeFiles/sayhello.dir/requires

src/CMakeFiles/sayhello.dir/clean:
	cd /home/pengbo/VSLAM/assignments/PA1/hello/build/src && $(CMAKE_COMMAND) -P CMakeFiles/sayhello.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/sayhello.dir/clean

src/CMakeFiles/sayhello.dir/depend:
	cd /home/pengbo/VSLAM/assignments/PA1/hello/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pengbo/VSLAM/assignments/PA1/hello /home/pengbo/VSLAM/assignments/PA1/hello/src /home/pengbo/VSLAM/assignments/PA1/hello/build /home/pengbo/VSLAM/assignments/PA1/hello/build/src /home/pengbo/VSLAM/assignments/PA1/hello/build/src/CMakeFiles/sayhello.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/sayhello.dir/depend

