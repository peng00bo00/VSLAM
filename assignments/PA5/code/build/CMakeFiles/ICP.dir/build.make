# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/pengbo/VSLAM/assignments/PA5/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pengbo/VSLAM/assignments/PA5/code/build

# Include any dependencies generated for this target.
include CMakeFiles/ICP.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ICP.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ICP.dir/flags.make

CMakeFiles/ICP.dir/ICP.cpp.o: CMakeFiles/ICP.dir/flags.make
CMakeFiles/ICP.dir/ICP.cpp.o: ../ICP.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pengbo/VSLAM/assignments/PA5/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ICP.dir/ICP.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ICP.dir/ICP.cpp.o -c /home/pengbo/VSLAM/assignments/PA5/code/ICP.cpp

CMakeFiles/ICP.dir/ICP.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ICP.dir/ICP.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pengbo/VSLAM/assignments/PA5/code/ICP.cpp > CMakeFiles/ICP.dir/ICP.cpp.i

CMakeFiles/ICP.dir/ICP.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ICP.dir/ICP.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pengbo/VSLAM/assignments/PA5/code/ICP.cpp -o CMakeFiles/ICP.dir/ICP.cpp.s

# Object files for target ICP
ICP_OBJECTS = \
"CMakeFiles/ICP.dir/ICP.cpp.o"

# External object files for target ICP
ICP_EXTERNAL_OBJECTS =

ICP: CMakeFiles/ICP.dir/ICP.cpp.o
ICP: CMakeFiles/ICP.dir/build.make
ICP: /home/pengbo/Pangolin/build/src/libpangolin.so
ICP: /usr/lib/x86_64-linux-gnu/libOpenGL.so
ICP: /usr/lib/x86_64-linux-gnu/libGLX.so
ICP: /usr/lib/x86_64-linux-gnu/libGLU.so
ICP: /usr/lib/x86_64-linux-gnu/libGLEW.so
ICP: /usr/lib/x86_64-linux-gnu/libEGL.so
ICP: /usr/lib/x86_64-linux-gnu/libX11.so
ICP: /usr/lib/x86_64-linux-gnu/libOpenGL.so
ICP: /usr/lib/x86_64-linux-gnu/libGLX.so
ICP: /usr/lib/x86_64-linux-gnu/libGLU.so
ICP: /usr/lib/x86_64-linux-gnu/libGLEW.so
ICP: /usr/lib/x86_64-linux-gnu/libEGL.so
ICP: /usr/lib/x86_64-linux-gnu/libX11.so
ICP: /usr/lib/x86_64-linux-gnu/libdc1394.so
ICP: /usr/lib/x86_64-linux-gnu/libavcodec.so
ICP: /usr/lib/x86_64-linux-gnu/libavformat.so
ICP: /usr/lib/x86_64-linux-gnu/libavutil.so
ICP: /usr/lib/x86_64-linux-gnu/libswscale.so
ICP: /usr/lib/x86_64-linux-gnu/libavdevice.so
ICP: CMakeFiles/ICP.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pengbo/VSLAM/assignments/PA5/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ICP"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ICP.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ICP.dir/build: ICP

.PHONY : CMakeFiles/ICP.dir/build

CMakeFiles/ICP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ICP.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ICP.dir/clean

CMakeFiles/ICP.dir/depend:
	cd /home/pengbo/VSLAM/assignments/PA5/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pengbo/VSLAM/assignments/PA5/code /home/pengbo/VSLAM/assignments/PA5/code /home/pengbo/VSLAM/assignments/PA5/code/build /home/pengbo/VSLAM/assignments/PA5/code/build /home/pengbo/VSLAM/assignments/PA5/code/build/CMakeFiles/ICP.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ICP.dir/depend

