# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_SOURCE_DIR = /home/tri/glui

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tri/glui/build

# Include any dependencies generated for this target.
include CMakeFiles/example6.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/example6.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/example6.dir/flags.make

CMakeFiles/example6.dir/example/example6.cpp.o: CMakeFiles/example6.dir/flags.make
CMakeFiles/example6.dir/example/example6.cpp.o: ../example/example6.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/tri/glui/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/example6.dir/example/example6.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/example6.dir/example/example6.cpp.o -c /home/tri/glui/example/example6.cpp

CMakeFiles/example6.dir/example/example6.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example6.dir/example/example6.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/tri/glui/example/example6.cpp > CMakeFiles/example6.dir/example/example6.cpp.i

CMakeFiles/example6.dir/example/example6.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example6.dir/example/example6.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/tri/glui/example/example6.cpp -o CMakeFiles/example6.dir/example/example6.cpp.s

CMakeFiles/example6.dir/example/example6.cpp.o.requires:
.PHONY : CMakeFiles/example6.dir/example/example6.cpp.o.requires

CMakeFiles/example6.dir/example/example6.cpp.o.provides: CMakeFiles/example6.dir/example/example6.cpp.o.requires
	$(MAKE) -f CMakeFiles/example6.dir/build.make CMakeFiles/example6.dir/example/example6.cpp.o.provides.build
.PHONY : CMakeFiles/example6.dir/example/example6.cpp.o.provides

CMakeFiles/example6.dir/example/example6.cpp.o.provides.build: CMakeFiles/example6.dir/example/example6.cpp.o

# Object files for target example6
example6_OBJECTS = \
"CMakeFiles/example6.dir/example/example6.cpp.o"

# External object files for target example6
example6_EXTERNAL_OBJECTS =

example6: CMakeFiles/example6.dir/example/example6.cpp.o
example6: CMakeFiles/example6.dir/build.make
example6: libglui_static.a
example6: /usr/lib/x86_64-linux-gnu/libglut.so
example6: /usr/lib/x86_64-linux-gnu/libXmu.so
example6: /usr/lib/x86_64-linux-gnu/libXi.so
example6: /usr/lib/x86_64-linux-gnu/libGLU.so
example6: /usr/lib/x86_64-linux-gnu/libGL.so
example6: /usr/lib/x86_64-linux-gnu/libSM.so
example6: /usr/lib/x86_64-linux-gnu/libICE.so
example6: /usr/lib/x86_64-linux-gnu/libX11.so
example6: /usr/lib/x86_64-linux-gnu/libXext.so
example6: CMakeFiles/example6.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable example6"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example6.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/example6.dir/build: example6
.PHONY : CMakeFiles/example6.dir/build

CMakeFiles/example6.dir/requires: CMakeFiles/example6.dir/example/example6.cpp.o.requires
.PHONY : CMakeFiles/example6.dir/requires

CMakeFiles/example6.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/example6.dir/cmake_clean.cmake
.PHONY : CMakeFiles/example6.dir/clean

CMakeFiles/example6.dir/depend:
	cd /home/tri/glui/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tri/glui /home/tri/glui /home/tri/glui/build /home/tri/glui/build /home/tri/glui/build/CMakeFiles/example6.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/example6.dir/depend
