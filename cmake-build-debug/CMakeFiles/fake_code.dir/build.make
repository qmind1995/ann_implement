# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

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
CMAKE_COMMAND = /home/tri/Downloads/clion-2017.1/bin/cmake/bin/cmake

# The command to remove a file.
RM = /home/tri/Downloads/clion-2017.1/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tri/Desktop/ann_implement

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tri/Desktop/ann_implement/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/fake_code.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fake_code.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fake_code.dir/flags.make

CMakeFiles/fake_code.dir/main.cpp.o: CMakeFiles/fake_code.dir/flags.make
CMakeFiles/fake_code.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tri/Desktop/ann_implement/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fake_code.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fake_code.dir/main.cpp.o -c /home/tri/Desktop/ann_implement/main.cpp

CMakeFiles/fake_code.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fake_code.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tri/Desktop/ann_implement/main.cpp > CMakeFiles/fake_code.dir/main.cpp.i

CMakeFiles/fake_code.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fake_code.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tri/Desktop/ann_implement/main.cpp -o CMakeFiles/fake_code.dir/main.cpp.s

CMakeFiles/fake_code.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/fake_code.dir/main.cpp.o.requires

CMakeFiles/fake_code.dir/main.cpp.o.provides: CMakeFiles/fake_code.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/fake_code.dir/build.make CMakeFiles/fake_code.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/fake_code.dir/main.cpp.o.provides

CMakeFiles/fake_code.dir/main.cpp.o.provides.build: CMakeFiles/fake_code.dir/main.cpp.o


CMakeFiles/fake_code.dir/Trainer.cpp.o: CMakeFiles/fake_code.dir/flags.make
CMakeFiles/fake_code.dir/Trainer.cpp.o: ../Trainer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tri/Desktop/ann_implement/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/fake_code.dir/Trainer.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fake_code.dir/Trainer.cpp.o -c /home/tri/Desktop/ann_implement/Trainer.cpp

CMakeFiles/fake_code.dir/Trainer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fake_code.dir/Trainer.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tri/Desktop/ann_implement/Trainer.cpp > CMakeFiles/fake_code.dir/Trainer.cpp.i

CMakeFiles/fake_code.dir/Trainer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fake_code.dir/Trainer.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tri/Desktop/ann_implement/Trainer.cpp -o CMakeFiles/fake_code.dir/Trainer.cpp.s

CMakeFiles/fake_code.dir/Trainer.cpp.o.requires:

.PHONY : CMakeFiles/fake_code.dir/Trainer.cpp.o.requires

CMakeFiles/fake_code.dir/Trainer.cpp.o.provides: CMakeFiles/fake_code.dir/Trainer.cpp.o.requires
	$(MAKE) -f CMakeFiles/fake_code.dir/build.make CMakeFiles/fake_code.dir/Trainer.cpp.o.provides.build
.PHONY : CMakeFiles/fake_code.dir/Trainer.cpp.o.provides

CMakeFiles/fake_code.dir/Trainer.cpp.o.provides.build: CMakeFiles/fake_code.dir/Trainer.cpp.o


CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o: CMakeFiles/fake_code.dir/flags.make
CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o: ../NeuralNetwork.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tri/Desktop/ann_implement/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o -c /home/tri/Desktop/ann_implement/NeuralNetwork.cpp

CMakeFiles/fake_code.dir/NeuralNetwork.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fake_code.dir/NeuralNetwork.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tri/Desktop/ann_implement/NeuralNetwork.cpp > CMakeFiles/fake_code.dir/NeuralNetwork.cpp.i

CMakeFiles/fake_code.dir/NeuralNetwork.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fake_code.dir/NeuralNetwork.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tri/Desktop/ann_implement/NeuralNetwork.cpp -o CMakeFiles/fake_code.dir/NeuralNetwork.cpp.s

CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o.requires:

.PHONY : CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o.requires

CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o.provides: CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o.requires
	$(MAKE) -f CMakeFiles/fake_code.dir/build.make CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o.provides.build
.PHONY : CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o.provides

CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o.provides.build: CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o


CMakeFiles/fake_code.dir/DataReader.cpp.o: CMakeFiles/fake_code.dir/flags.make
CMakeFiles/fake_code.dir/DataReader.cpp.o: ../DataReader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tri/Desktop/ann_implement/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/fake_code.dir/DataReader.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fake_code.dir/DataReader.cpp.o -c /home/tri/Desktop/ann_implement/DataReader.cpp

CMakeFiles/fake_code.dir/DataReader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fake_code.dir/DataReader.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tri/Desktop/ann_implement/DataReader.cpp > CMakeFiles/fake_code.dir/DataReader.cpp.i

CMakeFiles/fake_code.dir/DataReader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fake_code.dir/DataReader.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tri/Desktop/ann_implement/DataReader.cpp -o CMakeFiles/fake_code.dir/DataReader.cpp.s

CMakeFiles/fake_code.dir/DataReader.cpp.o.requires:

.PHONY : CMakeFiles/fake_code.dir/DataReader.cpp.o.requires

CMakeFiles/fake_code.dir/DataReader.cpp.o.provides: CMakeFiles/fake_code.dir/DataReader.cpp.o.requires
	$(MAKE) -f CMakeFiles/fake_code.dir/build.make CMakeFiles/fake_code.dir/DataReader.cpp.o.provides.build
.PHONY : CMakeFiles/fake_code.dir/DataReader.cpp.o.provides

CMakeFiles/fake_code.dir/DataReader.cpp.o.provides.build: CMakeFiles/fake_code.dir/DataReader.cpp.o


CMakeFiles/fake_code.dir/DataGenerator.cpp.o: CMakeFiles/fake_code.dir/flags.make
CMakeFiles/fake_code.dir/DataGenerator.cpp.o: ../DataGenerator.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tri/Desktop/ann_implement/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/fake_code.dir/DataGenerator.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fake_code.dir/DataGenerator.cpp.o -c /home/tri/Desktop/ann_implement/DataGenerator.cpp

CMakeFiles/fake_code.dir/DataGenerator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fake_code.dir/DataGenerator.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tri/Desktop/ann_implement/DataGenerator.cpp > CMakeFiles/fake_code.dir/DataGenerator.cpp.i

CMakeFiles/fake_code.dir/DataGenerator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fake_code.dir/DataGenerator.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tri/Desktop/ann_implement/DataGenerator.cpp -o CMakeFiles/fake_code.dir/DataGenerator.cpp.s

CMakeFiles/fake_code.dir/DataGenerator.cpp.o.requires:

.PHONY : CMakeFiles/fake_code.dir/DataGenerator.cpp.o.requires

CMakeFiles/fake_code.dir/DataGenerator.cpp.o.provides: CMakeFiles/fake_code.dir/DataGenerator.cpp.o.requires
	$(MAKE) -f CMakeFiles/fake_code.dir/build.make CMakeFiles/fake_code.dir/DataGenerator.cpp.o.provides.build
.PHONY : CMakeFiles/fake_code.dir/DataGenerator.cpp.o.provides

CMakeFiles/fake_code.dir/DataGenerator.cpp.o.provides.build: CMakeFiles/fake_code.dir/DataGenerator.cpp.o


CMakeFiles/fake_code.dir/HOGFeature.cpp.o: CMakeFiles/fake_code.dir/flags.make
CMakeFiles/fake_code.dir/HOGFeature.cpp.o: ../HOGFeature.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tri/Desktop/ann_implement/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/fake_code.dir/HOGFeature.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fake_code.dir/HOGFeature.cpp.o -c /home/tri/Desktop/ann_implement/HOGFeature.cpp

CMakeFiles/fake_code.dir/HOGFeature.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fake_code.dir/HOGFeature.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tri/Desktop/ann_implement/HOGFeature.cpp > CMakeFiles/fake_code.dir/HOGFeature.cpp.i

CMakeFiles/fake_code.dir/HOGFeature.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fake_code.dir/HOGFeature.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tri/Desktop/ann_implement/HOGFeature.cpp -o CMakeFiles/fake_code.dir/HOGFeature.cpp.s

CMakeFiles/fake_code.dir/HOGFeature.cpp.o.requires:

.PHONY : CMakeFiles/fake_code.dir/HOGFeature.cpp.o.requires

CMakeFiles/fake_code.dir/HOGFeature.cpp.o.provides: CMakeFiles/fake_code.dir/HOGFeature.cpp.o.requires
	$(MAKE) -f CMakeFiles/fake_code.dir/build.make CMakeFiles/fake_code.dir/HOGFeature.cpp.o.provides.build
.PHONY : CMakeFiles/fake_code.dir/HOGFeature.cpp.o.provides

CMakeFiles/fake_code.dir/HOGFeature.cpp.o.provides.build: CMakeFiles/fake_code.dir/HOGFeature.cpp.o


CMakeFiles/fake_code.dir/Layer.cpp.o: CMakeFiles/fake_code.dir/flags.make
CMakeFiles/fake_code.dir/Layer.cpp.o: ../Layer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tri/Desktop/ann_implement/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/fake_code.dir/Layer.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fake_code.dir/Layer.cpp.o -c /home/tri/Desktop/ann_implement/Layer.cpp

CMakeFiles/fake_code.dir/Layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fake_code.dir/Layer.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tri/Desktop/ann_implement/Layer.cpp > CMakeFiles/fake_code.dir/Layer.cpp.i

CMakeFiles/fake_code.dir/Layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fake_code.dir/Layer.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tri/Desktop/ann_implement/Layer.cpp -o CMakeFiles/fake_code.dir/Layer.cpp.s

CMakeFiles/fake_code.dir/Layer.cpp.o.requires:

.PHONY : CMakeFiles/fake_code.dir/Layer.cpp.o.requires

CMakeFiles/fake_code.dir/Layer.cpp.o.provides: CMakeFiles/fake_code.dir/Layer.cpp.o.requires
	$(MAKE) -f CMakeFiles/fake_code.dir/build.make CMakeFiles/fake_code.dir/Layer.cpp.o.provides.build
.PHONY : CMakeFiles/fake_code.dir/Layer.cpp.o.provides

CMakeFiles/fake_code.dir/Layer.cpp.o.provides.build: CMakeFiles/fake_code.dir/Layer.cpp.o


CMakeFiles/fake_code.dir/Utils.cpp.o: CMakeFiles/fake_code.dir/flags.make
CMakeFiles/fake_code.dir/Utils.cpp.o: ../Utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tri/Desktop/ann_implement/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/fake_code.dir/Utils.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fake_code.dir/Utils.cpp.o -c /home/tri/Desktop/ann_implement/Utils.cpp

CMakeFiles/fake_code.dir/Utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fake_code.dir/Utils.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tri/Desktop/ann_implement/Utils.cpp > CMakeFiles/fake_code.dir/Utils.cpp.i

CMakeFiles/fake_code.dir/Utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fake_code.dir/Utils.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tri/Desktop/ann_implement/Utils.cpp -o CMakeFiles/fake_code.dir/Utils.cpp.s

CMakeFiles/fake_code.dir/Utils.cpp.o.requires:

.PHONY : CMakeFiles/fake_code.dir/Utils.cpp.o.requires

CMakeFiles/fake_code.dir/Utils.cpp.o.provides: CMakeFiles/fake_code.dir/Utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/fake_code.dir/build.make CMakeFiles/fake_code.dir/Utils.cpp.o.provides.build
.PHONY : CMakeFiles/fake_code.dir/Utils.cpp.o.provides

CMakeFiles/fake_code.dir/Utils.cpp.o.provides.build: CMakeFiles/fake_code.dir/Utils.cpp.o


CMakeFiles/fake_code.dir/Visualize.cpp.o: CMakeFiles/fake_code.dir/flags.make
CMakeFiles/fake_code.dir/Visualize.cpp.o: ../Visualize.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tri/Desktop/ann_implement/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/fake_code.dir/Visualize.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fake_code.dir/Visualize.cpp.o -c /home/tri/Desktop/ann_implement/Visualize.cpp

CMakeFiles/fake_code.dir/Visualize.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fake_code.dir/Visualize.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tri/Desktop/ann_implement/Visualize.cpp > CMakeFiles/fake_code.dir/Visualize.cpp.i

CMakeFiles/fake_code.dir/Visualize.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fake_code.dir/Visualize.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tri/Desktop/ann_implement/Visualize.cpp -o CMakeFiles/fake_code.dir/Visualize.cpp.s

CMakeFiles/fake_code.dir/Visualize.cpp.o.requires:

.PHONY : CMakeFiles/fake_code.dir/Visualize.cpp.o.requires

CMakeFiles/fake_code.dir/Visualize.cpp.o.provides: CMakeFiles/fake_code.dir/Visualize.cpp.o.requires
	$(MAKE) -f CMakeFiles/fake_code.dir/build.make CMakeFiles/fake_code.dir/Visualize.cpp.o.provides.build
.PHONY : CMakeFiles/fake_code.dir/Visualize.cpp.o.provides

CMakeFiles/fake_code.dir/Visualize.cpp.o.provides.build: CMakeFiles/fake_code.dir/Visualize.cpp.o


# Object files for target fake_code
fake_code_OBJECTS = \
"CMakeFiles/fake_code.dir/main.cpp.o" \
"CMakeFiles/fake_code.dir/Trainer.cpp.o" \
"CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o" \
"CMakeFiles/fake_code.dir/DataReader.cpp.o" \
"CMakeFiles/fake_code.dir/DataGenerator.cpp.o" \
"CMakeFiles/fake_code.dir/HOGFeature.cpp.o" \
"CMakeFiles/fake_code.dir/Layer.cpp.o" \
"CMakeFiles/fake_code.dir/Utils.cpp.o" \
"CMakeFiles/fake_code.dir/Visualize.cpp.o"

# External object files for target fake_code
fake_code_EXTERNAL_OBJECTS =

fake_code: CMakeFiles/fake_code.dir/main.cpp.o
fake_code: CMakeFiles/fake_code.dir/Trainer.cpp.o
fake_code: CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o
fake_code: CMakeFiles/fake_code.dir/DataReader.cpp.o
fake_code: CMakeFiles/fake_code.dir/DataGenerator.cpp.o
fake_code: CMakeFiles/fake_code.dir/HOGFeature.cpp.o
fake_code: CMakeFiles/fake_code.dir/Layer.cpp.o
fake_code: CMakeFiles/fake_code.dir/Utils.cpp.o
fake_code: CMakeFiles/fake_code.dir/Visualize.cpp.o
fake_code: CMakeFiles/fake_code.dir/build.make
fake_code: /usr/lib/x86_64-linux-gnu/libGLU.so
fake_code: /usr/lib/x86_64-linux-gnu/libGL.so
fake_code: /usr/lib/x86_64-linux-gnu/libglut.so
fake_code: /usr/lib/x86_64-linux-gnu/libXmu.so
fake_code: /usr/lib/x86_64-linux-gnu/libXi.so
fake_code: /usr/lib/libarmadillo.so
fake_code: CMakeFiles/fake_code.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tri/Desktop/ann_implement/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable fake_code"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fake_code.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fake_code.dir/build: fake_code

.PHONY : CMakeFiles/fake_code.dir/build

CMakeFiles/fake_code.dir/requires: CMakeFiles/fake_code.dir/main.cpp.o.requires
CMakeFiles/fake_code.dir/requires: CMakeFiles/fake_code.dir/Trainer.cpp.o.requires
CMakeFiles/fake_code.dir/requires: CMakeFiles/fake_code.dir/NeuralNetwork.cpp.o.requires
CMakeFiles/fake_code.dir/requires: CMakeFiles/fake_code.dir/DataReader.cpp.o.requires
CMakeFiles/fake_code.dir/requires: CMakeFiles/fake_code.dir/DataGenerator.cpp.o.requires
CMakeFiles/fake_code.dir/requires: CMakeFiles/fake_code.dir/HOGFeature.cpp.o.requires
CMakeFiles/fake_code.dir/requires: CMakeFiles/fake_code.dir/Layer.cpp.o.requires
CMakeFiles/fake_code.dir/requires: CMakeFiles/fake_code.dir/Utils.cpp.o.requires
CMakeFiles/fake_code.dir/requires: CMakeFiles/fake_code.dir/Visualize.cpp.o.requires

.PHONY : CMakeFiles/fake_code.dir/requires

CMakeFiles/fake_code.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fake_code.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fake_code.dir/clean

CMakeFiles/fake_code.dir/depend:
	cd /home/tri/Desktop/ann_implement/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tri/Desktop/ann_implement /home/tri/Desktop/ann_implement /home/tri/Desktop/ann_implement/cmake-build-debug /home/tri/Desktop/ann_implement/cmake-build-debug /home/tri/Desktop/ann_implement/cmake-build-debug/CMakeFiles/fake_code.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fake_code.dir/depend

