# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build

# Include any dependencies generated for this target.
include examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/progress.make

# Include the compile flags for this target's objects.
include examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/flags.make

examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.o: examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/flags.make
examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.o: ../examples/train-text-from-scratch/train-text-from-scratch.cpp
examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.o: examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.o"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/train-text-from-scratch && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.o -MF CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.o.d -o CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.o -c /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/train-text-from-scratch/train-text-from-scratch.cpp

examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.i"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/train-text-from-scratch && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/train-text-from-scratch/train-text-from-scratch.cpp > CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.i

examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.s"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/train-text-from-scratch && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/train-text-from-scratch/train-text-from-scratch.cpp -o CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.s

# Object files for target train-text-from-scratch
train__text__from__scratch_OBJECTS = \
"CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.o"

# External object files for target train-text-from-scratch
train__text__from__scratch_EXTERNAL_OBJECTS =

bin/train-text-from-scratch: examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/train-text-from-scratch.cpp.o
bin/train-text-from-scratch: examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/build.make
bin/train-text-from-scratch: common/libcommon.a
bin/train-text-from-scratch: libllama.a
bin/train-text-from-scratch: examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/train-text-from-scratch"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/train-text-from-scratch && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/train-text-from-scratch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/build: bin/train-text-from-scratch
.PHONY : examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/build

examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/clean:
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/train-text-from-scratch && $(CMAKE_COMMAND) -P CMakeFiles/train-text-from-scratch.dir/cmake_clean.cmake
.PHONY : examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/clean

examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/depend:
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/train-text-from-scratch /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/train-text-from-scratch /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/train-text-from-scratch/CMakeFiles/train-text-from-scratch.dir/depend
