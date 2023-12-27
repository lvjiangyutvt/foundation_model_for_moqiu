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
include examples/batched-bench/CMakeFiles/batched-bench.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/batched-bench/CMakeFiles/batched-bench.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/batched-bench/CMakeFiles/batched-bench.dir/progress.make

# Include the compile flags for this target's objects.
include examples/batched-bench/CMakeFiles/batched-bench.dir/flags.make

examples/batched-bench/CMakeFiles/batched-bench.dir/batched-bench.cpp.o: examples/batched-bench/CMakeFiles/batched-bench.dir/flags.make
examples/batched-bench/CMakeFiles/batched-bench.dir/batched-bench.cpp.o: ../examples/batched-bench/batched-bench.cpp
examples/batched-bench/CMakeFiles/batched-bench.dir/batched-bench.cpp.o: examples/batched-bench/CMakeFiles/batched-bench.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/batched-bench/CMakeFiles/batched-bench.dir/batched-bench.cpp.o"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/batched-bench && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/batched-bench/CMakeFiles/batched-bench.dir/batched-bench.cpp.o -MF CMakeFiles/batched-bench.dir/batched-bench.cpp.o.d -o CMakeFiles/batched-bench.dir/batched-bench.cpp.o -c /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/batched-bench/batched-bench.cpp

examples/batched-bench/CMakeFiles/batched-bench.dir/batched-bench.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/batched-bench.dir/batched-bench.cpp.i"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/batched-bench && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/batched-bench/batched-bench.cpp > CMakeFiles/batched-bench.dir/batched-bench.cpp.i

examples/batched-bench/CMakeFiles/batched-bench.dir/batched-bench.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/batched-bench.dir/batched-bench.cpp.s"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/batched-bench && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/batched-bench/batched-bench.cpp -o CMakeFiles/batched-bench.dir/batched-bench.cpp.s

# Object files for target batched-bench
batched__bench_OBJECTS = \
"CMakeFiles/batched-bench.dir/batched-bench.cpp.o"

# External object files for target batched-bench
batched__bench_EXTERNAL_OBJECTS =

bin/batched-bench: examples/batched-bench/CMakeFiles/batched-bench.dir/batched-bench.cpp.o
bin/batched-bench: examples/batched-bench/CMakeFiles/batched-bench.dir/build.make
bin/batched-bench: common/libcommon.a
bin/batched-bench: libllama.a
bin/batched-bench: examples/batched-bench/CMakeFiles/batched-bench.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/batched-bench"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/batched-bench && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/batched-bench.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/batched-bench/CMakeFiles/batched-bench.dir/build: bin/batched-bench
.PHONY : examples/batched-bench/CMakeFiles/batched-bench.dir/build

examples/batched-bench/CMakeFiles/batched-bench.dir/clean:
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/batched-bench && $(CMAKE_COMMAND) -P CMakeFiles/batched-bench.dir/cmake_clean.cmake
.PHONY : examples/batched-bench/CMakeFiles/batched-bench.dir/clean

examples/batched-bench/CMakeFiles/batched-bench.dir/depend:
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/batched-bench /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/batched-bench /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/batched-bench/CMakeFiles/batched-bench.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/batched-bench/CMakeFiles/batched-bench.dir/depend

