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
include examples/beam-search/CMakeFiles/beam-search.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/beam-search/CMakeFiles/beam-search.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/beam-search/CMakeFiles/beam-search.dir/progress.make

# Include the compile flags for this target's objects.
include examples/beam-search/CMakeFiles/beam-search.dir/flags.make

examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.o: examples/beam-search/CMakeFiles/beam-search.dir/flags.make
examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.o: ../examples/beam-search/beam-search.cpp
examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.o: examples/beam-search/CMakeFiles/beam-search.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.o"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/beam-search && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.o -MF CMakeFiles/beam-search.dir/beam-search.cpp.o.d -o CMakeFiles/beam-search.dir/beam-search.cpp.o -c /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/beam-search/beam-search.cpp

examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/beam-search.dir/beam-search.cpp.i"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/beam-search && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/beam-search/beam-search.cpp > CMakeFiles/beam-search.dir/beam-search.cpp.i

examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/beam-search.dir/beam-search.cpp.s"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/beam-search && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/beam-search/beam-search.cpp -o CMakeFiles/beam-search.dir/beam-search.cpp.s

# Object files for target beam-search
beam__search_OBJECTS = \
"CMakeFiles/beam-search.dir/beam-search.cpp.o"

# External object files for target beam-search
beam__search_EXTERNAL_OBJECTS =

bin/beam-search: examples/beam-search/CMakeFiles/beam-search.dir/beam-search.cpp.o
bin/beam-search: examples/beam-search/CMakeFiles/beam-search.dir/build.make
bin/beam-search: common/libcommon.a
bin/beam-search: libllama.a
bin/beam-search: examples/beam-search/CMakeFiles/beam-search.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/beam-search"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/beam-search && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/beam-search.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/beam-search/CMakeFiles/beam-search.dir/build: bin/beam-search
.PHONY : examples/beam-search/CMakeFiles/beam-search.dir/build

examples/beam-search/CMakeFiles/beam-search.dir/clean:
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/beam-search && $(CMAKE_COMMAND) -P CMakeFiles/beam-search.dir/cmake_clean.cmake
.PHONY : examples/beam-search/CMakeFiles/beam-search.dir/clean

examples/beam-search/CMakeFiles/beam-search.dir/depend:
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/beam-search /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/beam-search /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/beam-search/CMakeFiles/beam-search.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/beam-search/CMakeFiles/beam-search.dir/depend

