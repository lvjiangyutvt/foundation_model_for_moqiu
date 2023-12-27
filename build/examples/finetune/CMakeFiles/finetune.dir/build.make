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
include examples/finetune/CMakeFiles/finetune.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/finetune/CMakeFiles/finetune.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/finetune/CMakeFiles/finetune.dir/progress.make

# Include the compile flags for this target's objects.
include examples/finetune/CMakeFiles/finetune.dir/flags.make

examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.o: examples/finetune/CMakeFiles/finetune.dir/flags.make
examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.o: ../examples/finetune/finetune.cpp
examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.o: examples/finetune/CMakeFiles/finetune.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.o"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/finetune && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.o -MF CMakeFiles/finetune.dir/finetune.cpp.o.d -o CMakeFiles/finetune.dir/finetune.cpp.o -c /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/finetune/finetune.cpp

examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/finetune.dir/finetune.cpp.i"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/finetune && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/finetune/finetune.cpp > CMakeFiles/finetune.dir/finetune.cpp.i

examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/finetune.dir/finetune.cpp.s"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/finetune && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/finetune/finetune.cpp -o CMakeFiles/finetune.dir/finetune.cpp.s

# Object files for target finetune
finetune_OBJECTS = \
"CMakeFiles/finetune.dir/finetune.cpp.o"

# External object files for target finetune
finetune_EXTERNAL_OBJECTS =

bin/finetune: examples/finetune/CMakeFiles/finetune.dir/finetune.cpp.o
bin/finetune: examples/finetune/CMakeFiles/finetune.dir/build.make
bin/finetune: common/libcommon.a
bin/finetune: libllama.a
bin/finetune: examples/finetune/CMakeFiles/finetune.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/finetune"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/finetune && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/finetune.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/finetune/CMakeFiles/finetune.dir/build: bin/finetune
.PHONY : examples/finetune/CMakeFiles/finetune.dir/build

examples/finetune/CMakeFiles/finetune.dir/clean:
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/finetune && $(CMAKE_COMMAND) -P CMakeFiles/finetune.dir/cmake_clean.cmake
.PHONY : examples/finetune/CMakeFiles/finetune.dir/clean

examples/finetune/CMakeFiles/finetune.dir/depend:
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/examples/finetune /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/finetune /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/examples/finetune/CMakeFiles/finetune.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/finetune/CMakeFiles/finetune.dir/depend

