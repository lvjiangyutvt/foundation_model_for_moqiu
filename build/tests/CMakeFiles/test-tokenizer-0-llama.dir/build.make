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
include tests/CMakeFiles/test-tokenizer-0-llama.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test-tokenizer-0-llama.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test-tokenizer-0-llama.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test-tokenizer-0-llama.dir/flags.make

tests/CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.o: tests/CMakeFiles/test-tokenizer-0-llama.dir/flags.make
tests/CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.o: ../tests/test-tokenizer-0-llama.cpp
tests/CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.o: tests/CMakeFiles/test-tokenizer-0-llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.o"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.o -MF CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.o.d -o CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.o -c /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/tests/test-tokenizer-0-llama.cpp

tests/CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.i"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/tests/test-tokenizer-0-llama.cpp > CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.i

tests/CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.s"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/tests/test-tokenizer-0-llama.cpp -o CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.s

# Object files for target test-tokenizer-0-llama
test__tokenizer__0__llama_OBJECTS = \
"CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.o"

# External object files for target test-tokenizer-0-llama
test__tokenizer__0__llama_EXTERNAL_OBJECTS =

bin/test-tokenizer-0-llama: tests/CMakeFiles/test-tokenizer-0-llama.dir/test-tokenizer-0-llama.cpp.o
bin/test-tokenizer-0-llama: tests/CMakeFiles/test-tokenizer-0-llama.dir/build.make
bin/test-tokenizer-0-llama: libllama.a
bin/test-tokenizer-0-llama: common/libcommon.a
bin/test-tokenizer-0-llama: libllama.a
bin/test-tokenizer-0-llama: tests/CMakeFiles/test-tokenizer-0-llama.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/test-tokenizer-0-llama"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-tokenizer-0-llama.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test-tokenizer-0-llama.dir/build: bin/test-tokenizer-0-llama
.PHONY : tests/CMakeFiles/test-tokenizer-0-llama.dir/build

tests/CMakeFiles/test-tokenizer-0-llama.dir/clean:
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test-tokenizer-0-llama.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test-tokenizer-0-llama.dir/clean

tests/CMakeFiles/test-tokenizer-0-llama.dir/depend:
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/tests /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests/CMakeFiles/test-tokenizer-0-llama.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/test-tokenizer-0-llama.dir/depend
