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
include tests/CMakeFiles/test-c.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/test-c.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/test-c.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/test-c.dir/flags.make

tests/CMakeFiles/test-c.dir/test-c.c.o: tests/CMakeFiles/test-c.dir/flags.make
tests/CMakeFiles/test-c.dir/test-c.c.o: ../tests/test-c.c
tests/CMakeFiles/test-c.dir/test-c.c.o: tests/CMakeFiles/test-c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object tests/CMakeFiles/test-c.dir/test-c.c.o"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT tests/CMakeFiles/test-c.dir/test-c.c.o -MF CMakeFiles/test-c.dir/test-c.c.o.d -o CMakeFiles/test-c.dir/test-c.c.o -c /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/tests/test-c.c

tests/CMakeFiles/test-c.dir/test-c.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/test-c.dir/test-c.c.i"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/tests/test-c.c > CMakeFiles/test-c.dir/test-c.c.i

tests/CMakeFiles/test-c.dir/test-c.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/test-c.dir/test-c.c.s"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/tests/test-c.c -o CMakeFiles/test-c.dir/test-c.c.s

# Object files for target test-c
test__c_OBJECTS = \
"CMakeFiles/test-c.dir/test-c.c.o"

# External object files for target test-c
test__c_EXTERNAL_OBJECTS =

bin/test-c: tests/CMakeFiles/test-c.dir/test-c.c.o
bin/test-c: tests/CMakeFiles/test-c.dir/build.make
bin/test-c: libllama.a
bin/test-c: tests/CMakeFiles/test-c.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/test-c"
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test-c.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/test-c.dir/build: bin/test-c
.PHONY : tests/CMakeFiles/test-c.dir/build

tests/CMakeFiles/test-c.dir/clean:
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/test-c.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/test-c.dir/clean

tests/CMakeFiles/test-c.dir/depend:
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/tests /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/tests/CMakeFiles/test-c.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/test-c.dir/depend

