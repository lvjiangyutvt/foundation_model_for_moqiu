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
include CMakeFiles/ggml.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ggml.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ggml.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ggml.dir/flags.make

CMakeFiles/ggml.dir/ggml.c.o: CMakeFiles/ggml.dir/flags.make
CMakeFiles/ggml.dir/ggml.c.o: ../ggml.c
CMakeFiles/ggml.dir/ggml.c.o: CMakeFiles/ggml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/ggml.dir/ggml.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/ggml.dir/ggml.c.o -MF CMakeFiles/ggml.dir/ggml.c.o.d -o CMakeFiles/ggml.dir/ggml.c.o -c /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/ggml.c

CMakeFiles/ggml.dir/ggml.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ggml.dir/ggml.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/ggml.c > CMakeFiles/ggml.dir/ggml.c.i

CMakeFiles/ggml.dir/ggml.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ggml.dir/ggml.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/ggml.c -o CMakeFiles/ggml.dir/ggml.c.s

CMakeFiles/ggml.dir/ggml-alloc.c.o: CMakeFiles/ggml.dir/flags.make
CMakeFiles/ggml.dir/ggml-alloc.c.o: ../ggml-alloc.c
CMakeFiles/ggml.dir/ggml-alloc.c.o: CMakeFiles/ggml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/ggml.dir/ggml-alloc.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/ggml.dir/ggml-alloc.c.o -MF CMakeFiles/ggml.dir/ggml-alloc.c.o.d -o CMakeFiles/ggml.dir/ggml-alloc.c.o -c /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/ggml-alloc.c

CMakeFiles/ggml.dir/ggml-alloc.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ggml.dir/ggml-alloc.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/ggml-alloc.c > CMakeFiles/ggml.dir/ggml-alloc.c.i

CMakeFiles/ggml.dir/ggml-alloc.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ggml.dir/ggml-alloc.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/ggml-alloc.c -o CMakeFiles/ggml.dir/ggml-alloc.c.s

CMakeFiles/ggml.dir/ggml-backend.c.o: CMakeFiles/ggml.dir/flags.make
CMakeFiles/ggml.dir/ggml-backend.c.o: ../ggml-backend.c
CMakeFiles/ggml.dir/ggml-backend.c.o: CMakeFiles/ggml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/ggml.dir/ggml-backend.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/ggml.dir/ggml-backend.c.o -MF CMakeFiles/ggml.dir/ggml-backend.c.o.d -o CMakeFiles/ggml.dir/ggml-backend.c.o -c /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/ggml-backend.c

CMakeFiles/ggml.dir/ggml-backend.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ggml.dir/ggml-backend.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/ggml-backend.c > CMakeFiles/ggml.dir/ggml-backend.c.i

CMakeFiles/ggml.dir/ggml-backend.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ggml.dir/ggml-backend.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/ggml-backend.c -o CMakeFiles/ggml.dir/ggml-backend.c.s

CMakeFiles/ggml.dir/ggml-quants.c.o: CMakeFiles/ggml.dir/flags.make
CMakeFiles/ggml.dir/ggml-quants.c.o: ../ggml-quants.c
CMakeFiles/ggml.dir/ggml-quants.c.o: CMakeFiles/ggml.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/ggml.dir/ggml-quants.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/ggml.dir/ggml-quants.c.o -MF CMakeFiles/ggml.dir/ggml-quants.c.o.d -o CMakeFiles/ggml.dir/ggml-quants.c.o -c /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/ggml-quants.c

CMakeFiles/ggml.dir/ggml-quants.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ggml.dir/ggml-quants.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/ggml-quants.c > CMakeFiles/ggml.dir/ggml-quants.c.i

CMakeFiles/ggml.dir/ggml-quants.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ggml.dir/ggml-quants.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/ggml-quants.c -o CMakeFiles/ggml.dir/ggml-quants.c.s

ggml: CMakeFiles/ggml.dir/ggml.c.o
ggml: CMakeFiles/ggml.dir/ggml-alloc.c.o
ggml: CMakeFiles/ggml.dir/ggml-backend.c.o
ggml: CMakeFiles/ggml.dir/ggml-quants.c.o
ggml: CMakeFiles/ggml.dir/build.make
.PHONY : ggml

# Rule to build all files generated by this target.
CMakeFiles/ggml.dir/build: ggml
.PHONY : CMakeFiles/ggml.dir/build

CMakeFiles/ggml.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ggml.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ggml.dir/clean

CMakeFiles/ggml.dir/depend:
	cd /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build /home/ljy/PycharmProjects/foundation_model_for_moqiu/PowerInfer/build/CMakeFiles/ggml.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ggml.dir/depend
