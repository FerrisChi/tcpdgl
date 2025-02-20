#--------------------------------------------------------------------
#  Template custom cmake configuration for compiling
#
#  This file is used to override the build options in build.
#  If you want to change the configuration, please use the following
#  steps. Assume you are on the root directory. First copy the this
#  file so that any local changes will be ignored by git
#
#  $ mkdir build
#  $ cp cmake/config.cmake build
#
#  Next modify the according entries, and then compile by
#
#  $ cd build
#  $ cmake ..
#
#  Then buld in parallel with 8 threads
#
#  $ make -j8
#--------------------------------------------------------------------

#---------------------------------------------
# Backend runtimes.
#---------------------------------------------

# Whether enable CUDA during compile,
#
# Possible values:
# - ON: enable CUDA with cmake's auto search
# - OFF: disable CUDA
# - /path/to/cuda: use specific path to cuda toolkit
set(USE_CUDA ON)

#---------------------------------------------
# Misc.
#---------------------------------------------
# Whether to build cpp unittest executables.
set(BUILD_CPP_TEST OFF)

# Whether to enable OpenMP.
set(USE_OPENMP ON)

# Whether to build PyTorch plugins.
set(BUILD_TORCH ON)

# Whether to enable CUDA kernels compiled with TVM.
set(USE_TVM OFF)

# Whether to build DGL sparse library.
set(BUILD_SPARSE ON)
# Whether to enable fp16 to support mixed precision training.
set(USE_FP16 OFF)

# https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html
# -DCMAKE_BUILD_TYPE="Debug" 
# Debug Release RelWithDebInfo MinSizeRel
set(CMAKE_BUILD_TYPE "Debug")
