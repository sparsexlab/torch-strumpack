cmake_minimum_required(VERSION 3.18)
project(torch_strumpack_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3 REQUIRED COMPONENTS Interpreter Development.Module)

# locate nanobind (pip-installed) via its cmake dir
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE NB_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ROCm deps that STRUMPACK's HIP build links (roc::hipblas, roc::rocsolver,
# roc::rocthrust, ...). These imported targets are carried transitively by
# STRUMPACK::strumpack, so they MUST be found BEFORE find_package(STRUMPACK)
# or the link `add_library(... STRUMPACK::strumpack ...)` fails with
# "links to target roc::<X> but the target was not found".
# rocthrust (-> roc::rocthrust) and its dep rocprim are required in addition
# to the blas/sparse/solver libs; mirror build_strumpack.sh's ROCM_PATH on
# CMAKE_PREFIX_PATH so all the *-config.cmake dirs under $ROCM_PATH resolve.
if(DEFINED ENV{ROCM_PATH})
  list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH})
else()
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
endif()
find_package(hip QUIET)
find_package(hipblas QUIET)
find_package(hipsparse QUIET)
find_package(rocblas QUIET)
find_package(rocsolver QUIET)
find_package(rocsparse QUIET)
find_package(rocprim QUIET)
find_package(rocthrust QUIET)

# CUDA deps that STRUMPACK's CUDA build links (CUDA::cublas, CUDA::cusolver, ...)
find_package(CUDAToolkit QUIET)

# STRUMPACK (pass -DSTRUMPACK_DIR=/opt/strumpack/lib/cmake/STRUMPACK)
find_package(STRUMPACK REQUIRED)

nanobind_add_module(_strumpack_ext src/strumpack_ext.cpp)
target_link_libraries(_strumpack_ext PRIVATE STRUMPACK::strumpack)

# drop the .so next to the python package so `from torch_strumpack import _strumpack_ext` works
install(TARGETS _strumpack_ext LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/torch_strumpack)
