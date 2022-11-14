# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_BUILD_EXPERIMENTAL_HAL_DRIVER_LEVEL_ZERO_ONECCL)
  return()
endif()

iree_add_all_subdirs()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TODO: Use externalproject_add.
# Overriding CMAKE_*_COMPILER is not considered a good practice.
set(CMAKE_C_COMPILER "${SYCL_C_COMPILER}")
set(CMAKE_CXX_COMPILER "${SYCL_CXX_COMPILER}")


if(NOT LEVEL_ZERO_HEADERS_API_ROOT)
  set(LEVEL_ZERO_HEADERS_API_ROOT "${IREE_ROOT_DIR}/third_party/level-zero/include")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(ONECCL_OPTIMIZATION_LEVEL "-O0")
endif()

iree_cc_library(
  NAME
    oneccl
  HDRS
    "src/iree/hal/drivers/level_zero/oneccl/oneccl.h"
  SRCS
    "src/iree/hal/drivers/level_zero/oneccl/oneccl.cc"
  INCLUDES
    "${LEVEL_ZERO_HEADERS_API_ROOT}"
    "${CMAKE_CURRENT_LIST_DIR}/src"
  COPTS
    # Intel's oneAPI uses exceptions and type info
    "-frtti"
    "-fexceptions"
    # This is needed because of
    # dpcpp: error: Note that use of '-g' without any optimization-level option will turn off most compiler optimizations similar to use of '-O0' [-Werror,-Wdebug-disables-optimization]
    # -Wno-error=debug-disables-optimization does not demote the error to warning.
    ${ONECCL_OPTIMIZATION_LEVEL}
  DEPS
    iree::hal
    iree::experimental::level_zero::internal_interface
    MPI::MPI_CXX
    oneCCL
    Intel::SYCL_level_zero
  PUBLIC
)
