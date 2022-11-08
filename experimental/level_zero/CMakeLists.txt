# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

if(NOT LEVEL_ZERO_HEADERS_API_ROOT)
  set(LEVEL_ZERO_HEADERS_API_ROOT "${IREE_ROOT_DIR}/third_party/level-zero/")
  message(STATUS "Using default level-zero directory at ${LEVEL_ZERO_HEADERS_API_ROOT}")
endif()

if(EXISTS ${LEVEL_ZERO_HEADERS_API_ROOT})
  message(STATUS "Level Zero Header Path: ${LEVEL_ZERO_HEADERS_API_ROOT}")
else()
  message(SEND_ERROR "Could not locate Level Zero: ${LEVEL_ZERO_HEADERS_API_ROOT}")
endif()

if(IREE_BUILD_EXPERIMENTAL_HAL_DRIVER_LEVEL_ZERO_ONECCL)
  set(LEVEL_ZERO_ONECCL_TARGET iree::experimental::level_zero::oneccl)
endif()

iree_cc_library(
  NAME
    level_zero
  HDRS
    "api.h"
  SRCS
    ${CMAKE_CURRENT_BINARY_DIR}/config.h
    "api.h"
    "level_zero_allocator.c"
    "level_zero_allocator.h"
    "level_zero_buffer.c"
    "level_zero_buffer.h"
    "level_zero_device.c"
    "level_zero_device.cc"
    "level_zero_driver.c"
    "level_zero_event.c"
    "level_zero_event.h"
    "event_semaphore.c"
    "event_semaphore.h"
    "pipeline_layout.c"
    "pipeline_layout.h"
    "direct_command_buffer.c"
    "direct_command_buffer.h"
    "native_executable.c"
    "native_executable.h"
    "nop_executable_cache.c"
    "nop_executable_cache.h"
    "status_util.c"
    "status_util.h"
  INCLUDES
    "${CMAKE_CURRENT_LIST_DIR}/../.."
    "${PROJECT_BINARY_DIR}"
    "${LEVEL_ZERO_HEADERS_API_ROOT}"
    "${CMAKE_CURRENT_BINARY_DIR}"
  DEPS
    ::internal_interface
    iree::base::internal
    iree::base::internal::flatcc::parsing
    iree::base::internal::synchronization
    iree::base::tracing
    iree::hal
    iree::hal::utils::buffer_transfer
    iree::hal::utils::semaphore_base
    iree::schemas::level_zero_executable_def_c_fbs
    ${LEVEL_ZERO_ONECCL_TARGET}
  PUBLIC
)

# This target is required because iree::experimental::level_zero::oneccl
# has to be compiled with DPC++.
# It needs some of the definitions from here. With this target we
# avoid circular dependency.
iree_cc_library(
  NAME
    internal_interface
  HDRS
    "api.h"
    "context_wrapper.h"
    "level_zero_driver.h"
    "level_zero_device.h"
    "level_zero_headers.h"
#  SRCS
#    "internal_interface.c"
  INCLUDES
    "${CMAKE_CURRENT_LIST_DIR}/../.."
    "${PROJECT_BINARY_DIR}"
    "${LEVEL_ZERO_HEADERS_API_ROOT}"
  DEPS
    ::dynamic_symbols
    iree::base::core_headers
    iree::base::internal::arena
    iree::base
    iree::hal
  PUBLIC
)

iree_cc_library(
  NAME
    dynamic_symbols
  HDRS
    "dynamic_symbols.h"
  TEXTUAL_HDRS
    "dynamic_symbol_tables.h"
  SRCS
    "level_zero_headers.h"
    "dynamic_symbols.c"
  INCLUDES
    "${LEVEL_ZERO_HEADERS_API_ROOT}/include"
    "${CMAKE_CURRENT_LIST_DIR}/../.."
  DEPS
    iree::base::core_headers
    iree::base::internal::dynamic_library
    iree::base::tracing
  PUBLIC
)

iree_cc_test(
  NAME
    dynamic_symbols_test
  SRCS
    "dynamic_symbols_test.cc"
  DEPS
    ::dynamic_symbols
    iree::base
    iree::testing::gtest
    iree::testing::gtest_main
  LABELS
    "driver=level_zero"
)

configure_file(config.h.in config.h)
