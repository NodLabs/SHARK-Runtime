// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LEVEL_ZERO_LEVEL_ZERO_DEVICE_H_
#define IREE_HAL_LEVEL_ZERO_LEVEL_ZERO_DEVICE_H_

#include "experimental/level_zero/api.h"
#include "experimental/level_zero/context_wrapper.h"
#include "experimental/level_zero/dynamic_symbols.h"
#include "experimental/level_zero/level_zero_headers.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_level_zero_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Optional driver that owns the Level Zero symbols. We retain it for our
  // lifetime to ensure the symbols remains valid.
  iree_hal_driver_t* driver;

  // Level Zero APIs.
  ze_device_handle_t device;
  uint32_t command_queue_ordinal;
  ze_command_queue_handle_t command_queue;
  ze_event_pool_handle_t event_pool;

  iree_hal_level_zero_context_wrapper_t context_wrapper;
  iree_hal_allocator_t* device_allocator;

} iree_hal_level_zero_device_t;

iree_hal_level_zero_device_t* iree_hal_level_zero_device_cast(
    iree_hal_device_t* base_value);

// Creates a device that owns and manages its own hipContext.
iree_status_t iree_hal_level_zero_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    iree_hal_level_zero_dynamic_symbols_t* syms,
    ze_device_handle_t level_zero_device,
    ze_context_handle_t level_zero_context, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device);

typedef enum iree_hal_level_zero_ccl_backend_e {
  IREE_HAL_LEVEL_ZERO_CCL_BACKEND_ONECCL = 0,
  IREE_HAL_LEVEL_ZERO_CCL_BACKEND_COUNT
} iree_hal_level_zero_ccl_backend_e;

typedef struct iree_hal_level_zero_device_params_t {
  iree_hal_level_zero_ccl_backend_e ccl_backend;
} iree_hal_level_zero_device_params_t;

iree_status_t iree_hal_level_zero_device_params_parse(
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_hal_level_zero_device_params_t* out_params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LEVEL_ZERO_LEVEL_ZERO_DEVICE_H_
