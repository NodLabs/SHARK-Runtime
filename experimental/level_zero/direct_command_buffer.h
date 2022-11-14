// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LEVEL_ZERO_DIRECT_COMMAND_BUFFER_H_
#define IREE_HAL_LEVEL_ZERO_DIRECT_COMMAND_BUFFER_H_

#include "experimental/level_zero/context_wrapper.h"
#include "experimental/level_zero/dynamic_symbols.h"
#include "experimental/level_zero/level_zero_device.h"
#include "experimental/level_zero/level_zero_headers.h"
#include "experimental/level_zero/pipeline_layout.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_arena_block_pool_t iree_arena_block_pool_t;

// Command buffer implementation that directly maps to level_zero direct.
// This records the commands on the calling thread without additional threading
// indirection.

typedef struct {
  iree_hal_command_buffer_t base;
  iree_hal_level_zero_device_t* device;
  iree_hal_level_zero_context_wrapper_t* context;
  iree_arena_block_pool_t* block_pool;
  ze_command_list_handle_t command_list;

  // Keep track of the current set of kernel arguments.
  int32_t push_constant[IREE_HAL_LEVEL_ZERO_MAX_PUSH_CONSTANT_COUNT];
  void* current_descriptor[];
} iree_hal_level_zero_direct_command_buffer_t;

// Level Zero Kernel Information Structure
typedef struct {
  ze_kernel_handle_t func;
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
  void** kernelParams;
} level_zero_launch_params;

// Creates a Level Zero direct command buffer.
iree_status_t iree_hal_level_zero_direct_command_buffer_create(
    iree_hal_device_t* device, iree_hal_level_zero_context_wrapper_t* context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, ze_device_handle_t level_zero_device,
    uint32_t command_queue_ordinal,
    iree_hal_command_buffer_t** out_command_buffer);

// Returns associated command_list from command buffer.
ze_command_list_handle_t iree_hal_level_zero_direct_command_buffer_exec(
    iree_hal_command_buffer_t* command_buffer);

iree_hal_level_zero_direct_command_buffer_t*
iree_hal_level_zero_direct_command_buffer_cast(
    iree_hal_command_buffer_t* base_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LEVEL_ZERO_DIRECT_COMMAND_BUFFER_H_
