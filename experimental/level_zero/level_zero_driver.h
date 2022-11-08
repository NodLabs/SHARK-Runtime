// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LEVEL_ZERO_LEVEL_ZERO_DRIVER_H_
#define IREE_HAL_LEVEL_ZERO_LEVEL_ZERO_DRIVER_H_

#include "experimental/level_zero/api.h"
#include "experimental/level_zero/dynamic_symbols.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_level_zero_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  // Identifier used for the driver in the IREE driver registry.
  // We allow overriding so that multiple LevelZero versions can be exposed in
  // the same process.
  iree_string_view_t identifier;
  int default_device_index;

  // Level Zero Driver Handle.
  ze_driver_handle_t driver_handle;
  ze_context_handle_t context;
  // LevelZero symbols.
  iree_hal_level_zero_dynamic_symbols_t syms;

} iree_hal_level_zero_driver_t;

iree_hal_level_zero_driver_t* iree_hal_level_zero_driver_cast(
    iree_hal_driver_t* base_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LEVEL_ZERO_LEVEL_ZERO_DRIVER_H_
