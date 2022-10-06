// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DEVICE_SET_H_
#define IREE_HAL_DEVICE_SET_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/event.h"
#include "iree/hal/executable_cache.h"
#include "iree/hal/fence.h"
#include "iree/hal/pipeline_layout.h"
#include "iree/hal/resource.h"
#include "iree/hal/semaphore.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_device_set_t
//===----------------------------------------------------------------------===//

#define MAX_DEVICES (16)

typedef struct iree_hal_device_set_t {
  iree_hal_device_t *devices[MAX_DEVICES];
  iree_host_size_t count;
} iree_hal_device_set_t;

// Initialize the set of devices
IREE_API_EXPORT iree_status_t iree_hal_device_set_initialize(iree_hal_device_set_t *devices);

// Get number of devices in the set
IREE_API_EXPORT iree_host_size_t iree_hal_device_set_num_devices(iree_hal_device_set_t *devices);

// Inserts a device into the set of devices
IREE_API_EXPORT iree_status_t iree_hal_device_set_insert(iree_hal_device_set_t* devices, iree_hal_device_t *device);

// Retrieves the ith device from the set of devices
IREE_API_EXPORT iree_status_t iree_hal_device_set_get(iree_hal_device_set_t* devices, int i, iree_hal_device_t *device);

// Retains the given set of devices for the caller.
IREE_API_EXPORT void iree_hal_device_set_retain(iree_hal_device_set_t* devices);

// Releases the given set of devices from the caller.
IREE_API_EXPORT void iree_hal_device_set_release(iree_hal_device_set_t* devices);

// Destroys the set of devices
IREE_API_EXPORT void iree_hal_device_set_destroy(iree_hal_device_set_t* devices);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DEVICE_SET_H_
