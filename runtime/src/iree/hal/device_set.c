// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/device_set.h"
#include "iree/hal/device.h"

#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/command_buffer.h"
#include "iree/hal/detail.h"
#include "iree/hal/resource.h"


IREE_API_EXPORT iree_status_t iree_hal_device_set_initialize(iree_hal_device_set_t *devices) {
  memset(devices, 0, sizeof(*devices));
  return iree_ok_status();
}

IREE_API_EXPORT iree_host_size_t iree_hal_device_set_num_devices(iree_hal_device_set_t *devices) {
  return devices->count;
}

IREE_API_EXPORT iree_status_t iree_hal_device_set_insert(iree_hal_device_set_t* device_set, iree_hal_device_t *device) {
  if (device_set->count == MAX_DEVICES - 1)
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  iree_hal_device_retain(device);
  device_set->devices[device_set->count++] = device;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_device_set_get(iree_hal_device_set_t* device_set, int i, iree_hal_device_t *device) {
  if (i >= device_set->count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "index %d out of bounds (%zu)", i, device_set->count);
  }
  device = device_set->devices[i];
  return iree_ok_status();
}

IREE_API_EXPORT void iree_hal_device_set_retain(iree_hal_device_set_t* device_set) {
  for (int i = 0; i < device_set->count; i++)
    iree_hal_device_retain(device_set->devices[i]);
}

IREE_API_EXPORT void iree_hal_device_set_release(iree_hal_device_set_t* device_set) {
  for (int i = 0; i < device_set->count; i++)
    iree_hal_device_release(device_set->devices[i]);
}

IREE_API_EXPORT void iree_hal_device_set_destroy(iree_hal_device_set_t* device_set) {
  for (int i = 0; i < device_set->count; i++)
    iree_hal_device_destroy(device_set->devices[i]);
}