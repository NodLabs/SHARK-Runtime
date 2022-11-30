// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_LOCAL_TASK_REGISTRATION_DRIVER_MODULE_H_
#define IREE_HAL_DRIVERS_LOCAL_TASK_REGISTRATION_DRIVER_MODULE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "third_party/numactl/numa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

IREE_API_EXPORT iree_status_t iree_hal_local_task_driver_module_register(
    iree_hal_driver_registry_t* registry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_LOCAL_TASK_REGISTRATION_DRIVER_MODULE_H_
