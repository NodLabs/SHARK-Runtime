// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ONECCL_ONECCL_H_
#define IREE_ONECCL_ONECCL_H_

#include <experimental/level_zero/level_zero_device.h>
#include <experimental/level_zero/level_zero_driver.h>
#include <iree/hal/api.h>

#ifdef __cplusplus
extern "C" {
#endif

void iree_hal_level_zero_device_oneccl_init_vtable(
    iree_hal_device_vtable_t* vtable);
void iree_hal_level_zero_direct_command_buffer_oneccl_init_vtable(
    iree_hal_command_buffer_vtable_t* vtable);

iree_status_t iree_hal_level_zero_device_oneccl_create_channel(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    iree_hal_channel_params_t params, iree_hal_channel_t** out_channel);

iree_status_t iree_hal_level_zero_direct_command_buffer_oneccl_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count);

#ifdef __cplusplus
}
#endif

#endif  // IREE_ONECCL_ONECCL_H_
