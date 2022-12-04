// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/level_zero/direct_command_buffer.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <cstdio>

#include "config.h"
#include "experimental/level_zero/dynamic_symbols.h"
#include "experimental/level_zero/level_zero_buffer.h"
#include "experimental/level_zero/status_util.h"
#include "iree/base/api.h"

using namespace iree::level_zero;

iree_status_t iree_hal_level_zero_command_buffer_segment_list_create(
    iree_hal_level_zero_device_t* device,
    iree_hal_level_zero_command_buffer_segment_list_t** out) {
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      device->context_wrapper.host_allocator,
      sizeof(iree_hal_level_zero_command_buffer_segment_list_t),
      reinterpret_cast<void**>(out)));
  new (&(*out)->segments)
      iree_hal_level_zero_command_buffer_segment_list_t(device);
  return iree_ok_status();
}

void iree_hal_level_zero_command_buffer_segment_list_destroy(
    iree_hal_level_zero_command_buffer_segment_list_t* list) {
  list->~iree_hal_level_zero_command_buffer_segment_list_t();
  iree_allocator_free(list->device->context_wrapper.host_allocator, list);
}

iree_status_t
iree_hal_level_zero_command_buffer_segment_list_get_ze_list_for_append(
    iree_hal_level_zero_command_buffer_segment_list_t* segment_list,
    ze_command_list_handle_t* out_command_list) {
  // Append a new ze_command_list_handle_t if the last element is not such.
  if (segment_list->segments.empty() ||
      !std::holds_alternative<command_list_t>(segment_list->segments.back())) {
    ze_command_list_handle_t command_list;
    ze_command_list_desc_t command_list_desc = {};
    command_list_desc.commandQueueGroupOrdinal =
        segment_list->device->command_queue_ordinal;
    LEVEL_ZERO_RETURN_IF_ERROR(
        segment_list->device->context_wrapper.syms,
        zeCommandListCreate(
            segment_list->device->context_wrapper.level_zero_context,
            segment_list->device->device, &command_list_desc, &command_list),
        "zeCommandListCreate");
    segment_list->segments.emplace_back(command_list_t(
        command_list,
        command_list_deleter_t(segment_list->device->context_wrapper.syms)));
  }

  *out_command_list =
      std::get<command_list_t>(segment_list->segments.back()).get();

  return iree_ok_status();
}

iree_status_t iree_hal_level_zero_buffer_list_create(
    iree_allocator_t host_allocator,
    iree_hal_level_zero_buffer_list_t** buffer_list) {
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(iree_hal_level_zero_buffer_list_t),
      reinterpret_cast<void**>(buffer_list)));
  new (*buffer_list) iree_hal_level_zero_buffer_list_t(host_allocator);
  return iree_ok_status();
}

iree_hal_level_zero_buffer_list_t::~iree_hal_level_zero_buffer_list_t() {
  iree_hal_level_zero_buffer_list_clear(this);
}

void iree_hal_level_zero_buffer_list_destroy(
    iree_hal_level_zero_buffer_list_t* buffer_list) {
  iree_allocator_t host_allocator = buffer_list->host_allocator;
  buffer_list->~iree_hal_level_zero_buffer_list_t();
  iree_allocator_free(host_allocator, buffer_list);
}

void iree_hal_level_zero_buffer_list_append(
    iree_hal_buffer_t* buffer, iree_hal_level_zero_buffer_list_t* buffer_list) {
  iree_hal_buffer_retain(buffer);
  buffer_list->buffers.push_back(buffer);
}

void iree_hal_level_zero_buffer_list_pop(
    iree_hal_level_zero_buffer_list_t* buffer_list) {
  iree_hal_buffer_release(buffer_list->buffers.back());
  buffer_list->buffers.pop_back();
}

void iree_hal_level_zero_buffer_list_clear(
    iree_hal_level_zero_buffer_list_t* buffer_list) {
  for (iree_hal_buffer_t* buffer : buffer_list->buffers) {
    iree_hal_buffer_release(buffer);
  }
  buffer_list->buffers.clear();
}
