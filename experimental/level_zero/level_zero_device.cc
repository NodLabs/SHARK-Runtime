// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/level_zero/level_zero_device.h"

#include <locale>
#include <unordered_map>

#include "experimental/level_zero/direct_command_buffer.h"
#include "experimental/level_zero/status_util.h"

using namespace iree::level_zero;

iree_status_t iree_hal_level_zero_device_parse_params(
    iree_host_size_t param_count, const iree_string_pair_t* params,
    iree_hal_level_zero_device_params_t* out_params) {
  struct iree_string_view_hash_t {
    iree_string_view_hash_t()
        : loc(), coll(std::use_facet<std::collate<char> >(loc)) {}
    std::size_t operator()(const iree_string_view_t& s) const {
      return coll.hash(s.data, s.data + s.size);
    }
    std::locale loc;
    const std::collate<char>& coll;
  };

  struct iree_string_view_equal_t {
    bool operator()(const iree_string_view_t& lhs,
                    const iree_string_view_t& rhs) const {
      return iree_string_view_equal(lhs, rhs);
    }
  };

  std::unordered_map<iree_string_view_t, iree_string_view_t,
                     iree_string_view_hash_t, iree_string_view_equal_t>
      params_map;
  for (std::size_t i = 0; i < param_count; ++i) {
    params_map.insert({params[i].first, params[i].second});
  }

  auto ccl_backend_kv = params_map.find(iree_make_cstring_view("ccl_backend"));
  if (ccl_backend_kv == params_map.end()) {
    out_params->ccl_backend = IREE_HAL_LEVEL_ZERO_CCL_BACKEND_ONECCL;
  } else {
    if (!iree_string_view_equal(ccl_backend_kv->second,
                                iree_make_cstring_view("oneccl"))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Unsupported Level Zero CCL backend.");
    }
    out_params->ccl_backend = IREE_HAL_LEVEL_ZERO_CCL_BACKEND_ONECCL;
  }

  return iree_ok_status();
}

template <class... Ts>
struct overload : Ts... {
  using Ts::operator()...;
};
template <class... Ts>
overload(Ts...) -> overload<Ts...>;

iree_status_t iree_hal_level_zero_device_queue_execute(
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  // TODO(raikonenfnu): Once semaphore is implemented wait for semaphores
  // TODO(thomasraoux): implement semaphores - for now this conservatively
  // synchronizes after every submit.
  for (iree_host_size_t i = 0; i < command_buffer_count; i++) {
    iree_hal_level_zero_direct_command_buffer_t* command_buffer =
        iree_hal_level_zero_direct_command_buffer_cast(command_buffers[i]);
    for (command_buffer_segment_t& segment_variant :
         command_buffer->command_segments->segments) {
      IREE_RETURN_IF_ERROR(std::visit(
          overload{
              [device](command_list_t& segment) {
                ze_command_list_handle_t command_list = segment.get();
                LEVEL_ZERO_RETURN_IF_ERROR(device->context_wrapper.syms,
                                           zeCommandListClose(command_list),
                                           "zeCommandListClose");
                LEVEL_ZERO_RETURN_IF_ERROR(
                    device->context_wrapper.syms,
                    zeCommandQueueExecuteCommandLists(device->command_queue, 1,
                                                      &command_list, NULL),
                    "zeCommandQueueExecuteCommandLists");
                return iree_ok_status();
              },
              [device](functional_command_buffer_segment_t& segment) {
                return segment(device->command_queue);
              },
          },
          segment_variant));
    }
  }

  LEVEL_ZERO_RETURN_IF_ERROR(
      device->context_wrapper.syms,
      zeCommandQueueSynchronize(device->command_queue, IREE_DURATION_INFINITE),
      "zeCommandQueueSynchronize");
  return iree_ok_status();
}
