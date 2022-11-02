// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tooling/vm_util_cc.h"

#include <vector>

#include "iree/modules/hal/types.h"
#include "iree/tooling/vm_util.h"
#include "iree/vm/api.h"

namespace iree {

Status ParseToVariantList(iree_hal_allocator_t* device_allocator,
                          iree::span<const std::string> input_strings,
                          iree_allocator_t host_allocator,
                          iree_vm_list_t** out_list) {
  std::vector<iree_string_view_t> input_string_views(input_strings.size());
  for (size_t i = 0; i < input_strings.size(); ++i) {
    input_string_views[i].data = input_strings[i].data();
    input_string_views[i].size = input_strings[i].size();
  }
  return iree_tooling_parse_to_variant_list(
      device_allocator, input_string_views.data(), input_string_views.size(),
      host_allocator, out_list);
}

Status PrintVariantList(iree_vm_list_t* variant_list, size_t max_element_count,
                        std::string* out_string) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  IREE_RETURN_IF_ERROR(iree_tooling_append_variant_list_lines(
      variant_list, max_element_count, &builder));
  out_string->assign(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  return iree_ok_status();
}

// TODO: Modify this to make use of ParseToVariantList
Status ParseToVariantListMultipleDevices(iree_hal_device_set_t* devices,
                                         iree::span<const std::string> input_strings,
                                         iree_allocator_t host_allocator,
                                         iree_vm_list_t** out_list) {
  IREE_TRACE_SCOPE();

  *out_list = NULL;
  vm::ref<iree_vm_list_t> variant_list;
  // Create a copy of the inputs for each devices
  iree_host_size_t num_devices = iree_hal_device_set_num_devices(devices); 
  IREE_RETURN_IF_ERROR(iree_vm_list_create(
      /*element_type=*/nullptr, input_strings.size() * num_devices, host_allocator,
      &variant_list));
  iree_hal_device_t* device = NULL;
  iree_hal_allocator_t* device_allocator = NULL;
  // Use active device for any device allocations
  iree_hal_device_set_get(devices, 0, &device);
  device_allocator = iree_hal_device_allocator(device);
  for (size_t i = 0; i < input_strings.size(); ++i) {
    iree_string_view_t input_view = iree_string_view_trim(iree_make_string_view(
        input_strings[i].data(), input_strings[i].size()));
    if (iree_string_view_consume_prefix(&input_view, IREE_SV("@"))) {
      IREE_RETURN_IF_ERROR(iree_tooling_load_ndarrays_from_file(
          input_view, device_allocator, variant_list.get()));
      continue;
    } else if (iree_string_view_equal(input_view, IREE_SV("(null)")) ||
               iree_string_view_equal(input_view, IREE_SV("(ignored)"))) {
      iree_vm_ref_t null_ref = iree_vm_ref_null();
      IREE_RETURN_IF_ERROR(
          iree_vm_list_push_ref_retain(variant_list.get(), &null_ref));
      continue;
    }
    bool has_equal =
        iree_string_view_find_char(input_view, '=', 0) != IREE_STRING_VIEW_NPOS;
    bool has_x =
        iree_string_view_find_char(input_view, 'x', 0) != IREE_STRING_VIEW_NPOS;
    if (has_equal || has_x) {
      // Buffer view (either just a shape or a shape=value) or buffer.
      bool is_storage_reference = iree_string_view_consume_prefix(
          &input_view, iree_make_cstring_view("&"));
      iree_hal_buffer_view_t* buffer_view = nullptr;
      bool has_at = iree_string_view_find_char(input_view, '@', 0) !=
                    IREE_STRING_VIEW_NPOS;
      if (has_at) {
        // Referencing an external file; split into the portion used to
        // initialize the buffer view and the file contents.
        iree_string_view_t metadata, file_path;
        iree_string_view_split(input_view, '@', &metadata, &file_path);
        iree_string_view_consume_suffix(&metadata, iree_make_cstring_view("="));
        IREE_RETURN_IF_ERROR(iree_create_buffer_view_from_file(
            metadata, file_path, device_allocator, &buffer_view));
      } else {
        IREE_RETURN_IF_ERROR(iree_hal_buffer_view_parse(
                                 input_view, device_allocator, &buffer_view),
                             "parsing value '%.*s'", (int)input_view.size,
                             input_view.data);
      }
      if (is_storage_reference) {
        // Storage buffer reference; just take the storage for the buffer view -
        // it'll still have whatever contents were specified (or 0) but we'll
        // discard the metadata.
        auto buffer_ref = iree_hal_buffer_retain_ref(
            iree_hal_buffer_view_buffer(buffer_view));
        iree_hal_buffer_view_release(buffer_view);
        IREE_RETURN_IF_ERROR(
            iree_vm_list_push_ref_move(variant_list.get(), &buffer_ref));
      } else {
        auto buffer_view_ref = iree_hal_buffer_view_move_ref(buffer_view);
        IREE_RETURN_IF_ERROR(
            iree_vm_list_push_ref_move(variant_list.get(), &buffer_view_ref));
      }
    } else {
      // Scalar.
      bool has_dot = iree_string_view_find_char(input_view, '.', 0) !=
                     IREE_STRING_VIEW_NPOS;
      iree_vm_value_t val;
      if (has_dot) {
        // Float.
        val = iree_vm_value_make_f32(0.0f);
        if (!iree_string_view_atof(input_view, &val.f32)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "parsing value '%.*s' as f32",
                                  (int)input_view.size, input_view.data);
        }
      } else {
        // Integer.
        val = iree_vm_value_make_i32(0);
        if (!iree_string_view_atoi_int32(input_view, &val.i32)) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "parsing value '%.*s' as i32",
                                  (int)input_view.size, input_view.data);
        }
      }
      IREE_RETURN_IF_ERROR(iree_vm_list_push_value(variant_list.get(), &val));
    }
  }
  *out_list = variant_list.release();
  return OkStatus();
}

}  // namespace iree
