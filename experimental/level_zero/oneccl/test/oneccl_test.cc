// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <iree/runtime/api.h>
#include <iree/testing/gtest.h>
#include <iree/testing/status_matchers.h>

#include <memory>
#include <sstream>
#include <string>

namespace {

template <typename T, typename Deleter>
std::unique_ptr<T, Deleter> make_unique(T* p, Deleter d) {
  return std::unique_ptr<T, Deleter>(p, d);
}

void get_rank_and_world_size(iree_hal_driver_t* driver, size_t* rank,
                             size_t* world_size) {
  iree_allocator_t host_allocator = iree_allocator_system();

  iree_hal_device_t* device = nullptr;
  ASSERT_EQ(
      iree_hal_driver_create_default_device(driver, host_allocator, &device),
      iree_ok_status());
  auto device_deleter = make_unique<iree_hal_device_t>(
      device, [](iree_hal_device_t* p) { iree_hal_device_release(p); });

  // Create channel.
  iree_hal_channel_params_t channel_params;
  iree_hal_channel_t* channel = NULL;
  channel_params.rank = IREE_HAL_CHANNEL_RANK_DEFAULT;
  IREE_ASSERT_OK(iree_hal_channel_create(device, iree_hal_queue_affinity_t(0),
                                         channel_params, &channel));
  auto channel_deleter = make_unique<iree_hal_channel_t>(
      channel, [](iree_hal_channel_t* p) { iree_hal_channel_release(p); });
  *world_size = iree_hal_channel_count(channel);
  *rank = iree_hal_channel_rank(channel);
}

void make_device(iree_hal_driver_t* driver, iree_hal_device_t** out_device) {
  iree_allocator_t host_allocator = iree_allocator_system();

  size_t rank = 0;
  size_t world_size = 0;
  get_rank_and_world_size(driver, &rank, &world_size);

  // Query the devices from the driver.
  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t* device_infos = NULL;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
      driver, host_allocator, &device_info_count, &device_infos));
  auto device_infos_deleter = make_unique<iree_hal_device_info_t>(
      device_infos, [host_allocator](iree_hal_device_info_t* p) {
        iree_allocator_free(host_allocator, p);
      });

  // Create the device.
  size_t device_ordinal = rank % device_info_count;
  IREE_ASSERT_OK(iree_hal_driver_create_device_by_ordinal(
      driver, device_ordinal,
      /*param_count=*/0, /*params=*/0, host_allocator, out_device));
}

}  // namespace

TEST(OneCcl, AllGather) {
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);

  static const char* driver_name = "level_zero";

  iree_allocator_t host_allocator = iree_allocator_system();

  // Make instance.
  iree_runtime_instance_t* instance = nullptr;
  IREE_ASSERT_OK(iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance));
  auto instance_deleter = make_unique<iree_runtime_instance_t>(
      instance,
      [](iree_runtime_instance_t* p) { iree_runtime_instance_release(p); });

  // Make Level Zero driver.
  iree_hal_driver_registry_t* driver_registry =
      iree_runtime_instance_driver_registry(instance);
  iree_hal_driver_t* driver = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_registry_try_create(
      driver_registry, iree_make_cstring_view(driver_name), host_allocator,
      &driver));
  auto driver_deleter = make_unique<iree_hal_driver_t>(
      driver, [](iree_hal_driver_t* p) { iree_hal_driver_release(p); });

  iree_hal_device_t* device = nullptr;
  make_device(driver, &device);
  auto device_deleter = make_unique<iree_hal_device_t>(
      device, [](iree_hal_device_t* p) { iree_hal_device_release(p); });

  // Create channel.
  iree_hal_channel_params_t channel_params;
  iree_hal_channel_t* channel = NULL;
  channel_params.rank = IREE_HAL_CHANNEL_RANK_DEFAULT;
  IREE_ASSERT_OK(iree_hal_channel_create(device, iree_hal_queue_affinity_t(0),
                                         channel_params, &channel));
  auto channel_deleter = make_unique<iree_hal_channel_t>(
      channel, [](iree_hal_channel_t* p) { iree_hal_channel_release(p); });
  size_t ranks_count = iree_hal_channel_count(channel);
  int32_t rank = iree_hal_channel_rank(channel);

  // Create command buffer.
  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device, iree_hal_command_buffer_mode_t(0), IREE_HAL_COMMAND_CATEGORY_ANY,
      iree_hal_queue_affinity_t(0),
      /*binding_capacity=*/0, &command_buffer));
  auto command_buffer_deleter = make_unique<iree_hal_command_buffer_t>(
      command_buffer,
      [](iree_hal_command_buffer_t* p) { iree_hal_command_buffer_release(p); });

  using data_type = int32_t;
  iree_hal_allocator_t* device_allocator = iree_hal_device_allocator(device);

  // Create input device buffer.
  iree_hal_buffer_t* in_device_buffer = nullptr;
  iree_hal_buffer_params_t in_buffer_params;
  in_buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  in_buffer_params.access =
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
  in_buffer_params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  in_buffer_params.queue_affinity = 0;
  in_buffer_params.min_alignment = sizeof(data_type);
  std::vector<data_type> in_buffer_data(
      {(rank + 1) * 100, (rank + 1) * 100 + 1});
  size_t element_count = in_buffer_data.size();
  iree_const_byte_span_t in_buffer_intial_data;
  in_buffer_intial_data.data =
      reinterpret_cast<uint8_t*>(in_buffer_data.data());
  in_buffer_intial_data.data_length = element_count * sizeof(data_type);
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator, in_buffer_params, in_buffer_intial_data.data_length,
      in_buffer_intial_data, &in_device_buffer));
  auto in_buffer_deleter = make_unique<iree_hal_buffer_t>(
      in_device_buffer,
      [](iree_hal_buffer_t* p) { iree_hal_buffer_release(p); });

  // Create output device buffer.
  iree_hal_buffer_t* out_device_buffer = nullptr;
  iree_hal_buffer_params_t out_buffer_params;
  out_buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
  out_buffer_params.access =
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
  out_buffer_params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  out_buffer_params.queue_affinity = 0;
  out_buffer_params.min_alignment = sizeof(data_type);
  std::vector<data_type> out_buffer_data(element_count * ranks_count, -1);
  iree_const_byte_span_t out_buffer_intial_data;
  out_buffer_intial_data.data =
      reinterpret_cast<uint8_t*>(out_buffer_data.data());
  out_buffer_intial_data.data_length =
      ranks_count * element_count * sizeof(data_type);
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator, out_buffer_params, out_buffer_intial_data.data_length,
      out_buffer_intial_data, &out_device_buffer));
  auto out_buffer_deleter = make_unique<iree_hal_buffer_t>(
      out_device_buffer,
      [](iree_hal_buffer_t* p) { iree_hal_buffer_release(p); });

  // Schedule collective op.
  iree_hal_collective_op_t collective_op;
  collective_op.kind = IREE_HAL_COLLECTIVE_KIND_ALL_GATHER;
  collective_op.reduction = 0;  // No reduction.
  collective_op.element_type = IREE_HAL_COLLECTIVE_ELEMENT_TYPE_SINT_32;
  iree_hal_buffer_binding_t send_binding;
  send_binding.buffer = in_device_buffer;
  send_binding.length = in_device_buffer->byte_length;
  send_binding.offset = 0;
  iree_hal_buffer_binding_t recv_binding;
  recv_binding.buffer = out_device_buffer;
  recv_binding.length = out_device_buffer->byte_length;
  recv_binding.offset = 0;
  IREE_ASSERT_OK(iree_hal_command_buffer_collective(
      command_buffer, channel, collective_op, /*param=*/0, send_binding,
      recv_binding, element_count));

  // Create output host buffer.
  iree_hal_buffer_t* out_host_buffer = nullptr;
  iree_hal_buffer_params_t out_host_buffer_params;
  out_host_buffer_params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  out_host_buffer_params.access = IREE_HAL_MEMORY_ACCESS_READ |
                                  IREE_HAL_MEMORY_ACCESS_WRITE |
                                  IREE_HAL_MEMORY_ACCESS_DISCARD;
  out_host_buffer_params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  out_host_buffer_params.queue_affinity = 0;
  out_host_buffer_params.min_alignment = sizeof(data_type);
  std::vector<data_type> out_host_buffer_data(element_count * ranks_count, -1);
  iree_const_byte_span_t out_host_buffer_intial_data;
  out_host_buffer_intial_data.data =
      reinterpret_cast<uint8_t*>(out_host_buffer_data.data());
  out_host_buffer_intial_data.data_length =
      ranks_count * element_count * sizeof(data_type);
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      device_allocator, out_host_buffer_params,
      out_host_buffer_intial_data.data_length, out_host_buffer_intial_data,
      &out_host_buffer));
  auto out_host_buffer_deleter = make_unique<iree_hal_buffer_t>(
      out_host_buffer,
      [](iree_hal_buffer_t* p) { iree_hal_buffer_release(p); });

  // Schedule copying the output from device to host.
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      command_buffer, out_device_buffer, 0, out_host_buffer, 0,
      out_device_buffer->byte_length));

  // Execute command buffer.
  iree_hal_semaphore_list_t wait_semaphore_list;
  wait_semaphore_list.count = 0;
  iree_hal_semaphore_list_t signal_semaphore_list;
  signal_semaphore_list.count = 0;
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device, /*queue_affinity=*/0, wait_semaphore_list, signal_semaphore_list,
      1, &command_buffer));

  iree_hal_buffer_mapping_t out_host_buffer_mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      out_host_buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, /*byte_offset=*/0,
      out_host_buffer->byte_length, &out_host_buffer_mapping));
  auto out_host_buffer_mapping_deleter = make_unique<iree_hal_buffer_mapping_t>(
      &out_host_buffer_mapping, [](iree_hal_buffer_mapping_t* p) {
        IREE_ASSERT_OK(iree_hal_buffer_unmap_range(p));
      });

  std::vector<data_type> expected_output;
  expected_output.reserve(element_count * ranks_count);
  for (size_t i = 0; i < ranks_count; ++i) {
    for (size_t j = 0; j < element_count; ++j) {
      expected_output.push_back(100 * (i + 1) + j);
    }
  }
  std::vector<data_type> output(
      reinterpret_cast<data_type*>(out_host_buffer_mapping.contents.data),
      reinterpret_cast<data_type*>(
          out_host_buffer_mapping.contents.data +
          out_host_buffer_mapping.contents.data_length));
  ASSERT_THAT(output, testing::ElementsAreArray(expected_output));
}
