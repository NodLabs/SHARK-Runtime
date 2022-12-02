// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_CTS_DRIVER_TEST_H_
#define IREE_HAL_CTS_DRIVER_TEST_H_

#include <iostream>
#include <string>
#include <sstream>
#include <memory>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace hal {
namespace cts {

template <typename T, typename Deleter>
std::unique_ptr<T, Deleter> make_unique(T* p, Deleter d) {
  return std::unique_ptr<T, Deleter>(p, d);
}

class driver_test : public CtsTestBase {};

TEST_P(driver_test, QueryAndCreateAvailableDevices) {
  iree_host_size_t device_info_count = 0;
  iree_hal_device_info_t* device_infos = NULL;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
      driver_, iree_allocator_system(), &device_info_count, &device_infos));

  std::cout << "Driver has " << device_info_count << " device(s)";
  for (iree_host_size_t i = 0; i < device_info_count; ++i) {
    std::cout << "  Creating device '"
              << std::string(device_infos[i].name.data,
                             device_infos[i].name.size)
              << "'";
    iree_hal_device_t* device = NULL;
    IREE_ASSERT_OK(iree_hal_driver_create_device_by_id(
        driver_, device_infos[i].device_id, /*param_count=*/0, /*params=*/NULL,
        iree_allocator_system(), &device));
    iree_string_view_t device_id = iree_hal_device_id(device);
    std::cout << "  Created device with id: '"
              << std::string(device_id.data, device_id.size) << "'";
    iree_hal_device_release(device);
  }

  iree_allocator_free(iree_allocator_system(), device_infos);
}

TEST_P(driver_test, CreateDeviceByPathUri) {
  iree_allocator_t host_allocator = iree_allocator_system();

  // Get list of available devices.
  iree_hal_device_info_t* device_infos = nullptr;
  iree_host_size_t device_infos_count = 0;
  IREE_ASSERT_OK(iree_hal_driver_query_available_devices(
                driver_, host_allocator, &device_infos_count, &device_infos));
  auto device_infos_deleter = make_unique<iree_hal_device_info_t>(
      device_infos, [host_allocator](iree_hal_device_info_t* p) {
        iree_allocator_free(host_allocator, p);
      });
  ASSERT_GT(device_infos_count, 0);

  // Create a valid device from URI.
  std::stringstream device_uri;
  device_uri << driver_name_ << "://";
  device_uri << std::string(device_infos[0].path.data,
                            device_infos[0].path.size);
  std::string device_uri_str = device_uri.str();
  iree_hal_device_t* device = nullptr;
  IREE_ASSERT_OK(iree_hal_driver_create_device_by_uri(
                driver_, iree_make_cstring_view(device_uri_str.c_str()),
                host_allocator, &device));
  auto device_deleter = make_unique<iree_hal_device_t>(
      device, [](iree_hal_device_t* p) { iree_hal_device_release(p); });

  // Try create an invalid device from URI.
  std::stringstream invalid_device_uri;
  invalid_device_uri << driver_name_
                     << "://4e5a272e-66a7-11ed-9342-4f1f581f812c";
  std::string invalid_device_uri_str = invalid_device_uri.str();
  iree_hal_device_t* invalid_device = nullptr;
  ASSERT_NE(iree_hal_driver_create_device_by_uri(
                driver_, iree_make_cstring_view(invalid_device_uri_str.c_str()),
                host_allocator, &invalid_device),
            iree_ok_status());
  auto invalid_device_deleter = make_unique<iree_hal_device_t>(
      invalid_device, [](iree_hal_device_t* p) { iree_hal_device_release(p); });
}

}  // namespace cts
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_CTS_DRIVER_TEST_H_
