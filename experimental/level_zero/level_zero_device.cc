// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/level_zero/level_zero_device.h"

#include <locale>
#include <unordered_map>

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
