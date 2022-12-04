// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_ALLOCATOR_CC_H_
#define IREE_BASE_ALLOCATOR_CC_H_

#ifndef __cplusplus
#error iree::Allocator is only usable in C++ code.
#endif  // !__cplusplus

#include <cstddef>

#include "iree/base/allocator.h"
#include "iree/base/status.h"

#ifdef __cplusplus

namespace iree {

// Wrapps an iree_allocator_t.
// Allocator provides an interface that is compliant with
// the requirements for and C++ std allocator.
template <typename T>
struct Allocator {
  using value_type = T;
  Allocator(iree_allocator_t allocator) : allocator(allocator) {}
  T* allocate(std::size_t n) {
    T* res = nullptr;
    IREE_IGNORE_ERROR(iree_allocator_malloc_uninitialized(
        allocator, n, reinterpret_cast<void**>(&res)));
    return res;
  }
  void deallocate(T* p, std::size_t n) { iree_allocator_free(allocator, p); }

 private:
  iree_allocator_t allocator;
};

}  // namespace iree

#endif  // __cplusplus

#endif  // IREE_BASE_ALLOCATOR_CC_H_
