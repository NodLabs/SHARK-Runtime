# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_local_py_test(
  NAME
    bisection_test
  SRC
    bisection_test.py
  PACKAGE_DIRS
  "${IREE_BINARY_DIR}/compiler/bindings/python"
  "${IREE_BINARY_DIR}/runtime/bindings/python"
  "${IREE_BINARY_DIR}/compiler/plugins/split_mlir"
)
