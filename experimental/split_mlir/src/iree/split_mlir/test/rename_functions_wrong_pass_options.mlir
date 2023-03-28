// RUN: ( \
// RUN:   iree-opt \
// RUN:     --split-input-file \
// RUN:     --iree-plugin=split_mlir \
// RUN:     --pass-pipeline="builtin.module(iree-rename-functions{functions=one,two new-names=one})" %s 2>&1 \
// RUN:   || true \
// RUN: ) \
// RUN: | FileCheck %s

// CHECK-LABEL{LITERAL}: Pass options functions and new-names must have the same number of elements.
