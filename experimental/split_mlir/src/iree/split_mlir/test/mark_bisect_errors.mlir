// RUN: ( \
// RUN:   iree-opt \
// RUN:     --split-input-file \
// RUN:     --iree-plugin=split_mlir \
// RUN:     --pass-pipeline="builtin.module(func.func(iree-mark-bisect{functions=two_ops,too_few_ops,multiple_ops}))" %s 2>&1 \
// RUN:   || true \
// RUN: ) \
// RUN: | FileCheck %s

// Degenerate case with too few ops should fail.
// CHECK-LABEL{LITERAL}: error: Can't bisect function block with less than 3 operations.
func.func @too_few_ops(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %cts1 = mhlo.constant dense<1.000000e+00> : tensor<2xf32>
  return %cts1 : tensor<2xf32>
}
