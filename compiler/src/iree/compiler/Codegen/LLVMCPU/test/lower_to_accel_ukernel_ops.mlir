// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-lower-to-accel-ukernels,cse,canonicalize))" %s | FileCheck %s

func.func @matmul_f32f32f32(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: func @matmul_f32f32f32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "aie_matmul_f32"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]] :
//  CHECK-DAG:       "processor_id"
//  CHECK-DAG:       "processor_data"
//      CHECK:   return %[[MICRO_KERNEL]]
