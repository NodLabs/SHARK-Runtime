// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmcpu-lower-to-accel-ukernels,cse,canonicalize))" %s | FileCheck %s

func.func @mmt4d_f32f32f32(%arg0 : tensor<1x1x?x?xf32>, %arg1 : tensor<1x1x?x?xf32>,
    %arg2 : tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xf32> {
  %0 = linalg.mmt4d ins(%arg0, %arg1 : tensor<1x1x?x?xf32>, tensor<1x1x?x?xf32>)
      outs(%arg2 : tensor<1x1x?x?xf32>) -> tensor<1x1x?x?xf32>
  return %0 : tensor<1x1x?x?xf32>
}

//      CHECK: func @mmt4d_f32f32f32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<1x1x?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x1x?x?xf32>
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//  CHECK-DAG:   %[[C1_i32:.+]] = arith.constant 1
//  CHECK-DAG:   %[[DIM:.+]] = tensor.dim %[[ARG2]], %[[C2]]
//  CHECK-DAG:   %[[DIM_0:.+]] = tensor.dim %[[ARG2]], %[[C3]]
//  CHECK-DAG:   %[[EXTRACTED_SLICE:.+]] = tensor.extract_slice %[[ARG2]]
//  CHECK-DAG:   %[[DIM_1:.+]] = tensor.dim %[[ARG0]], %[[C2]]
//  CHECK-DAG:   %[[DIM_2:.+]] = tensor.dim %[[ARG0]], %[[C3]]
//  CHECK-DAG:   %[[EXTRACTED_SLICE_3:.+]] = tensor.extract_slice %[[ARG0]]
//  CHECK-DAG:   %[[DIM_4:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//  CHECK-DAG:   %[[DIM_5:.+]] = tensor.dim %[[ARG1]], %[[C3]]
//  CHECK-DAG:   %[[EXTRACTED_SLICE_6:.+]] = tensor.extract_slice %[[ARG1]]
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[EXTRACTED_SLICE_3]], %[[C0]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[EXTRACTED_SLICE_6]], %[[C0]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[K:.+]] = tensor.dim %[[EXTRACTED_SLICE_6]], %[[C1]]: tensor<?x?xf32>
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "accel_matmul_f32"
// CHECK-SAME:       ins(%[[EXTRACTED_SLICE_3]], %[[EXTRACTED_SLICE_6]] :
// CHECK-SAME:       outs(%[[EXTRACTED_SLICE]] :
// CHECK-SAME:       (%[[M]], %[[N]], %[[K]] :
//  CHECK-DAG:       "processor_data"
//  CHECK-DAG:       "processor_id"
//  CHECK-DAG:       strided_outer_dims(0)
//  CHECK-DAG:   %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[MICRO_KERNEL]] into %[[ARG2]]: tensor<?x?xf32>
//      CHECK:   return %[[INSERTED_SLICE]]
