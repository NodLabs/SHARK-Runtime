// RUN: iree-opt --linalg-fold-unit-extent-dims --canonicalize --iree-flow-raise-special-ops --canonicalize %s | FileCheck %s

// CHECK-LABEL: @attention_block
//  CHECK-SAME: %[[ARG0:.+]]: tensor<10x4096x64xf16>
//  CHECK-SAME: %[[ARG1:.+]]: tensor<10x4096x64xf16>
//  CHECK-SAME: %[[ARG2:.+]]: tensor<10x4096x64xf16>
//       CHECK:   %[[C:.+]] = arith.constant 0.000000e+00 : f16
//       CHECK:   %[[E:.+]] = tensor.empty() : tensor<10x4096x64xf16>
//       CHECK:   %[[F:.+]] = linalg.fill ins(%cst : f16) outs(%0 : tensor<10x4096x64xf16>) -> tensor<10x4096x64xf16>
//       CHECK:   %[[A:.+]] = iree_linalg_ext.attention ins(%arg0, %arg1, %arg2 : tensor<10x4096x64xf16>, tensor<10x4096x64xf16>, tensor<10x4096x64xf16>) outs(%1 : tensor<10x4096x64xf16>) -> tensor<10x4096x64xf16>
//       CHECK:   return %[[S]] : tensor<10x4096x64xf16>

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
func.func @attention_block(%arg0: tensor<10x4096x64xf16>, %arg1: tensor<10x4096x64xf16>, %arg2: tensor<10x4096x64xf16>) -> tensor<10x4096x64xf16> {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant -6.550400e+04 : f16
  %cst_1 = arith.constant 1.000000e+00 : f16
  %0 = tensor.empty() : tensor<10x4096x4096xf16>
  %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<10x4096x4096xf16>) -> tensor<10x4096x4096xf16>
  %2 = tensor.empty() : tensor<10x4096x64xf16>
  %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<10x4096x64xf16>) -> tensor<10x4096x64xf16>
  %4 = tensor.empty() : tensor<10x4096x1xi64>
  %5 = linalg.fill ins(%c0_i64 : i64) outs(%4 : tensor<10x4096x1xi64>) -> tensor<10x4096x1xi64>
  %6 = tensor.empty() : tensor<10x4096x1xf16>
  %7 = linalg.fill ins(%cst_0 : f16) outs(%6 : tensor<10x4096x1xf16>) -> tensor<10x4096x1xf16>
  %8 = linalg.fill ins(%cst : f16) outs(%6 : tensor<10x4096x1xf16>) -> tensor<10x4096x1xf16>
  %9 = tensor.empty() : tensor<10x4096x4096xf16>
  %10 = linalg.fill ins(%cst : f16) outs(%9 : tensor<10x4096x4096xf16>) -> tensor<10x4096x4096xf16>
  %11 = tensor.empty() : tensor<10x64x4096xf16>
  %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<10x4096x64xf16>) outs(%11 : tensor<10x64x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    linalg.yield %in : f16
  } -> tensor<10x64x4096xf16>
  %13 = linalg.batch_matmul ins(%arg0, %12 : tensor<10x4096x64xf16>, tensor<10x64x4096xf16>) outs(%1 : tensor<10x4096x4096xf16>) -> tensor<10x4096x4096xf16>
  %14 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%13 : tensor<10x4096x4096xf16>) outs(%9 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %22 = arith.mulf %in, %cst_1 : f16
    linalg.yield %22 : f16
  } -> tensor<10x4096x4096xf16>
  %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14, %10 : tensor<10x4096x4096xf16>, tensor<10x4096x4096xf16>) outs(%9 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %in_2: f16, %out: f16):
    %22 = arith.mulf %in_2, %cst : f16
    %23 = arith.addf %in, %22 : f16
    linalg.yield %23 : f16
  } -> tensor<10x4096x4096xf16>
  %16:2 = linalg.generic {indexing_maps = [#map, #map2, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : tensor<10x4096x4096xf16>) outs(%7, %5 : tensor<10x4096x1xf16>, tensor<10x4096x1xi64>) {
  ^bb0(%in: f16, %out: f16, %out_2: i64):
    %22 = linalg.index 2 : index
    %23 = arith.index_cast %22 : index to i64
    %24 = arith.maxf %in, %out : f16
    %25 = arith.cmpf ogt, %in, %out : f16
    %26 = arith.select %25, %23, %out_2 : i64
    linalg.yield %24, %26 : f16, i64
  } -> (tensor<10x4096x1xf16>, tensor<10x4096x1xi64>)
  %17 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %16#0 : tensor<10x4096x4096xf16>, tensor<10x4096x1xf16>) outs(%9 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %in_2: f16, %out: f16):
    %22 = arith.subf %in, %in_2 : f16
    linalg.yield %22 : f16
  } -> tensor<10x4096x4096xf16>
  %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17 : tensor<10x4096x4096xf16>) outs(%9 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %out: f16):
    %22 = math.exp %in : f16
    linalg.yield %22 : f16
  } -> tensor<10x4096x4096xf16>
  %19 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%18 : tensor<10x4096x4096xf16>) outs(%8 : tensor<10x4096x1xf16>) {
  ^bb0(%in: f16, %out: f16):
    %22 = arith.addf %in, %out : f16
    linalg.yield %22 : f16
  } -> tensor<10x4096x1xf16>
  %20 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %19 : tensor<10x4096x4096xf16>, tensor<10x4096x1xf16>) outs(%9 : tensor<10x4096x4096xf16>) {
  ^bb0(%in: f16, %in_2: f16, %out: f16):
    %22 = arith.divf %in, %in_2 : f16
    linalg.yield %22 : f16
  } -> tensor<10x4096x4096xf16>
  %21 = linalg.batch_matmul ins(%20, %arg2 : tensor<10x4096x4096xf16>, tensor<10x4096x64xf16>) outs(%3 : tensor<10x4096x64xf16>) -> tensor<10x4096x64xf16>
  return %21 : tensor<10x4096x64xf16>
}
