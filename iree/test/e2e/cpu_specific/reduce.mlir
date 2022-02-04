
func @multi_result() -> (tensor<128xf32>) {
  %in = util.unfoldable_constant dense<1.0> : tensor<128x384xf32>
  %cst_6 = arith.constant dense<3.840000e+02> : tensor<128xf32>
  %init = linalg.init_tensor [128] : tensor<128xf32>
  %r = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%in : tensor<128x384xf32>) outs(%init : tensor<128xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      %2 = arith.addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
    } -> tensor<128xf32>
  %init2 = linalg.init_tensor [128] : tensor<128xf32>
  %d = linalg.generic {indexing_maps = [
    affine_map<(d0) -> (d0)>,affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%r, %cst_6 : tensor<128xf32>, tensor<128xf32>) outs(%init2 : tensor<128xf32>) {
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32):  // no predecessors
      %3 = arith.divf %arg7, %arg8 : f32
      linalg.yield %3 : f32
    } -> tensor<128xf32>
  return %d : tensor<128xf32>
}