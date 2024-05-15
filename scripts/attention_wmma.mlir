func.func @attention(%4: tensor<16x16384x128xf16>, %5: tensor<16x16384x128xf16>, %6: tensor<16x16384x128xf16>) -> tensor<16x16384x128xf16> {
  %c0 = arith.constant 0 : index
  %scale = arith.constant 0.08838834764 : f16
  %7 = tensor.empty() : tensor<16x16384x128xf16>
  %8 = iree_linalg_ext.attention ins(%4, %5, %6, %scale : tensor<16x16384x128xf16>, tensor<16x16384x128xf16>, tensor<16x16384x128xf16>, f16) outs(%7 : tensor<16x16384x128xf16>) -> tensor<16x16384x128xf16>
  return %8 : tensor<16x16384x128xf16>
}
