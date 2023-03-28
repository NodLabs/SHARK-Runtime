// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CHECK-LABEL: func.func nested @function_to_export_sin
func.func @function_to_export_sin(%arg0: f32) -> f32 {
//  CHECK-NOT: {
  %res = math.sin %arg0 : f32
  return %res : f32
}

// CHECK-LABEL: func.func nested @function_to_export_cos
func.func @function_to_export_cos(%arg0: f32) -> f32 {
//  CHECK-NOT: {
  %res = math.cos %arg0 : f32
  return %res : f32
}

// CHECK-LABEL: func.func @caller
func.func @caller(%arg0: f32) -> f32 {
//  CHECK-NEXT: call @function_to_export_sin
  %0 = call @function_to_export_sin(%arg0) : (f32) -> f32
//  CHECK-NEXT: call @function_to_export_cos
  %res = call @function_to_export_cos(%0) : (f32) -> f32
  return %res : f32
}
