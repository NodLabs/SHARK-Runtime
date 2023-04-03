// RUN: iree-opt \
// RUN:   --split-input-file \
// RUN:   --iree-plugin=split_mlir \
// RUN:   --pass-pipeline="builtin.module(iree-rename-functions{functions=func_to_rename_1,func_to_rename_2 new-names=renamed_func_1,renamed_func_2})" %s \
// RUN: | FileCheck %s

// Functions the must be rename.
// CHECK-LABEL: func.func @renamed_func_1
func.func @func_to_rename_1(%arg0: f32) -> f32 {
  return %arg0 : f32
}
// CHECK-LABEL: func.func @renamed_func_2
//  CHECK-SAME: ([[ARG0:%.+]]: f32) -> f32
func.func @func_to_rename_2(%arg0: f32) -> f32 {
//  CHECK-NEXT: [[RES:%.+]] = call @renamed_func_1([[ARG0]])
  %res = call @func_to_rename_1(%arg0) : (f32) -> f32
//  CHECK-NEXT: return [[RES]] : f32
  return %res : f32
}

// -----

// Function that must not be rename.
// CHECK-LABEL: func.func @do_not_rename
func.func @do_not_rename() {
  return
}
