// RUN: ( \
// RUN:   iree-opt \
// RUN:     --split-input-file \
// RUN:     --iree-plugin=split_mlir \
// RUN:     --pass-pipeline="builtin.module(iree-outline-functions)" %s 2>&1 \
// RUN:   || true \
// RUN: ) \
// RUN: | FileCheck %s
 
// CHECK-LABEL{LITERAL}: Unexpected attribute outline_range_first encountered. Possibly unclosed range with outline_range_last.
func.func @unclosed_range_before_opening_another(%arg0: i32, %arg1: i32) {
  %add = arith.addi %arg0, %arg0 {outline_range_first} : i32
  %mul = arith.muli %add, %arg1 {outline_range_first} : i32
  return
}

// -----

// CHECK-LABEL{LITERAL}: Unexpected attribute outline_range_last encountered. Possibly missing outline_range_first.
func.func @unopened_range_before_closing(%arg0: i32, %arg1: i32) {
  %add = arith.addi %arg0, %arg0 {outline_range_last} : i32
  return
}

// -----

// CHECK-LABEL{LITERAL}: Unexpected end of block encountered. Possibly missing outline_range_last.
func.func @unclosed_range_before_end_of_block(%arg0: i32, %arg1: i32) {
  %add = arith.addi %arg0, %arg0 {outline_range_first} : i32
  return
}
