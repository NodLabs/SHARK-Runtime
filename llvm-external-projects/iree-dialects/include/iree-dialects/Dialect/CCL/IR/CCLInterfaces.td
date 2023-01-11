//===- CCLInterfaces.td - CCL dialect definition -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the CCL (Collective Communications Library) dialect interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CCL_IR_CCLINTERFACES
#define MLIR_DIALECT_CCL_IR_CCLINTERFACES

include "mlir/IR/OpBase.td"

def CCL_OpInterface : OpInterface<"CCLOp"> {
  let description = [{Implemented by all CCL operations.}];
  let cppNamespace = "::mlir::ccl";
}

#endif // MLIR_DIALECT_CCL_IR_CCLINTERFACES
