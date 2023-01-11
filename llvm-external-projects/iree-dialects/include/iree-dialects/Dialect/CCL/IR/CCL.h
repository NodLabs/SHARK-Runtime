//===- CCL.h - MLIR CCL dialect ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CCL_IR_CCL_H
#define MLIR_DIALECT_CCL_IR_CCL_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Dialect/CCL/IR/CCLInterfaces.h.inc"

#include "mlir/Dialect/CCL/IR/CCLOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/CCL/IR/CCLOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/CCL/IR/CCLOps.h.inc"

#include "mlir/Dialect/CCL/IR/CCLOpsDialect.h.inc"

#endif // MLIR_DIALECT_CCL_IR_CCL_H
