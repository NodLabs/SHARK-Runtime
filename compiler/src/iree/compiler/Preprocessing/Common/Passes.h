// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PREPROCESSING_COMMON_PASSES_H_
#define IREE_COMPILER_PREPROCESSING_COMMON_PASSES_H_

#include <functional>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::Preprocessing {

/// Creates a pass to convert linalg convolution ops into linalg.matmul ops
/// using im2col tranformation.
std::unique_ptr<Pass> createConvertConv2DToImg2ColPass();

// Creates a pass to convert linalg NCHW Convolutions to NHWC.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvertConvNchwToNhwcPass();

// Pass to convert a linalg.matmul into linalg.transpose + linalg.matmul.
std::unique_ptr<Pass> createConvertLinalgMatmulToMmtPass();

// Generalizes named op and try to fuse them
std::unique_ptr<Pass> createGeneralizeAndFusePass();

/// Moves the body of the entire function into a single dispatch.
std::unique_ptr<OperationPass<func::FuncOp>>
createMakeSingleDispatchForFunctionPass();

// A pass to generalize all conv-like ops.
std::unique_ptr<Pass> createGeneralizeConvolutionsPass();

// A pass to pad linalg ops to the next integer multiple of `paddingSize`.
std::unique_ptr<Pass> createPadLinalgOpsToIntegerMultiplePass();

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerCommonPreprocessingPasses();

} // namespace mlir::iree_compiler::Preprocessing

#endif // IREE_COMPILER_PREPROCESSING_COMMON_PASSES_H_
