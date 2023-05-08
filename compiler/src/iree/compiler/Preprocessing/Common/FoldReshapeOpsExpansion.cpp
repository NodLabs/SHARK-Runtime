// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

namespace {

struct FoldReshapeOpsExpansionPass
    : FoldReshapeOpsExpansionBase<FoldReshapeOpsExpansionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    Operation *funcOp = getOperation();
    RewritePatternSet fusionPatterns(&getContext());

    linalg::ControlFusionFn fuseByExpansionControlFn =
        [](OpOperand *fusedOperand) {
          Operation *producer = fusedOperand->get().getDefiningOp();
          if (!producer) {
            return false;
          }
          // Do not fuse producer generic op if it has more than one user.
          if (auto producerGenericOp =
                  dyn_cast<linalg::GenericOp>(producer)) {
            return producerGenericOp->hasOneUse();
          }
          // Fuse in all other cases.
          return true;
        };
    linalg::populateFoldReshapeOpsByExpansionPatterns(
        fusionPatterns, fuseByExpansionControlFn);


    GreedyRewriteConfig rewriteConfig;
    rewriteConfig.maxIterations = GreedyRewriteConfig::kNoLimit;
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(fusionPatterns),
                                            rewriteConfig))) {
      funcOp->emitError("failed to apply fusion patterns");
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createFoldReshapeOpsExpansionPass() {
  return std::make_unique<FoldReshapeOpsExpansionPass>();
}

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
