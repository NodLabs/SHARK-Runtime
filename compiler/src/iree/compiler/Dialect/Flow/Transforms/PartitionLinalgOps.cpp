// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
/// A pattern to partition linalg ops based on user annotations
class PartitionUsingAnnotations : public OpInterfaceRewritePattern<linalg::LinalgOp> {
 public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("devices")) {
      //auto tileSizes = op->getAttr();
      //auto result = rewriter.create<tensor::ExtractSliceOp>();
      //result = rewriter.create<tensor::InsertSliceOp>();
    }
    return failure();
  }
};

class PartitionLinalgOpsPass : public PartitionLinalgOpsBase<PartitionLinalgOpsPass> {
 public:
  PartitionLinalgOpsPass() {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<PartitionUsingAnnotations>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }

};

}  // namespace

std::unique_ptr<Pass> createPartitionLinalgOpsUsingAnnotationsPass() {
  return std::make_unique<PartitionLinalgOpsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
