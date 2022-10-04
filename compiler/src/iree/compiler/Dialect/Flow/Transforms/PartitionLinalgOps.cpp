// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

/// A pattern to propagate device annotations to consumers
class PropagateAnnotations : public OpInterfaceRewritePattern<linalg::LinalgOp> {
 public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("device")) {
      bool annotated{false};
      for (auto *user : op->getUsers()) {
        if (!user->hasAttr("device")) {
          user->setAttr("device", op->getAttr("device"));  
          annotated = true;
        }
        if (!annotated)
          return failure();
      }
      return success();
    }
    return failure();
  }
};

}

namespace {
/// A pattern to partition linalg ops based on user annotations
class PartitionUsingAnnotations : public OpInterfaceRewritePattern<linalg::LinalgOp> {
 public:
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("partition_sizes") && op->hasAttr("devices")) {
      auto devicesAttr = op->getAttr("devices").cast<ArrayAttr>();
      auto partitionSizes = op->getAttr("partition_sizes").cast<ArrayAttr>();
      SmallVector<int64_t> tileSizes;
      for (Attribute val : partitionSizes.getValue())
        tileSizes.push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
      SmallVector<int64_t> numPartitions, devices;
      // Assumes parallel loops come first, then reduction
      for (Attribute val : devicesAttr.getValue()) {
        auto np = val.cast<ArrayAttr>();
        if (np.size() > 0) numPartitions.push_back(np.size());
        for (Attribute device : llvm::reverse(np))
          devices.push_back(device.cast<IntegerAttr>().getValue().getSExtValue());
      }
      op->removeAttr("partition_sizes");
      auto identityLoopOrder =
          llvm::to_vector<4>(llvm::seq<int64_t>(0, tileSizes.size()));

      FailureOr<linalg::TileLoopNest> loopNest =
          linalg::tileConsumerAndFuseProducers(rewriter, op, tileSizes,
                                               identityLoopOrder, llvm::None);
      if (failed(loopNest)) {
        op.emitOpError("failed tiling and fusing producers");
        return failure();
      }

      op->replaceAllUsesWith(loopNest->getRootOpReplacementResults());

      if (!loopNest->getLoopOps().empty()) {
        function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn =
            [devices] (unsigned i, Operation *op, OpBuilder b) {
               static int idx{0};
               if (op->hasAttr("devices")) {
                 op->setAttr("device", b.getI64IntegerAttr(devices[devices.size() - idx - 1]));
                 idx++;
                 op->removeAttr("devices");
               }
            };
        ArrayRef<scf::ForOp> loopOps = loopNest->getLoopOps();
        for (auto loopOp : llvm::enumerate(llvm::reverse(loopOps))) {
          int64_t loopIndex = loopOp.index();
          int64_t unrollFactor = numPartitions[loopOps.size() - loopIndex - 1];
          if (failed(mlir::loopUnrollByFactor(loopOp.value(), unrollFactor, 
                        loopIndex == loopOps.size() - 1 ? annotateFn : nullptr))) {
            ((scf::ForOp)loopOp.value()).emitOpError("failed unrolling");
            return failure();
          }
        }
      }
      return success();
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
    populateExtractFromInsertSliceDestOpPatterns(patterns);
    patterns.insert<PropagateAnnotations>(context);
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
