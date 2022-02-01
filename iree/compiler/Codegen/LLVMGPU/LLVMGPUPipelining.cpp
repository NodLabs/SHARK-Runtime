// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

//====---------------------------------------------------------------------===//
// Pass to pipeline copy to shared memory for matmul op.
//====---------------------------------------------------------------------===//

namespace mlir {
namespace iree_compiler {

static const StringLiteral kPipeliningLoopMarker = "__pipelining_K_loop__";
static const StringLiteral kPipeliningGlobalLoad = "__pipelining_global_load__";

/// Helper to recursively add operation dependencies within `block` to `dep`
/// set.
static void addDepOps(llvm::SmallDenseSet<Operation*>& dep, Operation* op,
                      Block* block) {
  if (!dep.insert(op).second) return;
  for (Value operand : op->getOperands()) {
    Operation* defOp = operand.getDefiningOp();
    if (defOp && defOp->getBlock() == block) addDepOps(dep, defOp, block);
  }
}

/// Assign stages to the loop ops. Simple logic for now, put load from global
/// memory in stage 0 and the rest in stage 1.
static void getPipelineStages(
    scf::ForOp forOp, std::vector<std::pair<Operation*, unsigned>>& ops) {
  if (!forOp->hasAttr(kPipeliningLoopMarker)) return;

  // Track dependencies of the global memory load.
  llvm::SmallDenseSet<Operation*> loadDep;
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (op.hasAttr(kPipeliningGlobalLoad)) {
      addDepOps(loadDep, &op, forOp.getBody());
    }
  }
  // Create a modulo schedule with loads from global memory and the operations
  // it depends on in stage 0. Store to shared memory and computation are in
  // stage 1. In order to have a correct scheduling even with back edges we
  // order stages in decreasing order.
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (loadDep.count(&op)) ops.push_back(std::make_pair(&op, 0));
  }
  for (Operation& op : forOp.getBody()->getOperations()) {
    if (!loadDep.count(&op) && !isa<scf::YieldOp>(op))
      ops.push_back(std::make_pair(&op, 4));
  }
}


namespace {

struct VectorCopyToGPU : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.permutation_map().isMinorIdentity() ||
        op.getShapedType().cast<MemRefType>().getMemorySpaceAsInt() != 3)
      return failure();
    auto read = op.vector().getDefiningOp<vector::TransferReadOp>();
    if (!read || read.getVectorType() != op.getVectorType() ||
        !read.permutation_map().isMinorIdentity())
      return failure();
    if (read.getVectorType().getNumElements() > 4 ||
        !read.getVectorType().getElementType().isF32())
      return failure();
    rewriter.replaceOpWithNewOp<gpu::DeviceAsyncCpOp>(
        op, op.source(), op.indices(), read.source(), read.indices(),
        rewriter.getIndexAttr(read.getVectorType().getNumElements() * 4));
    return success();
  }
};



struct LLVMGPUPipeliningPass
    : public LLVMGPUPipeliningBase<LLVMGPUPipeliningPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext* context = &getContext();
    if(1)
    {
      RewritePatternSet asyncTransform(context);
      asyncTransform.add<VectorCopyToGPU>(context);
      if (failed(applyPatternsAndFoldGreedily(funcOp,
                                              std::move(asyncTransform)))) {
        return signalPassFailure();
      }
      gpu::BarrierOp lastBarrier;
      funcOp.walk(
          [&lastBarrier](gpu::BarrierOp barrier) { lastBarrier = barrier; });
      OpBuilder builder(lastBarrier);
      Value token = builder.create<gpu::DeviceAsyncCommitOp>(lastBarrier.getLoc());
      builder.create<gpu::DeviceAsyncWaitOp>(lastBarrier.getLoc(), token, nullptr);
    }
    // Mark the loop with shared memory copy for pipelining.
    funcOp.walk([](scf::ForOp forOp) {
      bool copyToWorkgroupMemory = false;
      OpBuilder builder(forOp.getContext());
      for (Operation& op : forOp.getBody()->getOperations()) {
        // Pipeline the most inner for op that should be a flat region.
        if (op.getNumRegions() > 0) return;
        if (isa<gpu::DeviceAsyncCpOp, gpu::DeviceAsyncCommitOp>(op)) {
          copyToWorkgroupMemory = true;
          op.setAttr(kPipeliningGlobalLoad, builder.getUnitAttr());
          continue;
        }
        auto ld = dyn_cast<vector::TransferReadOp>(op);
        if (!ld) continue;
        unsigned ldAddSpace =
            ld.source().getType().cast<MemRefType>().getMemorySpaceAsInt();
        if (ldAddSpace != 0 || !ld->hasOneUse()) continue;
        auto st =
            dyn_cast<vector::TransferWriteOp>(ld->use_begin()->getOwner());
        if (!st) continue;
        unsigned stAddSpace =
            st.source().getType().cast<MemRefType>().getMemorySpaceAsInt();
        if (stAddSpace != 3) continue;
        copyToWorkgroupMemory = true;
        ld->setAttr(kPipeliningGlobalLoad, builder.getUnitAttr());
      }
      if (copyToWorkgroupMemory) {
        forOp->setAttr(kPipeliningLoopMarker, builder.getUnitAttr());
      }
    });
    scf::PipeliningOption options;
    options.getScheduleFn = getPipelineStages;
    RewritePatternSet pipeliningPatterns(context);
    scf::populateSCFLoopPipeliningPatterns(pipeliningPatterns, options);
    if (failed(applyPatternsAndFoldGreedily(funcOp,
                                            std::move(pipeliningPatterns)))) {
      return signalPassFailure();
    }
    int count = 4;
    funcOp.walk([&count](gpu::DeviceAsyncWaitOp waitOp) {
      OpBuilder b(waitOp);
      waitOp->setAttr(waitOp.numGroupsAttrName(), b.getI32IntegerAttr(count--));
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createLLVMGPUPipeliningPass() {
  return std::make_unique<LLVMGPUPipeliningPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
