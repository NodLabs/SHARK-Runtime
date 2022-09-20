// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include <iostream>

namespace mlir {
namespace iree_compiler {

namespace {

class FoldExtIntoContract
    : public OpRewritePattern<spirv::JointMatrixLoadINTELOp> {
 public:
  using OpRewritePattern<spirv::JointMatrixLoadINTELOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::JointMatrixLoadINTELOp op,
                                PatternRewriter& rewriter) const override {
    std::cout<<"In the SPIRV level pass for SPIRV\n";
    auto opLayout= op.layout();
    auto result = op.getResult();
    auto jointMatrixType =
          result.getType().dyn_cast<spirv::JointMatrixINTELType>();
    
    auto resultLayout = jointMatrixType.getMatrixLayout();

    if(resultLayout==opLayout){
      return failure();
    }
      auto newJointMatrixType = spirv::JointMatrixINTELType::get(
                jointMatrixType.getElementType(), spirv::Scope::Subgroup, jointMatrixType.getRows(),
                jointMatrixType.getColumns(), opLayout);
      rewriter.replaceOpWithNewOp<spirv::JointMatrixLoadINTELOp> (op,newJointMatrixType,
      op.pointer(),op.stride(), opLayout, spirv::Scope::Subgroup,
                spirv::MemoryAccessAttr(),IntegerAttr());
      return success();

  }
};

class CastJointMatrixLoadProducerPtr
    : public OpRewritePattern<spirv::JointMatrixLoadINTELOp> {
 public:
  using OpRewritePattern<spirv::JointMatrixLoadINTELOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::JointMatrixLoadINTELOp op,
                                PatternRewriter& rewriter) const override {
    
     auto opPointer= op.pointer();
     auto pointerType = opPointer.getType().cast<spirv::PointerType>();
     auto pointerStorage = pointerType.getStorageClass();
     if (pointerStorage == spirv::StorageClass::Generic)
      return failure();
    Location loc = op.getLoc();
    auto newPointerType = spirv::PointerType::get(pointerType.getPointeeType(),spirv::StorageClass::Generic);
    Value newPtr = rewriter.create<spirv::PtrCastToGenericOp>(loc,newPointerType,opPointer);
      rewriter.replaceOpWithNewOp<spirv::JointMatrixLoadINTELOp> (op,op.getResult().getType(),
      newPtr,op.stride(), op.layout(), spirv::Scope::Subgroup,
                spirv::MemoryAccessAttr(),IntegerAttr());
    return success();
  }
};

class CastJointMatrixStoreProducerPtr
    : public OpRewritePattern<spirv::JointMatrixStoreINTELOp> {
 public:
  using OpRewritePattern<spirv::JointMatrixStoreINTELOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spirv::JointMatrixStoreINTELOp op,
                                PatternRewriter& rewriter) const override {
    
     auto opPointer= op.pointer();
     auto pointerType = opPointer.getType().cast<spirv::PointerType>();
     auto pointerStorage = pointerType.getStorageClass();
     if (pointerStorage == spirv::StorageClass::Generic)
      return failure();
    Location loc = op.getLoc();
    auto newPointerType = spirv::PointerType::get(pointerType.getPointeeType(),spirv::StorageClass::Generic);
    Value newPtr = rewriter.create<spirv::PtrCastToGenericOp>(loc,newPointerType,opPointer);
      rewriter.replaceOpWithNewOp<spirv::JointMatrixStoreINTELOp> (op,
      newPtr,op.object(),op.stride(), op.layout(), spirv::Scope::Subgroup,
                spirv::MemoryAccessAttr(),IntegerAttr());
    return success();
  }
};

struct SPIRVMatchJointLoadPass
    : public SPIRVMatchJointLoadBase<SPIRVMatchJointLoadPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    {
      RewritePatternSet patterns(&getContext());
      patterns.add<FoldExtIntoContract>(&getContext());
      patterns.add<CastJointMatrixLoadProducerPtr>(&getContext());
      patterns.add<CastJointMatrixStoreProducerPtr>(&getContext());
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }

    }

  }


};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVMatchJointLoadPass() {
  return std::make_unique<SPIRVMatchJointLoadPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
