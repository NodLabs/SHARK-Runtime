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
    else{
      auto newJointMatrixType = spirv::JointMatrixINTELType::get(
                jointMatrixType.getElementType(), spirv::Scope::Subgroup, jointMatrixType.getRows(),
                jointMatrixType.getColumns(), opLayout);
      std::cout<<"old type\n";
      jointMatrixType.dump();
      std::cout<<"\n new type\n";
      newJointMatrixType.dump();
      rewriter.replaceOpWithNewOp<spirv::JointMatrixLoadINTELOp> (op,newJointMatrixType,
      op.pointer(),op.stride(), opLayout, spirv::Scope::Subgroup,
                spirv::MemoryAccessAttr(),IntegerAttr());
      /*rewriter.replaceOpWithNewOp<spirv::JointMatrixLoadINTELOp>(
          op, matType1,spirv::Scope::Subgroup,spirv::MatrixLayout::RowMajor,
          bufferPtr, strideValue,spirv::MemoryAccessAttr(),IntegerAttr());*/
      return success();
    }
    return failure();
    op.dump();
    //jointMatrixType.dump();
    
    /*if(auto readop = dyn_cast<vector::TransferReadOp>(op.getLhs().getDefiningOp())){
    std::cout<<"This is the good case do nothing\n";
    return failure();
    }
    std::cout<<"This case needs handling\n";
    //auto indexingMaps = op.getIndexingMapsArray();
    //auto iteratorTypes = op.getIteratorTypes();
    auto parentOpLhs = op.getLhs().getDefiningOp()->getOperands()[0].getDefiningOp();
auto parentOpRhs = op.getRhs().getDefiningOp()->getOperands()[0].getDefiningOp();

    rewriter.replaceOpWithNewOp<vector::ContractionOp>(op, op.getResultType(),parentOpLhs->getResults()[0],parentOpRhs->getResults()[0] ,op.getAcc(),
        op.getMasks(),
        rewriter.getAffineMapArrayAttr(op.getIndexingMapsArray()),
        op.getIteratorTypes(), op.getKind());
    return success();*/
  }
};



struct SPIRVMatchJointLoadPass
    : public SPIRVMatchJointLoadBase<SPIRVMatchJointLoadPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<FoldExtIntoContract>(&getContext());
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
