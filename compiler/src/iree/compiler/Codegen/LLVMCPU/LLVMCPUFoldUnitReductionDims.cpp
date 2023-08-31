// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- FoldUnitExtentDims.cpp - Pass to fold unit extent dims of tensors -===//
//
// Light weight wrapper to call the patterns to fold unit extent dims with
// IREE control.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"

#define DEBUG_TYPE "iree-llvmcpu-fold-unit-reduction-dims"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace iree_compiler {


// Helper to 
static std::optional<std::pair<Value, AffineMap>> 
reshapeVector(PatternRewriter &rewriter,
              Location loc,
              Value vector,
              vector::ContractionOp contractOp,
              AffineMap indexingMap) {
  SmallVector<int64_t> contractShape = *contractOp.getShapeForUnroll();
  SmallVector<vector::IteratorType> iteratorTypes = contractOp.getIteratorTypesArray();

  SmallVector<int64_t> dstShape;
  SmallVector<int64_t> dstExprsIndices;
  for (const auto &expr : enumerate(indexingMap.getResults())) {
    if (auto dimExpr = expr.value().dyn_cast<AffineDimExpr>()) {
      if (contractShape[dimExpr.getPosition()] != 1 ||
          iteratorTypes[dimExpr.getPosition()] == vector::IteratorType::parallel) {
        dstShape.push_back(contractShape[dimExpr.getPosition()]);
        dstExprsIndices.push_back(dimExpr.getPosition());
      }
    } else {
      return std::nullopt;
    }
  }

  unsigned numUnitReductions = 0;
  for (const auto size : enumerate(contractShape)) {
    if (size.value() == 1 &&
        iteratorTypes[size.index()] == vector::IteratorType::reduction){
      numUnitReductions++;
      for (int64_t &index : dstExprsIndices) {
        if (index > size.index()) {
          index--;
        }
      }
    }
  }

  LDBG("numUnitReductions: " << numUnitReductions);

  SmallVector<AffineExpr> dstExprs;
  for (int64_t index : dstExprsIndices){
    dstExprs.push_back(rewriter.getAffineDimExpr(index));
  }

  if (dstShape.size() == indexingMap.getResults().size()) {
    return std::make_pair(vector,
                          indexingMap);
  }

  VectorType inputVecType = llvm::cast<VectorType>(vector.getType());
  VectorType dstType = VectorType::get(dstShape,
                                       inputVecType.getElementType());
  Value shapeCastResult = rewriter.create<vector::ShapeCastOp>(loc, dstType, vector);
  AffineMap newIndexingMap = AffineMap::get(/*dimCount=*/contractShape.size() - numUnitReductions,
                                            /*symCount=*/0, dstExprs, contractOp.getContext());
  LDBG("New AffineMap:" << newIndexingMap);
  return std::make_pair(shapeCastResult, newIndexingMap);
}


class DropVectorContractUnitReductionDims final
    : public OpRewritePattern<vector::ContractionOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override{
    Value lhs = contractOp.getLhs();
    Value rhs = contractOp.getRhs();
    Value acc = contractOp.getAcc();
    auto indexingMaps = contractOp.getIndexingMapsArray();
    auto iteratorTypes = contractOp.getIteratorTypesArray();
    auto lhsIndexingMap = indexingMaps[0];
    auto rhsIndexingMap = indexingMaps[1];
    auto accIndexingMap = indexingMaps[2];
    SmallVector<int64_t> contractDims = *contractOp.getShapeForUnroll();
    Location loc = contractOp.getLoc();

    // Fail if reduction dimensions are not innermost
    unsigned numParallel = 0;
    unsigned numReduction = 0;
    for (int i = 0; i < contractDims.size(); i++){
      if (iteratorTypes[i] == vector::IteratorType::parallel) {
        if (numReduction){
          return failure();
        }
        numParallel++;
      }
      else {
        numReduction++;
      }
    }

    // Create vector.shape_cast ops to fold unit reduction dims
    Value newLhs;
    AffineMap newLhsMap;
    auto maybeNewLhs = reshapeVector(rewriter, loc, lhs, contractOp, lhsIndexingMap);
    if (maybeNewLhs) {
      newLhs = maybeNewLhs.value().first;
      newLhsMap = maybeNewLhs.value().second;
    } else {
      return failure();
    }
    LDBG("Added lhs vector.shape_cast:\n" << newLhs);
    Value newRhs;
    AffineMap newRhsMap;
    auto maybeNewRhs = reshapeVector(rewriter, loc, rhs, contractOp, rhsIndexingMap);
    if (maybeNewRhs) {
      newRhs = maybeNewRhs.value().first;
      newRhsMap = maybeNewRhs.value().second;
    } else {
      return failure();
    }
    LDBG("Added rhs vector.shape_cast:\n" << newLhs);
    // Fail if vector.shape_cast ops can't be formed due to no foldable unit dims
    if (newLhs == lhs && newRhs == rhs) {
      return failure();
    }

    // Build new iterator types without unit dim iterators
    SmallVector<vector::IteratorType> newIteratorTypes;
    for (const auto dim : llvm::enumerate(contractDims)) {
      if (iteratorTypes[dim.index()] == vector::IteratorType::parallel) {
        newIteratorTypes.push_back(vector::IteratorType::parallel);
      } else if (dim.value() != 1) {
        newIteratorTypes.push_back(vector::IteratorType::reduction);
      }
    }
    auto newAccMap = AffineMap::get(/*dimCount=*/newIteratorTypes.size(),
                                    /*symCount=*/0, accIndexingMap.getResults(), 
                                    contractOp.getContext());
    SmallVector<AffineMap> newMaps = {newLhsMap, newRhsMap, newAccMap};
    auto newContract = rewriter.replaceOpWithNewOp<vector::ContractionOp>(
      contractOp, newLhs, newRhs, acc,
        rewriter.getAffineMapArrayAttr(newMaps), 
        rewriter.getArrayAttr(llvm::to_vector(llvm::map_range(
            newIteratorTypes, [&](vector::IteratorType t) -> mlir::Attribute {
              return vector::IteratorTypeAttr::get(rewriter.getContext(), t);
            }))));

    LDBG("Replaced vector.contract:\n" << newContract);
    
    return success();
  }
};

namespace {
struct LLVMCPUFoldUnitReductionDimsPass
    : public LLVMCPUFoldUnitReductionDimsBase<LLVMCPUFoldUnitReductionDimsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    vector::VectorDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LLVMCPUFoldUnitReductionDimsPass::runOnOperation() {
  Operation *funcOp = getOperation();
  MLIRContext *context = &getContext();
  RewritePatternSet foldUnitDimsPatterns(context);
  foldUnitDimsPatterns.add<DropVectorContractUnitReductionDims>(context);
  if (failed(applyPatternsAndFoldGreedily(funcOp,
                                          std::move(foldUnitDimsPatterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUFoldUnitReductionDimsPass() {
  return std::make_unique<LLVMCPUFoldUnitReductionDimsPass>();
}

void populateFoldUnitReductionDimsPatterns(RewritePatternSet &patterns, MLIRContext *context){
  patterns.add<DropVectorContractUnitReductionDims>(context);
}

} // namespace iree_compiler
} // namespace mlir
