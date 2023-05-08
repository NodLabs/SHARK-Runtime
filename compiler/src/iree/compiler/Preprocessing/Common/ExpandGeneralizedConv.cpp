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

class ExpandGenericConv final : public OpRewritePattern<linalg::GenericOp> {
 public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp convOp,
                                PatternRewriter &rewriter) const override {
    // Only expand generalized conv whose producer is the dequantize filter
    if (!isa<linalg::ConvolutionOpInterface>(*convOp)
        && !linalg::detail::getMatchConvolutionMessage(
            mlir::linalg::detail::isConvolutionInterfaceImpl(convOp)).empty()) {
       return failure();
    }
    if (convOp.getNumLoops() != 7) return failure();

    auto producer = convOp->getOpOperands()[1].get().getDefiningOp();
    if (!producer || !isa<tensor::CollapseShapeOp>(producer)) return failure();

    Value input = convOp.getInputs()[0];
    Value filter = convOp.getInputs()[1];
    Value output = convOp.getOutputs()[0];

    auto inputType = input.getType().cast<ShapedType>();
    auto outputType = convOp.getOutputs()[0].getType().cast<ShapedType>();
    auto inputShape = inputType.getShape();

    auto producerOp = dyn_cast<tensor::CollapseShapeOp>(producer);
    auto expandFilterType = producerOp.getOperand().getType().cast<RankedTensorType>();
    auto expandFilterShape = expandFilterType.getShape();
    auto expandInputShape = {inputShape[0], expandFilterShape[1], expandFilterShape[2],
                             inputShape[2], inputShape[3]};
    RankedTensorType expandInputType =
          RankedTensorType::get(expandInputShape, inputType.getElementType());

    auto loc = convOp.getLoc();

    SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1, 2}, {3}, {4}};

    auto reshapedInput = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandInputType, input, outputReassocIndices);
    auto reshapedFilter = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandFilterType, filter, outputReassocIndices);

    AffineExpr nDim, ocDim, ohDim, owDim, e1Dim, e2Dim, khDim, kwDim;
    bindDims(getContext(), nDim, ocDim, ohDim, owDim, e1Dim, e2Dim, khDim, kwDim);
    auto lhsMap = AffineMap::get(8, 0, {nDim, e1Dim, e2Dim, ohDim + khDim, owDim + kwDim}, getContext());
    auto rhsMap = AffineMap::get(8, 0, {ocDim, e1Dim, e2Dim, khDim, kwDim}, getContext());
    auto resultMap = AffineMap::get(8, 0, {nDim, ocDim, ohDim, owDim}, getContext());

    SmallVector<utils::IteratorType> iterators = convOp.getIteratorTypesArray();
    iterators.append({utils::IteratorType::reduction});
    auto expandOutput = rewriter.create<linalg::GenericOp>(
          loc, outputType,
          /*inputs=*/ValueRange{reshapedInput, reshapedFilter},
          /*outputs=*/output,
          ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, iterators);
    IRMapping mapper;
    convOp->getRegion(0).cloneInto(&expandOutput.getRegion(), mapper);

    rewriter.replaceOp(convOp, ArrayRef<Value>{expandOutput.getResult(0)});
    return success();
  }
};

struct ExpandGeneralizedConvPass
    : ExpandGeneralizedConvBase<ExpandGeneralizedConvPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ExpandGenericConv>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createExpandGeneralizedConvPass() {
  return std::make_unique<ExpandGeneralizedConvPass>();
}

}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
