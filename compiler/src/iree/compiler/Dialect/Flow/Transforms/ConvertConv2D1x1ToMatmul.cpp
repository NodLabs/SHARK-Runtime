// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

// Converts linalg.conv_2d_input_nhwc_filter_nhwc op to linalg.matmul
template <typename Conv2DOpType>
class Convert1x1ConvolutionMatmulOp
    : public OpRewritePattern<Conv2DOpType> {
 public:
  using OpRewritePattern<Conv2DOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DOpType convOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType inputShapeType =
        convOp.getInputOperand(0)->get().getType().template dyn_cast<RankedTensorType>();
    RankedTensorType filterShapeType =
        convOp.getInputOperand(1)->get().getType().template dyn_cast<RankedTensorType>();
    RankedTensorType outputShapeType = convOp.getOutputOperand(0)
                                           ->get()
                                           .getType()
                                           .template dyn_cast<RankedTensorType>();

    int kcIndex, kfIndex, khIndex, kwIndex, ohIndex, owIndex, ocIndex;
    SmallVector<ReassociationIndices, 4> reassociationInputOutputIndices;
    SmallVector<ReassociationIndices, 4> reassociationFilterIndices;
    const bool isNCHW = isa<linalg::Conv2DNchwFchwOp>(convOp);
    const bool isNHWC = isa<linalg::Conv2DNhwcHwcfOp>(convOp);
    if (!isNCHW & !isNHWC) return failure();
    
    kcIndex = isNHWC ? 2 : 1;
    kfIndex = isNHWC ? 3 : 0;
    khIndex = isNHWC ? 0 : 2;
    kwIndex = isNHWC ? 1 : 3;
    ohIndex = isNHWC ? 1 : 2;
    owIndex = isNHWC ? 2 : 3;
    ocIndex = isNHWC ? 3 : 1;
    if(isNHWC) {
      reassociationInputOutputIndices = {{0, 1, 2}, {3}};
      reassociationFilterIndices = {{0, 1, 2}, {3}};
    } else if (isNCHW) {
      reassociationInputOutputIndices = {{0, 1}, {2, 3}};
      reassociationFilterIndices = {{0}, {1, 2, 3}};
    }

    if (!inputShapeType || !filterShapeType || !outputShapeType)
      return failure();

    auto inputShape = inputShapeType.getShape();
    auto filterShape = filterShapeType.getShape();
    auto outputShape = outputShapeType.getShape();

    bool inputDynWidthHeight = inputShape[ohIndex] == ShapedType::kDynamicSize &&
                               inputShape[owIndex] == ShapedType::kDynamicSize;

    // We cannot merge the width and height if they are both dynamic as we
    // cannot expand them back to their dynamic values.
    if (inputDynWidthHeight) return failure();

    if (filterShape[khIndex] != 1 || filterShape[kwIndex] != 1) return failure();

    // TODO(ataei): Support conversion to linalg.batch_matmul.
    if (inputShape[0] != 1) return failure();

    if (!llvm::all_of(convOp.getStrides(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();
    if (!llvm::all_of(convOp.getDilations(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();

    auto combineDims = [](int64_t a, int64_t b) {
      if (a == ShapedType::kDynamicSize || b == ShapedType::kDynamicSize)
        return ShapedType::kDynamicSize;
      return a * b;
    };

    SmallVector<int64_t> reshapedInputShape(2, 0);
    SmallVector<int64_t> reshapedFilterShape(2, 0);
    SmallVector<int64_t> reshapedOutputShape(2, 0);
    if (isNHWC) {
      reshapedInputShape = {combineDims(inputShape[ohIndex], inputShape[owIndex]), inputShape[ocIndex]};
      reshapedFilterShape = {filterShape[kcIndex], filterShape[kfIndex]};
      reshapedOutputShape = {combineDims(outputShape[ohIndex], outputShape[owIndex]), outputShape[ocIndex]};
    }
    else if (isNCHW) {
      reshapedInputShape = {inputShape[ocIndex], combineDims(inputShape[ohIndex], inputShape[owIndex])};
      reshapedFilterShape = {filterShape[kfIndex], filterShape[kcIndex]};
      reshapedOutputShape = {outputShape[ocIndex], combineDims(outputShape[ohIndex], outputShape[owIndex])};
    }

    auto reshapedInputType = RankedTensorType::get(
        reshapedInputShape,
        inputShapeType.getElementType());

    auto reshapedFilterType = RankedTensorType::get(
        reshapedFilterShape, filterShapeType.getElementType());

    auto reshapedOutputType = RankedTensorType::get(
        reshapedOutputShape,
        outputShapeType.getElementType());

    Value input = convOp.getInputOperand(0)->get();
    Value filter = convOp.getInputOperand(1)->get();
    Value output = convOp.getOutputOperand(0)->get();
    auto loc = convOp.getLoc();

    Value reshapedInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedInputType, input, reassociationInputOutputIndices);
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, reassociationFilterIndices);
    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, reassociationInputOutputIndices);

    SmallVector<Value, 2> matmulInput;
    if (isNHWC) {
      matmulInput = {reshapedInput, reshapedFilter};
    }
    else if (isNCHW) {
      matmulInput = {reshapedFilter, reshapedInput};
    }
    auto matmulResult = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputType, matmulInput,
        ArrayRef<Value>{reshapedOutput});

    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputShapeType, matmulResult.getResults()[0],
        reassociationInputOutputIndices);

    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

struct ConvertConv2D1x1ConvToMatmulPass
    : public ConvertConv2D1x1ConvToMatmulBase<
          ConvertConv2D1x1ConvToMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<Convert1x1ConvolutionMatmulOp<linalg::Conv2DNhwcHwcfOp>, 
                    Convert1x1ConvolutionMatmulOp<linalg::Conv2DNchwFchwOp>>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createConvertConv2D1x1ToMatmulPass() {
  return std::make_unique<ConvertConv2D1x1ConvToMatmulPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
