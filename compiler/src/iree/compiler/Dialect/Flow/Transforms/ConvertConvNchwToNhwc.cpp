// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-flow-convert-conv-nchw-to-nhwc"

#define TRANSPOSE_ATTR_NAME "_ConvNchwToNhwcTranspose"
#define GENERIC_ATTR_NAME "_NormalGeneric"
#define CLAST "CLast"
#define CFIRST "CFirst"
#define FLAST "FLast"
#define FFIRST "FFirst"
#define TRANSPOSE_INIT "TransposeInit"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

// Helper function for propagating transpose tags to ops that can handle them.
// This assumes propagation happens moving up the use-def chain because we want
// to make sure to hit the highest level ops possible, then when applying the
// propagations we can reach most ops not caught by this (mainly inputs to
// linalg.generics).
static LogicalResult propagateTagThroughOp(Operation *op) {
  // We don't want to overwrite existing tags
  if (op->hasAttr(TRANSPOSE_ATTR_NAME)) return success();

  if (op->getNumResults() != 1) return success();

  auto result = op->getResults()[0];
  if (result.use_empty()) return success();

  RankedTensorType outputType = dyn_cast<RankedTensorType>(result.getType());
  if (!outputType || outputType.getRank() != 4) return success();

  MLIRContext *context = op->getContext();
  Attribute tag;

  if (llvm::any_of(op->getResults()[0].getUses(), [&tag](const OpOperand &use) {
        auto owner = use.getOwner();
        bool propagate = owner->hasAttr(TRANSPOSE_ATTR_NAME);
        if (propagate) tag = owner->getAttr(TRANSPOSE_ATTR_NAME);
        return propagate;
      })) {
    if (llvm::all_of(op->getResults()[0].getUses(),
                     [&tag](const OpOperand &use) {
                       auto owner = use.getOwner();
                       return (owner->getAttr(TRANSPOSE_ATTR_NAME) == tag) ||
                              dyn_cast<linalg::GenericOp>(owner);
                     })) {
      op->setAttr(TRANSPOSE_ATTR_NAME, tag);

      // We have to mark naturally occuring linalg.generics separately from ones
      // we generate when transposing
      if (dyn_cast<linalg::GenericOp>(op)) {
        op->setAttr(GENERIC_ATTR_NAME, UnitAttr::get(context));
      }
    }
  }
  return success();
}

static SmallVector<uint64_t> getShuffleIndicesFromTag(MLIRContext *context,
                                                      Attribute tag) {
  SmallVector<uint64_t> targetIndices;
  if (tag == StringAttr::get(context, CLAST)) {
    targetIndices.append({0, 2, 3, 1});
  } else if (tag == StringAttr::get(context, CFIRST)) {
    targetIndices.append({0, 3, 1, 2});
  } else if (tag == StringAttr::get(context, FLAST)) {
    targetIndices.append({2, 3, 1, 0});
  } else {
    targetIndices.append({3, 2, 0, 1});
  }
  return targetIndices;
}

static bool isPropagatingTag(MLIRContext *context, Attribute tag) {
  if (tag == StringAttr::get(context, CLAST) ||
      tag == StringAttr::get(context, FLAST) ||
      tag == StringAttr::get(context, CFIRST) ||
      tag == StringAttr::get(context, FFIRST)) {
    return true;
  }
  return false;
}

static Attribute invertTag(MLIRContext *context, Attribute tag) {
  if (tag == StringAttr::get(context, CLAST)) {
    return StringAttr::get(context, CFIRST);
  } else if (tag == StringAttr::get(context, FLAST)) {
    return StringAttr::get(context, FFIRST);
  } else if (tag == StringAttr::get(context, CFIRST)) {
    return StringAttr::get(context, CLAST);
  } else if (tag == StringAttr::get(context, FFIRST)) {
    return StringAttr::get(context, FLAST);
  }
  return tag;
}

// Helper to shuffle vectors according to the tag type
template <typename T>
static SmallVector<T> shuffle4DFromTag(MLIRContext *context,
                                       SmallVector<T> unshuffled, Attribute tag,
                                       bool invert) {
  if (invert) tag = invertTag(context, tag);

  SmallVector<uint64_t> targetIndices = getShuffleIndicesFromTag(context, tag);
  SmallVector<T> shuffled(
      {unshuffled[targetIndices[0]], unshuffled[targetIndices[1]],
       unshuffled[targetIndices[2]], unshuffled[targetIndices[3]]});
  return shuffled;
}

static Attribute oneWayTagDown(MLIRContext *context, Attribute tag) {
  if (tag == StringAttr::get(context, CFIRST)) {
    tag = StringAttr::get(context, CLAST);
  } else if (tag == StringAttr::get(context, FFIRST)) {
    tag = StringAttr::get(context, FLAST);
  }
  return tag;
}

// Transpose the input tensor based on the given tag. The tensor being
// transposed by this helper should always be rank 4.
static Value create4DTransposeWithAttr(PatternRewriter &rewriter, Location loc,
                                       Value input,
                                       SmallVector<uint64_t> targetIndices,
                                       Attribute tag) {
  RankedTensorType inType = input.getType().cast<RankedTensorType>();
  auto inputRank = inType.getRank();
  assert(inputRank == 4);
  auto elementType = inType.getElementType();
  SmallVector<int64_t> inputShape(inType.getShape());

  MLIRContext *context = rewriter.getContext();

  SmallVector<AffineExpr> idExprs;
  for (auto i = 0; i < inputRank; i++)
    idExprs.push_back(getAffineDimExpr(i, context));

  SmallVector<int64_t> outputShape =
      shuffle4DFromTag<int64_t>(context, inputShape, tag, false);
  SmallVector<AffineExpr> swapExprs =
      shuffle4DFromTag<AffineExpr>(context, idExprs, tag, false);

  Value output =
      rewriter.create<tensor::EmptyOp>(loc, outputShape, elementType);

  output.getDefiningOp()->setAttr(TRANSPOSE_ATTR_NAME,
                                  StringAttr::get(context, TRANSPOSE_INIT));

  SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(inputRank, 0, idExprs, context),
      AffineMap::get(inputRank, 0, swapExprs, context)};
  SmallVector<StringRef> iteratorTypes(inputRank, "parallel");
  auto transpose = rewriter.create<linalg::GenericOp>(
      loc, output.getType(), input, output, indexingMaps, iteratorTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args[0]);
      });
  transpose->setAttr(TRANSPOSE_ATTR_NAME, tag);
  return transpose.getResult(0);
}

// if inputIsNchw {0, 1, 2, 3} -> {0, 3, 1, 2}
// else           {0, 1, 2, 3} -> {0, 2, 3, 1}
static Value createNchwTransposeWithAttr(PatternRewriter &rewriter,
                                         Location loc, Value input,
                                         bool inputIsNchw) {
  StringAttr tag;
  MLIRContext *context = rewriter.getContext();
  if (inputIsNchw) {
    tag = StringAttr::get(context, CLAST);
  } else {
    tag = StringAttr::get(context, CFIRST);
  }
  SmallVector<uint64_t> targetIndices = getShuffleIndicesFromTag(context, tag);
  return create4DTransposeWithAttr(rewriter, loc, input, targetIndices, tag);
}

// if inputIsFchw {0, 1, 2, 3} -> {2, 3, 1, 0}
// else           {0, 1, 2, 3} -> {3, 2, 0, 1}
static Value createFchwTransposeWithAttr(PatternRewriter &rewriter,
                                         Location loc, Value input,
                                         bool inputIsFchw) {
  StringAttr tag;
  MLIRContext *context = rewriter.getContext();
  if (inputIsFchw) {
    tag = StringAttr::get(rewriter.getContext(), FLAST);
  } else {
    tag = StringAttr::get(rewriter.getContext(), FFIRST);
  }
  SmallVector<uint64_t> targetIndices = getShuffleIndicesFromTag(context, tag);
  return create4DTransposeWithAttr(rewriter, loc, input, targetIndices, tag);
}

static Value createTransposeWithAttrFromTag(PatternRewriter &rewriter,
                                            Location loc, Value input,
                                            Attribute tag, bool inputIsFirst) {
  MLIRContext *context = rewriter.getContext();
  if (!inputIsFirst) {
    if (tag == StringAttr::get(context, CLAST)) {
      tag = StringAttr::get(context, CFIRST);
    } else if (tag == StringAttr::get(context, FLAST)) {
      tag = StringAttr::get(context, FFIRST);
    }
  } else {
    if (tag == StringAttr::get(context, CFIRST)) {
      tag = StringAttr::get(context, CLAST);
    } else if (tag == StringAttr::get(context, FFIRST)) {
      tag = StringAttr::get(context, FLAST);
    }
  }
  SmallVector<uint64_t> targetIndices = getShuffleIndicesFromTag(context, tag);
  return create4DTransposeWithAttr(rewriter, loc, input, targetIndices, tag);
}

// Supports conv and pooling ops, where pooling ops don't transpose the filter
template <typename ConvOpTy, typename ConvTargetOpTy>
static LogicalResult convertConvLikeNchwToNhwc(PatternRewriter &rewriter,
                                               ConvOpTy convOp,
                                               bool transposeFilter) {
  LLVM_DEBUG(llvm::dbgs() << "inspecting " << convOp << "\n");

  Location loc = convOp.getLoc();

  Value input = convOp.image();
  Value filter = convOp.filter();
  Value output = convOp.getOutputs()[0];

  auto inputType = input.getType().cast<RankedTensorType>();
  auto filterType = filter.getType().cast<RankedTensorType>();
  auto outputType = output.getType().cast<RankedTensorType>();

  // Require rank 4
  if (inputType.getRank() != 4 ||
      (transposeFilter && filterType.getRank() != 4) ||
      outputType.getRank() != 4) {
    return failure();
  }

  auto transposedInput =
      createNchwTransposeWithAttr(rewriter, loc, input, true);
  auto transposedFilter = filter;
  if (transposeFilter)
    transposedFilter = createFchwTransposeWithAttr(rewriter, loc, filter, true);
  auto transposedOutput =
      createNchwTransposeWithAttr(rewriter, loc, output, true);

  auto conv =
      rewriter
          .create<ConvTargetOpTy>(loc, transposedOutput.getType(),
                                  ValueRange{transposedInput, transposedFilter},
                                  transposedOutput, convOp.getStrides(),
                                  convOp.getDilations())
          .getResult(0);

  auto returnToNCHW = createNchwTransposeWithAttr(rewriter, loc, conv, false);

  rewriter.replaceOp(convOp, returnToNCHW);
  return success();
}

// Conversion Patterns ------------------------------------

namespace {

/*
 *  Convolution conversion patterns
 */

struct ConvertLinalgConvNchwFchw : OpRewritePattern<linalg::Conv2DNchwFchwOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    return convertConvLikeNchwToNhwc<linalg::Conv2DNchwFchwOp,
                                     linalg::Conv2DNhwcHwcfOp>(rewriter, convOp,
                                                               true);
  }
};

struct ConvertLinalgPoolingNchwMax
    : OpRewritePattern<linalg::PoolingNchwMaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PoolingNchwMaxOp poolOp,
                                PatternRewriter &rewriter) const override {
    return convertConvLikeNchwToNhwc<linalg::PoolingNchwMaxOp,
                                     linalg::PoolingNhwcMaxOp>(rewriter, poolOp,
                                                               false);
  }
};

struct ConvertLinalgPoolingNchwSum
    : OpRewritePattern<linalg::PoolingNchwSumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PoolingNchwSumOp poolOp,
                                PatternRewriter &rewriter) const override {
    return convertConvLikeNchwToNhwc<linalg::PoolingNchwSumOp,
                                     linalg::PoolingNhwcSumOp>(rewriter, poolOp,
                                                               false);
  }
};

/*
 *  Transpose propagation patterns
 */

struct PropagateThroughTensorPad : OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    if (!padOp->hasAttr(TRANSPOSE_ATTR_NAME)) return failure();
    LLVM_DEBUG(llvm::dbgs() << "propagating " << padOp << "\n");
    Attribute tag = padOp->getAttr(TRANSPOSE_ATTR_NAME);

    Location loc = padOp.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto input = padOp.getSource();
    SmallVector<OpFoldResult> mixedLow = shuffle4DFromTag<OpFoldResult>(
        context, padOp.getMixedLowPad(), tag, false);
    SmallVector<OpFoldResult> mixedHigh = shuffle4DFromTag<OpFoldResult>(
        context, padOp.getMixedHighPad(), tag, false);

    auto transposedInput =
        createTransposeWithAttrFromTag(rewriter, loc, input, tag, true);

    SmallVector<int64_t> outputShape(padOp.getResultType().getShape());
    SmallVector<int64_t> transposedOutputShape =
        shuffle4DFromTag<int64_t>(context, outputShape, tag, false);
    RankedTensorType transposedOutputType = RankedTensorType::get(
        transposedOutputShape, padOp.getResultType().getElementType());

    auto newPad = rewriter.create<tensor::PadOp>(loc, transposedOutputType,
                                                 transposedInput, mixedLow,
                                                 mixedHigh, padOp.getNofold());
    BlockAndValueMapping mapper;
    padOp.getRegion().cloneInto(&newPad.getRegion(), mapper);
    newPad->removeAttr(TRANSPOSE_ATTR_NAME);

    auto returnToNCHW = createTransposeWithAttrFromTag(
        rewriter, loc, newPad.getResult(), tag, false);

    rewriter.replaceOp(padOp, returnToNCHW);
    return success();
  }
};

struct PropagateThroughLinalgFill : OpRewritePattern<linalg::FillOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {
    if (!fillOp->hasAttr(TRANSPOSE_ATTR_NAME)) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "propagating " << fillOp << "\n");
    Attribute tag = fillOp->getAttr(TRANSPOSE_ATTR_NAME);

    Location loc = fillOp.getLoc();

    auto transposedOutput = createTransposeWithAttrFromTag(
        rewriter, loc, fillOp.output(), tag, true);

    auto newTensor =
        rewriter.create<linalg::FillOp>(loc, fillOp.value(), transposedOutput)
            .getResult(0);

    auto returnToNCHW =
        createTransposeWithAttrFromTag(rewriter, loc, newTensor, tag, false);

    rewriter.replaceOp(fillOp, returnToNCHW);
    return success();
  }
};

struct PropagateThroughLinalgGeneric : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    bool propagateThrough = genericOp->hasAttr(TRANSPOSE_ATTR_NAME);
    if (!genericOp->hasAttr(GENERIC_ATTR_NAME) && propagateThrough) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "propagating " << genericOp << "\n");
    Attribute tag = genericOp->getAttr(TRANSPOSE_ATTR_NAME);

    Location loc = genericOp.getLoc();
    MLIRContext *context = rewriter.getContext();

    // For now we are restricting to single outputs.
    auto transposedOutput = genericOp.getOutputs()[0];
    auto indexingMaps = genericOp.getIndexingMapsArray();

    if (transposedOutput.getDefiningOp()->hasAttr(TRANSPOSE_ATTR_NAME)) {
      if (!propagateThrough) {
        auto tmpTag =
            transposedOutput.getDefiningOp()->getAttr(TRANSPOSE_ATTR_NAME);
        if (isPropagatingTag(context, tmpTag)) {
          tag = tmpTag;
          propagateThrough = true;
        }
      }
    }

    if (propagateThrough) {
      transposedOutput = createTransposeWithAttrFromTag(
          rewriter, loc, transposedOutput, tag, true);

      AffineMap outMap = indexingMaps.back();
      SmallVector<AffineExpr> outExprs(outMap.getResults());
      SmallVector<AffineExpr> exprs =
          shuffle4DFromTag<AffineExpr>(context, outExprs, tag, false);
      indexingMaps[indexingMaps.size() - 1] =
          AffineMap::get(outMap.getNumDims(), outMap.getNumSymbols(), exprs,
                         genericOp->getContext());
    }

    SmallVector<Value> newInputs;
    bool needsUpdate = false;
    for (auto input : llvm::enumerate(genericOp.getInputs())) {
      auto parentOp = input.value().getDefiningOp();
      if (parentOp->hasAttr(TRANSPOSE_ATTR_NAME)) {
        Attribute inputTag =
            oneWayTagDown(context, parentOp->getAttr(TRANSPOSE_ATTR_NAME));
        auto transposedInput = createTransposeWithAttrFromTag(
            rewriter, loc, input.value(), inputTag, true);
        AffineMap inMap = indexingMaps[input.index()];
        SmallVector<AffineExpr> inputExprs(inMap.getResults());
        SmallVector<AffineExpr> shuffledInputExprs =
            shuffle4DFromTag<AffineExpr>(context, inputExprs, inputTag, false);
        indexingMaps[input.index()] =
            AffineMap::get(inMap.getNumDims(), inMap.getNumSymbols(),
                           shuffledInputExprs, context);
        newInputs.push_back(transposedInput);
        needsUpdate = true;
      } else {
        newInputs.push_back(input.value());
      }
    }

    if (!(needsUpdate || propagateThrough)) return failure();

    SmallVector<StringRef> iteratorTypes = llvm::to_vector(llvm::map_range(
        genericOp.getIteratorTypes(),
        [](Attribute attr) { return attr.cast<StringAttr>().getValue(); }));

    auto newGeneric = rewriter.create<linalg::GenericOp>(
        loc, transposedOutput.getType().cast<RankedTensorType>(), newInputs,
        transposedOutput, indexingMaps, iteratorTypes);
    BlockAndValueMapping mapper;
    genericOp.getRegion().cloneInto(&newGeneric.getRegion(), mapper);
    newGeneric->removeAttr(GENERIC_ATTR_NAME);
    newGeneric->setAttr(TRANSPOSE_ATTR_NAME,
                        StringAttr::get(context, "Transposed"));

    Value returnToNCHW = newGeneric.getResult(0);
    if (propagateThrough) {
      returnToNCHW = createTransposeWithAttrFromTag(rewriter, loc, returnToNCHW,
                                                    tag, false);
    }

    rewriter.replaceOp(genericOp, returnToNCHW);
    return success();
  }
};

struct PropagateThroughLinalgTensorEmpty
    : OpRewritePattern<tensor::EmptyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::EmptyOp tensorEmptyOp,
                                PatternRewriter &rewriter) const override {
    if (!tensorEmptyOp->hasAttr(TRANSPOSE_ATTR_NAME) ||
        tensorEmptyOp->getAttr(TRANSPOSE_ATTR_NAME) ==
            StringAttr::get(tensorEmptyOp.getContext(), TRANSPOSE_INIT)) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "propagating " << tensorEmptyOp << "\n");
    Attribute tag = tensorEmptyOp->getAttr(TRANSPOSE_ATTR_NAME);

    Location loc = tensorEmptyOp.getLoc();
    MLIRContext *context = rewriter.getContext();

    SmallVector<OpFoldResult> mixedSizes = shuffle4DFromTag<OpFoldResult>(
        context, tensorEmptyOp.getMixedSizes(), tag, false);

    auto newTensor = rewriter.create<tensor::EmptyOp>(
        loc, mixedSizes, tensorEmptyOp.getType().getElementType());
    auto returnToNCHW = createTransposeWithAttrFromTag(
        rewriter, loc, newTensor.getResult(), tag, false);

    rewriter.replaceOp(tensorEmptyOp, returnToNCHW);
    return success();
  }
};

// Currently doesn't do the static transposing of weights so this pattern is
// disabled
struct PropagateThroughArithConstant : OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                PatternRewriter &rewriter) const override {
    if (!constantOp->hasAttr(TRANSPOSE_ATTR_NAME)) {
      return failure();
    }
    LLVM_DEBUG(llvm::dbgs() << "propagating " << constantOp << "\n");
    Attribute tag = constantOp->getAttr(TRANSPOSE_ATTR_NAME);

    Location loc = constantOp.getLoc();
    MLIRContext *context = rewriter.getContext();

    RankedTensorType outputType =
        dyn_cast<RankedTensorType>(constantOp.getType());

    SmallVector<int64_t> outputShape(outputType.getShape());
    SmallVector<int64_t> transposedOutputShape =
        shuffle4DFromTag<int64_t>(context, outputShape, tag, false);
    RankedTensorType transposedOutputType = RankedTensorType::get(
        transposedOutputShape, outputType.getElementType());

    DenseElementsAttr elements;
    if (!(elements = constantOp.getValue().dyn_cast<DenseElementsAttr>())) {
      return failure();
    }
    DenseElementsAttr newElements = elements.reshape(transposedOutputType);

    auto newTensor = rewriter.create<arith::ConstantOp>(
        loc, transposedOutputType, newElements);
    auto returnToNCHW = createTransposeWithAttrFromTag(
        rewriter, loc, newTensor.getResult(), tag, false);

    rewriter.replaceOp(constantOp, returnToNCHW);
    return success();
  }
};

/*
 *  Folding away cancelling transposes
 */

// Cancel if this transpose is tagged with a propagating tag and the defining op
// for the input is the inverse of this transpose
struct CancelNCHWToNHWCTranspose : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp transposeOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "trying to fold " << transposeOp << "\n");

    if (!transposeOp->hasAttr(TRANSPOSE_ATTR_NAME)) return failure();

    MLIRContext *context = transposeOp->getContext();

    Attribute transposeType = transposeOp->getAttr(TRANSPOSE_ATTR_NAME);

    if (!isPropagatingTag(context, transposeType)) {
      return failure();
    }

    auto parentOp =
        transposeOp->getOperand(0).getDefiningOp<linalg::GenericOp>();
    if (parentOp) {
      Attribute tag =
          invertTag(context, parentOp->getAttr(TRANSPOSE_ATTR_NAME));
      if (transposeType == tag) {
        rewriter.replaceOp(transposeOp, parentOp->getOperand(0));
        return success();
      }
    }

    return failure();
  }
};

// The high level strategy for this pass is as follows:
//     1. Do the conversions for all conv_nchw_fchw ops (and pooling ops) and
//     wrap the converted convolutions in transposes. Each transpose is tagged
//     to indicate whether it is transposing into channels last or back to
//     channels first.
//     2. Propagate the tags from transposes that do nchw -> nhwc up their
//     use-def chains through ops where there is support for propagating
//     transposes.
//     3. Rewrite all of the ops tagged with propagating transposes and wrap
//     them in transposes same as with the convolutions.
//     4. Canonicalize out all adjacent cancelling transposes.
struct ConvertConvNchwToNhwcPass
    : public ConvertConvNchwToNhwcBase<ConvertConvNchwToNhwcPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    Operation *funcOp = getOperation();
    MLIRContext *context = &getContext();

    {
      RewritePatternSet patterns(context);
      patterns.insert<ConvertLinalgConvNchwFchw>(context);
      patterns.insert<ConvertLinalgPoolingNchwMax>(context);
      patterns.insert<ConvertLinalgPoolingNchwSum>(context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    {
      auto transposePropagationFn = [&](Operation *op) -> WalkResult {
        return TypeSwitch<Operation *, LogicalResult>(op)
            .Case<tensor::PadOp, linalg::FillOp, tensor::EmptyOp,
                  // linalg::GenericOp, arith::ConstantOp>([&](auto taggableOp)
                  // {
                  linalg::GenericOp>([&](auto taggableOp) {
              return propagateTagThroughOp(taggableOp);
            })
            .Default([&](Operation *op) -> LogicalResult { return success(); });
      };

      for (Region &region : llvm::reverse(funcOp->getRegions())) {
        for (Block &block : llvm::reverse(region.getBlocks())) {
          for (Operation &op : llvm::reverse(block.getOperations())) {
            transposePropagationFn(&op);
          }
        }
      }
    }

    {
      RewritePatternSet patterns(context);
      patterns.insert<PropagateThroughTensorPad>(context);
      patterns.insert<PropagateThroughLinalgTensorEmpty>(context);
      patterns.insert<PropagateThroughLinalgFill>(context);
      patterns.insert<PropagateThroughLinalgGeneric>(context);
      // patterns.insert<PropagateThroughArithConstant>(context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    {
      RewritePatternSet patterns(context);
      patterns.insert<CancelNCHWToNHWCTranspose>(context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvertConvNchwToNhwcPass() {
  return std::make_unique<ConvertConvNchwToNhwcPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
