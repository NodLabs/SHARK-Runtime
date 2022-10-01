//===- ExtractFromInsertSliceDestPatterns.cpp -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

bool areDisjointRanges(ArrayRef<OpFoldResult> aOffsets,
                             ArrayRef<OpFoldResult> aSizes,
                             ArrayRef<OpFoldResult> bOffsets,
                             ArrayRef<OpFoldResult> bSizes) {
  assert(llvm::all_equal(
      {aOffsets.size(), aSizes.size(), bOffsets.size(), bSizes.size()}));

  for (const auto &t : llvm::zip(aOffsets, aSizes, bOffsets, bSizes)) {
    auto [aBeginVal, aSizeVal, bBeginVal, bSizeVal] = t;
    Optional<int64_t> aBegin = getConstantIntValue(aBeginVal);
    Optional<int64_t> aSize = getConstantIntValue(aSizeVal);
    Optional<int64_t> bBegin = getConstantIntValue(bBeginVal);
    Optional<int64_t> bSize = getConstantIntValue(bSizeVal);

    // If there are dynamic offsets/sizes, we cannot prove this dimension is
    // disjoint. Look at other dimensions.
    if (!aBegin || !aSize || !bBegin || !bSize)
      continue;

    int aEnd = *aBegin + *aSize;
    int bEnd = *bBegin + *bSize;
    // As long as one dimension is disjoint, the whole slices are disjoint.
    if (aEnd <= *bBegin || bEnd <= *aBegin)
      return true;
  }
  return false;
}

bool areDisjointSlices(OffsetSizeAndStrideOpInterface aSlice,
                             OffsetSizeAndStrideOpInterface bSlice) {
  SmallVector<OpFoldResult> aOffsets = aSlice.getMixedOffsets();
  SmallVector<OpFoldResult> bOffsets = bSlice.getMixedOffsets();
  SmallVector<OpFoldResult> aSizes = aSlice.getMixedSizes();
  SmallVector<OpFoldResult> bSizes = bSlice.getMixedSizes();
  SmallVector<OpFoldResult> aStrides = aSlice.getMixedStrides();
  SmallVector<OpFoldResult> bStrides = bSlice.getMixedStrides();

  // For simplicity only look at stride 1 cases for now.
  auto hasAllOnes = [](ArrayRef<OpFoldResult> strides) {
    return llvm::all_of(strides, [](::mlir::OpFoldResult ofr) {
      return getConstantIntValue(ofr) == static_cast<int64_t>(1);
    });
  };
  if (!hasAllOnes(aStrides) || !hasAllOnes(bStrides))
    return false;

  return areDisjointRanges(aOffsets, aSizes, bOffsets, bSizes);
}

/// Updates extract_slice to extrace from insert_slice op's destination tensor
/// when the extract_slice and insert_slice are covering disjoint slices.
///
/// Example:
/// ```mlir
/// %i = tensor.insert_slice %src into %dst[0, 0, 0, 0][1, 1, 2, 4][1, 1, 1, 1]
///        : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
/// %e = tensor.extract_slice %i[0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
///        : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
/// ```
/// Can be converted into
/// ```mlir
/// %i = tensor.insert_slice %src into %dst[0, 0, 0, 0][1, 1, 2, 4][1, 1, 1, 1]
///        : tensor<1x2x4xf32> into tensor<1x2x2x4xf32>
/// %e = tensor.extract_slice %dest[0, 1, 0, 0] [1, 1, 2, 4] [1, 1, 1, 1]
///        : tensor<1x2x2x4xf32> to tensor<1x2x4xf32>
/// ```
/// This helps to break the chain of insert_slice and extract_slices, which
/// might enable further optimizations.
struct ExtractFromInsertDest final : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto insertOp = extractOp.getSource().getDefiningOp<tensor::InsertSliceOp>();
    if (!insertOp)
      return failure();

    if (!areDisjointSlices(insertOp, extractOp))
      return rewriter.notifyMatchFailure(extractOp, "not disjoint");

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        extractOp, extractOp.getType(), insertOp.getDest(),
        extractOp.getMixedOffsets(), extractOp.getMixedSizes(),
        extractOp.getMixedStrides());

    return success();
  }
};
} // namespace

void populateExtractFromInsertSliceDestOpPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ExtractFromInsertDest>(patterns.getContext());
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir