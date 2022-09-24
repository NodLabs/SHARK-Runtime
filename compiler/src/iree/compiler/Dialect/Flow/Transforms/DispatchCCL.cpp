// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-flow-dispatch-ccl"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

/// Returns true if the operation has only uses in `tensor.dim` ops.
bool hasComputeUsesOutsideDispatch(
    Operation *op, ArrayRef<Operation *> dispatchOps = {}) {
  return !llvm::all_of(op->getUsers(), [&](Operation *user) {
    return isa<tensor::DimOp>(user) || llvm::is_contained(dispatchOps, user);
  });
}

/// For an operation to be moved into the dispatch region, append `resultTypes`
/// with the type of the results dispatch region has to return. Also
/// append `resultDynamicDims` with values that represent the dynamic shapes of
/// result values returned.
static LogicalResult computeDispatchResultTypeAndDynamicDims(
    PatternRewriter &rewriter, Operation *dispatchOp,
    SmallVector<Type> &resultTypes, SmallVector<Value> &resultDynamicDims) {
  auto currResultTypes = dispatchOp->getResultTypes();
  resultTypes.append(currResultTypes.begin(), currResultTypes.end());


  // Get the values for the result dims.
  for (auto outputType : llvm::enumerate(currResultTypes)) {
    auto shapedOutputType = outputType.value().dyn_cast<ShapedType>();
    if (!shapedOutputType) continue;
    for (auto dim : llvm::enumerate(shapedOutputType.getShape())) {
      if (ShapedType::isDynamic(dim.value())) {
        return rewriter.notifyMatchFailure(dispatchOp,
                                           "dynamic dimensions are not implemented.");
      }
    }
  }
  return success();
}

/// Creates a flow.dispatch.workgroup op without arguments.
/// All the necessary operands are transiently captured and rewritten late as
/// operands. This greatly simplifies transformations into the resulting op.
FailureOr<SmallVector<Operation *>>
buildOperandLessFlowDispatchCollectivesOp(PatternRewriter &rewriter, Location loc,
                                        ArrayRef<Operation *> dispatchOps) {
  SmallVector<Value> resultDynamicDims;
  SmallVector<Type> resultTypes;

  // 1. Compute the result types for the dispatch and the dynamic dimensions
  //    of the result of the dispatch. If operation has only dim uses
  //    do not make the dispatch op return those values. Those uses are
  //    kept on the original op, and later patterns are expected to take care
  //    of them.
  for (auto op : dispatchOps) {
    if (!hasComputeUsesOutsideDispatch(op, dispatchOps)) continue;
    if (failed(computeDispatchResultTypeAndDynamicDims(
            rewriter, op, resultTypes, resultDynamicDims))) {
      return failure();
    }
  }

  // 2. Create a dispatch op with just the `flow.return` terminator.
  auto dispatchOp = rewriter.create<IREE::Flow::DispatchCollectivesOp>(
      loc, resultTypes, resultDynamicDims,
      /*arguments=*/ArrayRef<Value>{},
      /*argument_dims=*/ArrayRef<Value>{},
      /*tiedOperands=*/ArrayRef<int64_t>{});
  Region &region = dispatchOp.getCollectivesBody();
  Block *block = &region.front();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointToEnd(block);
  auto returnOp = rewriter.create<IREE::Flow::ReturnOp>(loc);
  rewriter.setInsertionPoint(returnOp);

  // 3. Clone the necessary operations into the dispatch and replace
  //    all uses of the original op with the cloned op within the dispatch.
  auto resultArgs = region.getArguments();
  unsigned resultPos = 0;
  unsigned resultDynamicDimsPos = 0;
  SmallVector<Value> dispatchOpResults = dispatchOp.getResults();
  SmallVector<Operation *> clonedOps;
  clonedOps.reserve(dispatchOps.size());
  for (auto op : dispatchOps) {
    Operation *clonedOp = rewriter.clone(*op);
    clonedOps.push_back(clonedOp);
    rewriter.replaceOpWithinBlock(op, clonedOp->getResults(), block);
    rewriter.setInsertionPoint(clonedOp);
    if (!hasComputeUsesOutsideDispatch(op, dispatchOps)) continue;

    // 3a. Replace all non-dim uses of the original operation with the
    //     corresponding result of the dispatch.
    rewriter.replaceOpWithIf(op,
                             ArrayRef<Value>(dispatchOpResults)
                                 .slice(resultPos, op->getNumResults()),
                             [&](OpOperand &operand) {
                               return !isa<tensor::DimOp>(operand.getOwner());
                             });

    // 3b. For each of the result create a `flow.dispatch.tensor.store`
    //     operation to publish the result of the cloned operation (from within
    //     the dispatch).
    for (auto clonedOpResult : clonedOp->getResults()) {
      auto resultType = clonedOpResult.getType().dyn_cast<ShapedType>();
      if (resultType) {
        OpBuilder::InsertionGuard g2(rewriter);
        rewriter.setInsertionPoint(returnOp);
        unsigned numDynamicDims = resultType.getNumDynamicDims();
        rewriter.create<IREE::Flow::DispatchTensorStoreOp>(
            loc, clonedOpResult, resultArgs[resultPos],
            ArrayRef<Value>(resultDynamicDims)
                .slice(resultDynamicDimsPos, numDynamicDims));
        resultDynamicDimsPos += numDynamicDims;
      }
      resultPos++;
    }
  }

  return clonedOps;
}

template <typename OpType, template <typename> class Base>
struct CreateCollectivesDispatchRegionOp : Base<OpType> {
  using Base<OpType>::Base;

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (!hasComputeUsesOutsideDispatch(op)) return failure();
    if (op->template getParentOfType<IREE::Flow::DispatchCollectivesOp>()) {
      return failure();
    }

    SmallVector<Operation *> dispatchOps = { op };
    auto clonedOps = buildOperandLessFlowDispatchCollectivesOp(
        rewriter, op.getLoc(), dispatchOps);
    if (failed(clonedOps)) {
      return failure();
    }

    return success();
  }
};

// After outlining in dispatch region we can rewrite the dispatch ops with
// proper captures.
LogicalResult legalizeDispatchCollectivesOperands(
    IREE::Flow::DispatchCollectivesOp dispatchOp) {
  Location loc = dispatchOp.getLoc();
  Region &region = dispatchOp.getCollectivesBody();
  Block &block = region.front();
  OpBuilder b = OpBuilder::atBlockBegin(&block);

  llvm::SetVector<Value> valuesDefinedAbove;
  mlir::getUsedValuesDefinedAbove(region, valuesDefinedAbove);
  if (valuesDefinedAbove.empty()) return success();

  b.setInsertionPointToStart(&block);

  // Build a map from current operands to arguments.
  std::pair<unsigned, unsigned> operandsIndexAndLength =
      dispatchOp.getODSOperandIndexAndLength(1);
  std::pair<unsigned, unsigned> operandDimsIndexAndLength =
      dispatchOp.getODSOperandIndexAndLength(2);
  llvm::DenseMap<Value, BlockArgument> operandToBBArg;
  for (auto operand : llvm::enumerate(dispatchOp.getArguments())) {
    operandToBBArg[operand.value()] = block.getArgument(operand.index());
  }

  // Of the values defined above and used in the region, add values that are not
  // operands to the region already.
  unsigned numOperands = operandsIndexAndLength.second;
  unsigned numOperandDims = operandDimsIndexAndLength.second;
  for (auto value : valuesDefinedAbove) {
    BlockArgument bbArg = operandToBBArg.lookup(value);
    bool wasPresent = bbArg != nullptr;
    auto tensorType = value.getType().dyn_cast<RankedTensorType>();
    if (!bbArg) {
      // Create a new basic block argument for this value.
      Type bbArgType = value.getType();
      if (tensorType) {
        bbArgType = IREE::Flow::DispatchTensorType::get(
            TensorAccess::ReadOnly, tensorType.getShape(),
            tensorType.getElementType());
      }
      bbArg = block.insertArgument(numOperands, bbArgType, value.getLoc());
    }

    // Insert the operand if this is not already one.
    if (!wasPresent) {
      unsigned insertIdx = operandsIndexAndLength.first + numOperands;
      dispatchOp->insertOperands(insertIdx, {value});
      operandToBBArg[dispatchOp->getOperand(insertIdx)] = bbArg;
      numOperands++;
    }

    Value repl = bbArg;
    if (!wasPresent && bbArg.getType().isa<IREE::Flow::DispatchTensorType>()) {
      // This dims for this operand does not exist. Add those.
      SmallVector<Value> dynamicDimArgs;
      {
        OpBuilder::InsertionGuard g(b);
        b.setInsertionPoint(dispatchOp);

        // Fast-path for if the value comes from ops that support our dynamic
        // shape interfaces. Otherwise we have to insert tensor.dim ops.
        auto availableDims = IREE::Util::findDynamicDims(value);

        // Add operands/args for each dynamic shape dimension.
        SmallVector<Value> dynamicDimOperands;
        unsigned dynamicDimIdx = 0;
        for (auto dim : llvm::enumerate(tensorType.getShape())) {
          if (dim.value() != ShapedType::kDynamicSize) continue;
          if (availableDims.has_value()) {
            dynamicDimOperands.push_back(availableDims.value()[dynamicDimIdx]);
          } else {
            dynamicDimOperands.push_back(b.createOrFold<tensor::DimOp>(
                dispatchOp.getLoc(), value, dim.index()));
          }
          dynamicDimArgs.push_back(
              block.insertArgument(numOperands + dynamicDimIdx,
                                   b.getIndexType(), dispatchOp.getLoc()));
          ++dynamicDimIdx;
        }
        dispatchOp->insertOperands(
            operandsIndexAndLength.first + numOperands + numOperandDims,
            dynamicDimOperands);
        numOperandDims += dynamicDimOperands.size();
        dispatchOp->insertOperands(operandsIndexAndLength.first + numOperands,
                                   dynamicDimOperands);
        numOperands += dynamicDimOperands.size();
      }

      // For arguments of type flow.dispatch.tensor, create a
      // flow.dispatch.tensor.load to get the replacement values.
      repl = b.create<IREE::Flow::DispatchTensorLoadOp>(
          loc, value.getType().cast<RankedTensorType>(), bbArg, dynamicDimArgs);
    }

    value.replaceUsesWithIf(repl, [&](OpOperand &use) {
      return use.getOwner()
                 ->getParentOfType<IREE::Flow::DispatchWorkgroupsOp>() ==
             dispatchOp;
    });
  }

  // Update the `operand_segment_sizes`.
  auto operandSegmentSizes = dispatchOp->getAttrOfType<DenseI32ArrayAttr>(
      dispatchOp.getOperandSegmentSizesAttrName());
  auto newValues = llvm::to_vector<4>(operandSegmentSizes.asArrayRef());
  newValues[1] = numOperands;
  newValues[2] = numOperandDims;
  dispatchOp->setAttr(dispatchOp.getOperandSegmentSizesAttrName(),
                      b.getDenseI32ArrayAttr(newValues));
  return success();
}

/// Returns the tied operand for the given `resultArg`. Returns nullptr if error
/// or not found.
BlockArgument getTiedOperandBlockArgument(BlockArgument resultArg) {
  auto resultArgType =
      resultArg.getType().dyn_cast<IREE::Flow::DispatchTensorType>();
  if (!resultArgType ||
      resultArgType.getAccess() != IREE::Flow::TensorAccess::WriteOnly) {
    return nullptr;
  }
  // Each output block argument should just have one use.
  if (!resultArg.hasOneUse()) return nullptr;

  // And that's a flow.dispatch.output.store op.
  auto storeOp = dyn_cast<IREE::Flow::DispatchTensorStoreOp>(
      (*resultArg.getUses().begin()).getOwner());
  if (!storeOp) return nullptr;

  // Check if that block argument is tied to another block argument.
  auto tieOp = storeOp.getValue().getDefiningOp<Util::TiedOpInterface>();
  if (!tieOp) return nullptr;
  auto tiedArg =
      tieOp.getTiedResult(storeOp.getValue().cast<OpResult>().getResultNumber())
          .dyn_cast_or_null<BlockArgument>();
  if (!tiedArg) return nullptr;
  assert(isa<IREE::Flow::DispatchCollectivesOp>(
             tiedArg.getOwner()->getParentOp()) &&
         "expected that BbArg belongs to DispatchCollectivesOp");

  // Check that the type of the tied argument candidate and type of the output
  // match and that the tied argument is readonly.
  auto type = tiedArg.getType().dyn_cast<IREE::Flow::DispatchTensorType>();
  if (!type || type.getAccess() != IREE::Flow::TensorAccess::ReadOnly ||
      type.getElementType() != resultArgType.getElementType() ||
      llvm::any_of(llvm::zip(type.getShape(), resultArgType.getShape()),
                   [](std::tuple<int64_t, int64_t> sizes) {
                     return std::get<0>(sizes) !=
                                IREE::Flow::DispatchTensorType::kDynamicSize &&
                            std::get<1>(sizes) !=
                                IREE::Flow::DispatchTensorType::kDynamicSize &&
                            std::get<0>(sizes) != std::get<1>(sizes);
                   })) {
    return nullptr;
  }
  return tiedArg;
}

/// Modifies `dispatchOp` to attach operand-result tie information when
/// possible.
static void tryToTieOperandsAndResults(
    IREE::Flow::DispatchCollectivesOp dispatchOp) {
  Block *block = dispatchOp.getBody(0);

  // Go over each result to tie operand when possible, by:
  // 1. Update the tied operand argument to take readwrite tensors.
  // 2. Erase the result argument.
  // 3. Attach the tie information to the DispatchWorkgroupsOp.
  for (auto result : llvm::enumerate(dispatchOp.getResults())) {
    if (dispatchOp.getTiedResultOperand(result.value())) continue;
    BlockArgument outputArgument =
        dispatchOp.getOutputBlockArgument(result.index());
    BlockArgument tiedOperandArgument =
        getTiedOperandBlockArgument(outputArgument);
    if (!tiedOperandArgument) continue;
    auto oldType =
        tiedOperandArgument.getType().cast<IREE::Flow::DispatchTensorType>();
    tiedOperandArgument.setType(IREE::Flow::DispatchTensorType::get(
        IREE::Flow::TensorAccess::ReadWrite, oldType.getShape(),
        oldType.getElementType()));
    outputArgument.replaceAllUsesWith(tiedOperandArgument);
    block->eraseArgument(outputArgument.getArgNumber());
    dispatchOp.setTiedResultOperandIndex(result.index(),
                                         tiedOperandArgument.getArgNumber());
  }
}

/// For all ops within `funcOp` tagged as root ops, create dispatch regions.
LogicalResult createDispatchRegionsFromRootOps(mlir::Operation *funcOp,
                                               RewritePatternSet &&patterns) {
  MLIRContext *context = funcOp->getContext();

  // Create the dispatch region, first without the isolate region from above
  // property.
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    return failure();
  }

  // Run canonicalization patterns and pattern to resolve tensor.dim of result
  // values into tensor.dim of its operands..
  RewritePatternSet canonicalizationPatterns(context);
  memref::populateResolveRankedShapeTypeResultDimsPatterns(
      canonicalizationPatterns);
  if (failed(applyPatternsAndFoldGreedily(
          funcOp, std::move(canonicalizationPatterns)))) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After dispatch op formation ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // After outlining in dispatch region we can rewrite the dispatch ops with
  // proper captures to make it isolated from above.
  if (funcOp
          ->walk([&](IREE::Flow::DispatchCollectivesOp op) -> WalkResult {
            return legalizeDispatchCollectivesOperands(op);
          })
          .wasInterrupted()) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After dispatch op legalization ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Now try to see if we can tie certain results to operands in order to
  // indicate sharing storage. This need to happen here because it needs to
  // access region block arguments for input/output tensors, which aren't
  // available until now.
  funcOp->walk([&](IREE::Flow::DispatchCollectivesOp op) {
    tryToTieOperandsAndResults(op);
  });

  LLVM_DEBUG({
    llvm::dbgs() << "\n--- After tieing operands and results ---\n";
    funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    llvm::dbgs() << "\n\n";
  });

  // Finally fold `tensor.insert_slice/extract_slice` operations with
  // `flow.dispatch.tensor.load/store`.
  RewritePatternSet foldExtractInsertSliceOps(context);
  populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
      foldExtractInsertSliceOps, context);
  if (failed(applyPatternsAndFoldGreedily(
          funcOp, std::move(foldExtractInsertSliceOps)))) {
    return failure();
  }

  return success();
}

/// Pass declaration.
struct DispatchCCLPass
    : public DispatchCCLBase<DispatchCCLPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<ccl::CCLDialect, FlowDialect>();
  }
  DispatchCCLPass() = default;
  DispatchCCLPass(const DispatchCCLPass &pass) {}
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();
    RewritePatternSet rewritePatterns(context);
    rewritePatterns.add<
      CreateCollectivesDispatchRegionOp<ccl::CCLOp, OpInterfaceRewritePattern>>(
        context);
    if (failed(createDispatchRegionsFromRootOps(
        funcOp, std::move(rewritePatterns)))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "\n--- After first step of dispatch region formation ---\n";
      funcOp->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>> createDispatchCCLPass() {
  return std::make_unique<DispatchCCLPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
