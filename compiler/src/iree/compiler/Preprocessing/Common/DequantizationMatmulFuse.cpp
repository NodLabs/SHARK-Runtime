

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {

namespace {

//-----------------------------------------------------------//
//                        Utility
//-----------------------------------------------------------//

static LogicalResult fuseDequantAndMatmul(RewriterBase &rewriter,
                                          Operation *dequant,
                                          Operation *matmul) {
  Flow::DispatchRegionOp regionOp =
      matmul->getParentOfType<Flow::DispatchRegionOp>();
  if (!regionOp) {
    FailureOr<Flow::DispatchRegionOp> maybeRegionOp =
        Flow::wrapOpInDispatchRegion(rewriter, matmul);
    if (failed(maybeRegionOp))
      return failure();
    regionOp = maybeRegionOp.value();
  }

  FailureOr<Flow::DispatchRegionOp> maybeFusedRegionOp =
      movePrecedingOpsIntoDispatchRegion(rewriter, dequant, regionOp);
  if (failed(maybeFusedRegionOp))
    return failure();

  return success();
}

static LogicalResult isMatmulOnGroupedInput(linalg::GenericOp op) {
  if (op.getNumResults() != 1)
    return failure();
  if (op.getNumOperands() != 3)
    return failure();

  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();
  auto rank = iteratorTypes.size();
  if (rank < 4)
    return failure();

  // Check that last two iterator types are reduction and the rest are parallel
  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  for (auto [index, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType != parallel && rank - index > 2)
      return failure();
    if (iteratorType != reduction && rank - index <= 2)
      return failure();
  }

  return success();
}

static LogicalResult isGroupedDequantizationOp(linalg::GenericOp op) {
  if (op.getNumResults() != 1)
    return failure();
  if (op.getNumOperands() != 4)
    return failure();

  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();
  auto rank = iteratorTypes.size();
  if (rank < 3)
    return failure();

  // Check that all iterator types are parallel
  auto parallel = utils::IteratorType::parallel;
  for (utils::IteratorType iteratorType : iteratorTypes) {
    if (iteratorType != parallel)
      return failure();
  }

  return success();
}

//-----------------------------------------------------------//
//                        Patterns
//-----------------------------------------------------------//

class DequantizationMatmulFusePattern final
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Match first generic op as matmul
    if (failed(isMatmulOnGroupedInput(genericOp)))
      return failure();

    // Fail if matmul has already been fused
    Value genericOpResult = genericOp->getResult(0);
    Operation *matmulOp = genericOpResult.getDefiningOp();
    if (matmulOp->getParentOfType<Flow::DispatchRegionOp>())
      return failure();

    // Match operands to dequantizations and fuse if matched
    Value lhs = genericOp->getOperand(0);
    Value rhs = genericOp->getOperand(1);
    auto lhsOp = lhs.getDefiningOp<linalg::GenericOp>();
    auto rhsOp = rhs.getDefiningOp<linalg::GenericOp>();

    LogicalResult succeeded = failure();
    if (lhsOp && !failed(isGroupedDequantizationOp(
                     llvm::dyn_cast<linalg::GenericOp>(*lhsOp)))) {
      if (!failed(fuseDequantAndMatmul(rewriter, lhsOp, matmulOp)))
        succeeded = success();
    }

    if (rhsOp && !failed(isGroupedDequantizationOp(
                     llvm::dyn_cast<linalg::GenericOp>(*rhsOp)))) {
      if (!failed(fuseDequantAndMatmul(rewriter, rhsOp, matmulOp)))
        succeeded = success();
    }

    return succeeded;
  }
};

struct DequantizationMatmulFusePass
    : public DequantizationMatmulFuseBase<DequantizationMatmulFusePass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, Flow::FlowDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    // Main pattern.
    {
      RewritePatternSet patterns(&getContext());
      patterns.insert<DequantizationMatmulFusePattern>(context);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createDequantizationMatmulFusePass() {
  return std::make_unique<DequantizationMatmulFusePass>();
}

} // namespace IREE
} // namespace iree_compiler
} // namespace mlir