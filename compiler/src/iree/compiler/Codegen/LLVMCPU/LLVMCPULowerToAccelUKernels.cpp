// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/builtins/ukernel/exported_bits.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/UKernelOps.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

class LLVMCPULowerToAccelUKernelsPass
    : public LLVMCPULowerToAccelUKernelsBase<LLVMCPULowerToAccelUKernelsPass> {
public:
  LLVMCPULowerToAccelUKernelsPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }

  void runOnOperation() override;

  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    return success();
  }
};

/// Holds a function name and attributes.
struct FnNameAndDefAttrs {
  std::string name;
  SmallVector<NamedAttribute> defAttrs;
};

/// Returns the function name and attributes to use for a ukernel with given
/// `ukernelName` on the target described by `targetAttr`.
static FnNameAndDefAttrs
getFnNameAndDefAttrs(const char *ukernelName, RewriterBase &rewriter,
                     IREE::HAL::ExecutableTargetAttr targetAttr) {
  FnNameAndDefAttrs result;
  result.name = ukernelName;
  result.defAttrs.emplace_back(
      rewriter.getStringAttr("hal.import.fields"),
      rewriter.getArrayAttr({rewriter.getStringAttr("processor_data"),
                             rewriter.getStringAttr("processor_id")}));
  result.defAttrs.emplace_back(
      rewriter.getStringAttr("hal.import.cconv"),
      IREE::HAL::CallingConventionAttr::get(
          rewriter.getContext(),
          IREE::HAL::CallingConvention::ParameterStruct));
  return result;
}

/// Matches an (linalg.fill -> )? linalg.mmt4d operation sequence and converts
/// it into a iree_codegen.ukernel.generic "accel_matmul_t_f32" operation, that is
/// later lowered into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface>
matchDAGForUKernel(RewriterBase &rewriter, linalg::Mmt4DOp op) {
  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value out = op.getDpsInitOperand(0)->get();
  //auto lhsType = llvm::cast<ShapedType>(lhs.getType());
  //auto rhsType = llvm::cast<ShapedType>(rhs.getType());
  auto outType = llvm::cast<ShapedType>(out.getType());
  /*
  Type lhsElemType = lhsType.getElementType();
  Type rhsElemType = rhsType.getElementType();
  Type outElemType = outType.getElementType();
  uint32_t flags = 0;
  if (lhsElemType.isSignlessInteger(8) && rhsElemType.isSignlessInteger(8) &&
      outElemType.isSignlessInteger(32)) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_I8I8I32;
  } else if (lhsElemType.isF32() && rhsElemType.isF32() &&
             outElemType.isF32()) {
    flags = IREE_UK_FLAG_MMT4D_TYPE_F32F32F32;
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types");
  }
  */
  Location loc = op.getLoc();

  if (outType.getShape()[0] != 1 || outType.getShape()[1] != 1) {
    return rewriter.notifyMatchFailure(op, "outer dims need to be 1");
  }
  auto outTypeRanked = out.getType().cast<RankedTensorType>();
  RankedTensorType intermediateOutType =
      RankedTensorType::Builder(outTypeRanked).dropDim(0);
  RankedTensorType reducedOutType =
      RankedTensorType::Builder(intermediateOutType).dropDim(0);
  Value reducedOut;
  Value initTensor;
  if (auto oldFillOp = out.getDefiningOp<linalg::FillOp>()) {
    initTensor = oldFillOp.output();
    auto newInit = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, initTensor, reducedOutType);
    reducedOut =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{oldFillOp.value()},
                                    ValueRange{newInit})
            .result();
  } else {
    reducedOut = tensor::createCanonicalRankReducingExtractSliceOp(
        rewriter, loc, out, reducedOutType);
    initTensor = out;
    }

  auto lhsTypeRanked = lhs.getType().cast<RankedTensorType>();
  RankedTensorType intermediateLhsType =
      RankedTensorType::Builder(lhsTypeRanked).dropDim(0);
  RankedTensorType reducedLhsType =
      RankedTensorType::Builder(intermediateLhsType).dropDim(0);
  auto reducedLhs = tensor::createCanonicalRankReducingExtractSliceOp(
      rewriter, loc, lhs, reducedLhsType);

  auto rhsTypeRanked = rhs.getType().cast<RankedTensorType>();
  RankedTensorType intermediateRhsType =
      RankedTensorType::Builder(rhsTypeRanked).dropDim(0);
  RankedTensorType reducedRhsType =
      RankedTensorType::Builder(intermediateRhsType).dropDim(0);
  auto reducedRhs = tensor::createCanonicalRankReducingExtractSliceOp(
      rewriter, loc, rhs, reducedRhsType);
  /*
    auto getDimAsI32 = [](RewriterBase &rewriter, Location loc, Value value,
                          int dim) -> Value {
      return rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getI32Type(),
          rewriter.create<tensor::DimOp>(loc, value, dim));
    };
    Value m = getDimAsI32(rewriter, loc, reducedLhs, 0);
    Value n = getDimAsI32(rewriter, loc, reducedRhs, 0);
    Value k = getDimAsI32(rewriter, loc, reducedRhs, 1);
  */
  Value m = rewriter.create<tensor::DimOp>(loc, reducedLhs, 0);
  Value n = rewriter.create<tensor::DimOp>(loc, reducedLhs, 1);
  Value k = rewriter.create<tensor::DimOp>(loc, reducedRhs, 0);
  //Value flagsVal = rewriter.create<arith::ConstantOp>(
  //    loc, rewriter.getI32IntegerAttr(flags));
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  auto fn = getFnNameAndDefAttrs("accel_matmul_t_f32", rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, reducedOutType, fn.name, ValueRange{reducedLhs, reducedRhs},
      reducedOut, ValueRange{m, n, k},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(0));
  auto insertSliceOp = tensor::createCanonicalRankReducingInsertSliceOp(
      rewriter, loc, genericMicroKernelOp.getResult(0), initTensor);
  op.getResults()[0].replaceAllUsesWith(insertSliceOp);
  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

template <typename OpType>
struct LowerToAccelUKernelPattern : OpRewritePattern<OpType> {
  LowerToAccelUKernelPattern(MLIRContext *context)
      : OpRewritePattern<OpType>(context) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    FailureOr<IREE::Codegen::UKernelOpInterface> ukernelOp =
        matchDAGForUKernel(rewriter, op);
    if (failed(ukernelOp)) {
      return rewriter.notifyMatchFailure(
          op, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(op, ukernelOp.value()->getResults());
    return success();
  }
};

void LLVMCPULowerToAccelUKernelsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  // Convert mmt4d ops to iree_codegen.ukernel.generic "accel_matmul_f32" ops.
  RewritePatternSet patterns(context);
  patterns.insert<LowerToAccelUKernelPattern<linalg::Mmt4DOp>>(context);
  // Canonicalize extract and insert slice ops created during the conversion.
  tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  tensor::InsertSliceOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExtractSliceOp::getCanonicalizationPatterns(patterns, context);
  //mlir::memref::populateResolveShapedTypeResultDimsPatterns(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

} // namespace

std::unique_ptr<OperationPass<>> createLLVMCPULowerToAccelUKernelsPass() {
  return std::make_unique<LLVMCPULowerToAccelUKernelsPass>();
}

} // namespace iree_compiler
} // namespace mlir
