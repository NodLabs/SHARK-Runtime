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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

/// Matches an (linalg.fill -> )? linalg.matmul operation sequence and converts
/// it into a iree_codegen.ukernel.generic "accel_matmul_f32" operation, that is later lowered
/// into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface>
matchDAGForUKernel(RewriterBase &rewriter, linalg::MatmulOp op) {
  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value out = op.getDpsInitOperand(0)->get();
  auto outType = llvm::cast<ShapedType>(out.getType());

  Location loc = op.getLoc();
  Value m = rewriter.create<tensor::DimOp>(loc, lhs, 0);
  Value n = rewriter.create<tensor::DimOp>(loc, rhs, 0);
  Value k = rewriter.create<tensor::DimOp>(loc, rhs, 1);

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  auto fn = getFnNameAndDefAttrs("accel_matmul_f32", rewriter, targetAttr);
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{lhs, rhs}, out, ValueRange{m, n, k},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(0));
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
  RewritePatternSet patterns(context);
  // Enabling a lowering of an op to a microkernel is a trade-off between the
  // potential performance advantage of a microkernel over pure code generation
  // for that op, and the potential benefits of fusions. Indeed, once an op
  // lowered into a microkernel, it will never be fused at any MLIR level.
  // Since microkernels are linked as bitcode, they will still undergo LTO-like
  // optimization in their calling contexts, but we shouldn't expect this to
  // achieve similar results as fusing structured ops.
  patterns.insert<LowerToAccelUKernelPattern<linalg::MatmulOp>>(context);
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
