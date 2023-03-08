// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using transform_ext::StructuredOpMatcher;

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

struct RaiseSpecialOpsPass : public RaiseSpecialOpsBase<RaiseSpecialOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override {
  SmallVector<std::pair<linalg::LinalgOp, SmallVector<Value>>> attentionRoots;
    getOperation()->walk([&](linalg::LinalgOp op) {
      {
        transform_ext::MatcherContext matcherContext;
        transform_ext::StructuredOpMatcher *queryCapture;
        transform_ext::StructuredOpMatcher *keyCapture;
        transform_ext::StructuredOpMatcher *valueCapture;
        transform_ext::StructuredOpMatcher *attentionroot;
        makeAttentionMatcher(matcherContext, queryCapture, keyCapture, valueCapture, attentionroot);

        if (matchPattern(op, *attentionroot)) {
          Value query = queryCapture->getCaptured()->getOperand(0);
          Value key = keyCapture->getCaptured()->getOperand(0);
          Value value = valueCapture->getCaptured()->getOperand(1);
          SmallVector<Value> src = {query, key, value};
          attentionRoots.push_back(std::make_pair(op, src));
        }
      }
    });
    for (std::pair<linalg::LinalgOp, SmallVector<Value>> attention : attentionRoots) {
      linalg::LinalgOp op = attention.first;
      SmallVector<Value> src = attention.second;
      IRRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(attention.first);
      rewriter.replaceOpWithNewOp<IREE::LinalgExt::AttentionOp>(
          op, op.getDpsInitOperand(0)->get().getType(), src, op.getDpsInitOperand(0)->get());
    }

    SmallVector<std::pair<linalg::LinalgOp, Value>> softmaxRoots;
    getOperation()->walk([&](linalg::LinalgOp op) {
      {
        transform_ext::MatcherContext matcherContext;
        transform_ext::StructuredOpMatcher *maxReduction;
        transform_ext::StructuredOpMatcher *softmaxroot;
        makeSoftmaxMatcher(matcherContext, maxReduction, softmaxroot);
        if (matchPattern(op, *softmaxroot)) {
          Value src = maxReduction->getCaptured()->getOperand(0);
          softmaxRoots.push_back(std::make_pair(op, src));
        }
      }
    });
    for (std::pair<linalg::LinalgOp, Value> softmax : softmaxRoots) {
      linalg::LinalgOp op = softmax.first;
      Value src = softmax.second;
      IRRewriter rewriter(op.getContext());
      rewriter.setInsertionPoint(softmax.first);
      rewriter.replaceOpWithNewOp<IREE::LinalgExt::SoftmaxOp>(
          op, src, op.getDpsInitOperand(0)->get(), op.getNumLoops() - 1);
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createRaiseSpecialOps() {
  return std::make_unique<RaiseSpecialOpsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
