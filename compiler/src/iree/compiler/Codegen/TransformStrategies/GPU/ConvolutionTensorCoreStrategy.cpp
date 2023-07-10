// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformStrategies/GPU/ConvolutionTensorCoreStrategy.h"

#include "iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.h"
#include "iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.h"
#include "iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensions.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/MappingInfo.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Strategies.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

#define DEBUG_TYPE "iree-transform-builder"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// TODO: significantly better namespacing.
using iree_compiler::buildPad;
using iree_compiler::buildSelectFirstNonEmpty;
using iree_compiler::buildTileFuseDistToForallWithNumThreads;
using iree_compiler::buildTileFuseDistToForallWithTileSizes;
using iree_compiler::TileToForallAndFuseAndDistributeResult;
using iree_compiler::gpu::buildBufferize;
using iree_compiler::gpu::buildConvertToAsyncCopies;
using iree_compiler::gpu::buildConvertToTensorCoreOp;
using iree_compiler::gpu::buildDistributeMatmulCopies;
using iree_compiler::gpu::buildHoistOutputPaddingOp;
using iree_compiler::gpu::DataTiledConvolutionStrategy;
using iree_compiler::gpu::MappingInfo;
using iree_compiler::gpu::scaleUpByBitWidth;
using iree_compiler::IREE::transform_dialect::EliminateGpuBarriersOp;
using iree_compiler::IREE::transform_dialect::
    IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp;
using transform::FuseIntoContainingOp;
using transform::MatchOp;
using transform_ext::RegisterMatchCallbacksOp;

void DataTiledConvolutionStrategy::initDefaultValues(const GPUModel &gpuModel) {
  // Set the configuration for padding the matmul.
  paddingValueTypes = {captures.inputElementType, captures.filterElementType,
                       captures.outputElementType};
  paddingDimensions = {0, 1, 2};
  packingDimensions = {1, 0, 1};

  // Pull in tile configs from flags.
  AbstractGemmLikeStrategy::initDefaultValues(gpuModel);
  if (!cliOptionsSpecified) {
    blockTileSizes[1] = 1;
    while (m() % blockTileSizes[0]) {
      blockTileSizes[0] /= 2;
    }
    useWmma = true;
  }
}

LLVM_DUMP_METHOD void DataTiledConvolutionStrategy::dump() const {
  print(llvm::errs());
}

void DataTiledConvolutionStrategy::print(llvm::raw_ostream &os) const {
  os << "\n--- Data Tiled Convolution strategy ---\n";
  AbstractGemmLikeStrategy::print(os);
}

// TODO: implement validator.
LogicalResult
DataTiledConvolutionStrategy::validate(const GPUModel &gpuModel) const {
  return success();
}

static std::tuple<Value, Value, Value, Value, Value>
buildDataTiledConvolutionStrategyBlockDistribution(
    ImplicitLocOpBuilder &b, Value variantH,
    const DataTiledConvolutionStrategy &strategy) {
  // Step 1. Call the matcher. Note that this is the same matcher as used to
  // trigger this compilation path, so it must always apply.
  b.create<RegisterMatchCallbacksOp>();
  auto [padH, fillH, convH, maybeTrailingH] = unpackRegisteredMatchCallback<4>(
      b, "convolution", transform::FailurePropagationMode::Propagate, variantH);

  // Step 2. Create the block/mapping tiling level and fusee.
  auto [fusionTargetH, fusionGroupH] =
      buildSelectFirstNonEmpty(b, maybeTrailingH, convH);
  MappingInfo blockMapping = strategy.getBlockMapping();
  TileToForallAndFuseAndDistributeResult tileResult =
      buildTileFuseDistToForallWithTileSizes(
          /*builder=*/b,
          /*variantH=*/variantH,
          /*rootH=*/convH,
          /*opsToFuseH=*/fusionGroupH,
          /*tileSizes=*/
          getAsOpFoldResult(b.getI64ArrayAttr(blockMapping.tileSizes)),
          /*threadDimMapping=*/
          b.getArrayAttr(blockMapping.threadMapping));

  auto [blockConvH, maybeBlockTrailingH] = buildSelectFirstNonEmpty(
      b, tileResult.resultingFusedOpsHandles.front(), tileResult.tiledOpH);

  Value fusedPadH =
      b.create<FuseIntoContainingOp>(padH, tileResult.forallH).getFusedOp();
  Value fusedFillH =
      b.create<FuseIntoContainingOp>(fillH, tileResult.forallH).getFusedOp();

  // Handle the workgroup count region.
  b.create<IREEPopulateWorkgroupCountRegionUsingNumThreadsSliceOp>(
      tileResult.forallH);

  return std::make_tuple(fusedPadH, fusedFillH, blockConvH, maybeBlockTrailingH,
                         tileResult.forallH);
}

/// Builds the common part of the schedule for matmuls and batched matmuls.
static void buildCommonConvolutionLikeThreadSchedule(
    ImplicitLocOpBuilder &b, Value variantH, Value padH, Value fillH,
    Value convH, Value trailingH,
    const DataTiledConvolutionStrategy &strategy) {
  using mlir::iree_compiler::buildLowerVectorMasksAndCleanup;
  using mlir::iree_compiler::buildTileFuseToScfFor;
  using namespace mlir::iree_compiler::gpu;

  // Tile the outer input channel dimension.
  if (strategy.captures.convolutionDims.inputChannel.size() > 1) {
    SmallVector<int64_t> tileSizes(
        strategy.captures.convolutionDims.outputChannel.size(), 0);
    tileSizes.append(strategy.captures.convolutionDims.outputImage.size(), 0);
    // tileSizes.append(strategy.captures.convolutionDims.filterLoop.size(), 0);
    tileSizes.push_back(1);

    // Avoid canonicalizing before the pad to avoid folding away the
    // extract_slice on the output needed to hoist the output pad.
    auto tileReductionResult = buildTileFuseToScfFor(
        b, variantH, convH, {}, getAsOpFoldResult(b.getI64ArrayAttr(tileSizes)),
        /*canonicalize=*/false);
    convH = tileReductionResult.tiledOpH;
  }

  // Step 2. Pad the (batch) matmul op.
  auto paddedConvOpH = buildPad(
      b, convH, strategy.getZeroPadAttrFromElementalTypes(b).getValue(),
      strategy.paddingDimensions, strategy.packingDimensions);

  // Step 3. Hoist the padding of the output operand above the reduction loop.
  // The resulting fillOp will be mapped with the contraction using an SIMD
  // programming model.
  Value fillOpH = fillH;
  if (!strategy.alignedRes()) {
    fillOpH = buildHoistOutputPaddingOp(b, variantH, paddedConvOpH);
  }

  // Running canonicalization is required here to enable aligned pads to become
  // linalg.copy ops when rewriting in DPS.
  Value funcH =
      b.create<transform::MatchOp>(variantH, func::FuncOp::getOperationName());
  iree_compiler::buildCanonicalizationAndEnablingTransforms(b, funcH);

  // Step 4. Distribute pad and copies: SIMT programming model.
  // auto [lhsCopyOpH, rhsCopyOpH, copyBackOpH] =
  buildDistributeMatmulCopies(b, variantH, paddedConvOpH, strategy);

  // Step 5. Tile the filter loop dimensions.
  SmallVector<int64_t> tileSizes(
      strategy.captures.convolutionDims.outputChannel.size(), 0);
  tileSizes.append(strategy.captures.convolutionDims.outputImage.size(), 0);
  tileSizes.append(strategy.captures.convolutionDims.filterLoop.size(), 1);

  auto tileReductionResult =
      buildTileFuseToScfFor(b, variantH, paddedConvOpH, {},
                            getAsOpFoldResult(b.getI64ArrayAttr(tileSizes)),
                            /*canonicalize=*/true);
  Value filterTiledConvH = tileReductionResult.tiledOpH;

  // Step 6. Distribute to warps: SIMD programming model.
  // TODO: get the number of warps from strategy.
  MappingInfo computeMapping = strategy.computeMapping();
  buildTileFuseDistToForallWithNumThreads(
      b, variantH, filterTiledConvH, ValueRange(),
      getAsOpFoldResult(b.getI64ArrayAttr(computeMapping.numThreads)),
      b.getArrayAttr(computeMapping.threadMapping));
  buildTileFuseDistToForallWithNumThreads(
      b, variantH, fillOpH, ValueRange(),
      getAsOpFoldResult(b.getI64ArrayAttr(computeMapping.numThreads)),
      b.getArrayAttr(computeMapping.threadMapping));

  // Step 7. Apply vectorization + cleanups to what remains.
  funcH = iree_compiler::buildVectorize(b, funcH, /*applyCleanups=*/true);

  // Step 8. Bufferize and drop HAL descriptor from memref ops.
  variantH = buildBufferize(b, variantH);

  // Step 9. Post-bufferization mapping to blocks and threads.
  // Need to match again since bufferize invalidated all handles.
  // TODO: assumes a single func::FuncOp to transform, needs hardening.
  funcH = b.create<MatchOp>(variantH, func::FuncOp::getOperationName());
  funcH =
      buildMapToBlockAndThreads(b, funcH,
                                /*blockSize=*/strategy.numThreads,
                                /*warpDims=*/strategy.numWarps,
                                /*subgroupSize=*/strategy.targetSubgroupSize);
  funcH = b.create<EliminateGpuBarriersOp>(funcH);

  // Step 10. Convert to tensor core ops.
  // TODO: avoid consuming handles and returning here.
  funcH = buildConvertToTensorCoreOp(b, funcH, strategy);

  // Step 11. Late lowerings and cleanups.
  buildLowerVectorMasksAndCleanup(b, funcH);
}

void iree_compiler::gpu::buildConvolutionTensorCoreStrategy(
    ImplicitLocOpBuilder &b, Value variantH,
    const DataTiledConvolutionStrategy &strategy) {
  LLVM_DEBUG(strategy.print(DBGS()));

  // Step 1. Apply block-level part of the strategy, keeps everything fused.
  auto [padH, fillH, convH, trailingH, forall] =
      buildDataTiledConvolutionStrategyBlockDistribution(b, variantH, strategy);
  buildCommonConvolutionLikeThreadSchedule(b, variantH, padH, fillH, convH,
                                           trailingH, strategy);
}
