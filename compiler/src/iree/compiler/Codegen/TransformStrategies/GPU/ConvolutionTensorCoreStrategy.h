// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_CONVOLUTION_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_CONVOLUTION_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/AbstractGemmLikeStrategy.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/CopyMapping.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"

namespace llvm {
class raw_ostream;
}

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct GPUModel;

class DataTiledConvolutionStrategy : public AbstractGemmLikeStrategy {
public:
  DataTiledConvolutionStrategy(
      MLIRContext *context,
      const transform_ext::MatchedConvolutionCaptures &captures,
      const GPUModel &gpuModel)
      : AbstractGemmLikeStrategy(gpuModel), ctx(context), captures(captures) {
    initDefaultValues(gpuModel);
  }

  DataTiledConvolutionStrategy(const DataTiledConvolutionStrategy &) = default;
  DataTiledConvolutionStrategy &
  operator=(const DataTiledConvolutionStrategy &) = default;

  /// Constructor quantities.
  MLIRContext *ctx;
  transform_ext::MatchedConvolutionCaptures captures;

  /// Initialize values from the CLI. Set cliOptionsSpecified to true if the
  /// default CLI values have been overriden.
  void initDefaultValues(const GPUModel &gpuModel) override;

  LogicalResult validate(const GPUModel &gpuModel) const override;

  int64_t m() const override {
    int64_t imgElements = 1;
    for (auto i : captures.convolutionDims.outputImage) {
      imgElements *= captures.convolutionOpSizes[i];
    }
    return imgElements;
  }
  int64_t n() const override {
    int64_t ocElements = 1;
    for (auto i : captures.convolutionDims.outputChannel) {
      ocElements *= captures.convolutionOpSizes[i];
    }
    return ocElements;
  }
  int64_t k() const override {
    int64_t icElements = 1;
    for (auto i : captures.convolutionDims.outputChannel) {
      icElements *= captures.convolutionOpSizes[i];
    }
    return icElements;
  }

  int64_t blockTileM() const override {
    assert(blockTileSizes.size() >= 2 && "need at least 2 tile sizes");
    return blockTileSizes[0];
  }
  int64_t blockTileN() const override {
    assert(blockTileSizes.size() >= 2 && "need at least 2 tile sizes");
    return blockTileSizes[1];
  }

  int64_t numWarpsX() const override {
    assert(numWarps.size() >= 2 && "need at least 2 warp sizes");
    return numWarps[0];
  }
  int64_t numWarpsY() const override {
    assert(numWarps.size() >= 2 && "need at least 2 warp sizes");
    return numWarps[1];
  }

  Type getLhsElementalType() const override {
    return captures.inputElementType;
  }
  Type getRhsElementalType() const override {
    return captures.filterElementType;
  }
  Type getResElementalType() const override {
    return captures.outputElementType;
  }

  virtual bool alignedLhs() const { return true; }
  virtual bool alignedRhs() const { return true; }
  virtual bool alignedRes() const { return true; }

  bool hasLhsCopy() const override { return true; }
  // Filter is not copied.
  bool hasRhsCopy() const override { return false; }

  MappingInfo getBlockMapping() const override {
    SmallVector<int64_t> tileSizes;
    SmallVector<Attribute> threadMapping = {blockY(ctx), blockX(ctx)};
    // Outer output channel.
    if (captures.convolutionDims.outputChannel.size() == 2) {
      tileSizes.push_back(blockTileN());
      threadMapping = {blockZ(ctx), blockY(ctx), blockX(ctx)};
    }
    // Image height.
    tileSizes.push_back(1);
    // Image width.
    tileSizes.push_back(blockTileM());
    return MappingInfo{/*numThreads=*/{},
                       /*tileSizes=*/tileSizes,
                       /*threadMapping=*/threadMapping,
                       /*vectorSize=*/std::nullopt};
  }

  MappingInfo lhsCopyMapping() const override {
    int64_t inputTileH =
        captures.convolutionOpSizes[captures.convolutionDims.filterLoop[0]];
    int64_t inputTileW =
        captures.convolutionOpSizes[captures.convolutionDims.filterLoop[1]];
    +blockTileM() - 1;
    int64_t icInnerTileSize =
        captures
            .convolutionOpSizes[captures.convolutionDims.inputChannel.back()];
    MappingInfo mapping = CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/k(),
        /*copySizes=*/
        ArrayRef<int64_t>{inputTileH, inputTileW, icInnerTileSize},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/lhsElementalBitWidth());
    if (captures.convolutionDims.inputChannel.size() == 2) {
      mapping.tileSizes.insert(mapping.tileSizes.begin(), 1);
      mapping.numThreads.insert(mapping.numThreads.begin(), 0);
    }
    return mapping;
  }
  // TODO: Write a validator.
  LogicalResult validateLhsCopyMapping() const override { return success(); }

  // Filter is not copied.
  MappingInfo rhsCopyMapping() const override { return MappingInfo(); }
  LogicalResult validateRhsCopyMapping() const override { return success(); }

  MappingInfo resCopyMapping() const override {
    int64_t outputTileH = 1;
    int64_t outputTileW = blockTileM();
    int64_t ocInnerTileSize =
        captures
            .convolutionOpSizes[captures.convolutionDims.outputChannel.back()];
    MappingInfo mapping = CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/n(),
        /*copySizes=*/ArrayRef<int64_t>{blockTileM(), blockTileN()},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/resElementalBitWidth());
    if (captures.convolutionDims.inputChannel.size() == 2) {
      mapping.tileSizes.insert(mapping.tileSizes.begin(), 1);
      mapping.numThreads.insert(mapping.numThreads.begin(), 0);
    }
    return mapping;
  }
  // TODO: Write a validator.
  LogicalResult validateResCopyMapping() const override { return success(); }

  // COMPUTE is of size mxn.
  MappingInfo computeMapping() const override {
    // FMA disabled.
    // if (useFma) {
    //   // When using FMA we don't need to map to warps, instead just match
    //   what
    //   // the copy does.
    //   return CopyMapping::getMappingInfo(ctx, totalNumThreads(),
    //                                      /*alignment=*/n(),
    //                                      {blockTileM(), blockTileN()});
    // }
    return MappingInfo{
        /*numThreads=*/captures.convolutionDims.outputChannel.size() == 2
            ? SmallVector<int64_t>{0, 0, numWarpsY(), numWarpsX()}
            : SmallVector<int64_t>{0, numWarpsY(), numWarpsX()},
        /*tileSizes=*/{},
        /*threadMapping=*/{warpY(ctx), warpX(ctx)},
        /*vectorSize=*/std::nullopt};
  }

  void print(llvm::raw_ostream &os) const override;
  LLVM_DUMP_METHOD void dump() const override;
};

} // namespace gpu
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_TENSOR_CORE_CONVOLUTION_STRATEGY_H_
