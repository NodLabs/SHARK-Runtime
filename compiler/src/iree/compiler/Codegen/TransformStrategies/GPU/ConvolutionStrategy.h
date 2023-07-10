// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_STRATEGY_H_

#include "iree-dialects/Transforms/TransformMatchers.h"
#include "iree/compiler/Codegen/TransformStrategies/Common/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Common.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/CopyMapping.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/MappingInfo.h"
#include "iree/compiler/Codegen/TransformStrategies/GPU/Strategies.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class raw_ostream;
}

namespace mlir {
namespace iree_compiler {
namespace gpu {

struct GPUModel;

class ConvolutionStrategy : public GPUStrategy {
public:
  ConvolutionStrategy(MLIRContext *context,
                      const transform_ext::MatchedConvolutionCaptures &captures,
                      const GPUModel &gpuModel)
      : GPUStrategy(gpuModel), ctx(context), captures(captures) {
    initDefaultValues(gpuModel);
  }

  ConvolutionStrategy(const ConvolutionStrategy &) = default;
  ConvolutionStrategy &operator=(const ConvolutionStrategy &) = default;

  /// Constructor quantities.
  MLIRContext *ctx;
  transform_ext::MatchedConvolutionCaptures captures;

  /// Initialize values from the CLI.
  void initDefaultValues(const GPUModel &gpuModel);

  LogicalResult validate(const GPUModel &gpuModel) const;

  //===--------------------------------------------------------------------===//
  // Parameters that control the tiling and mapping.
  //===--------------------------------------------------------------------===//

  /// Tile sizes for the workgroup / determines grid size for all known
  /// reduction strategies. The initial values are set by initDefaultValues();
  SmallVector<int64_t> blockTileSizes;
  int64_t reductionTileSize;
  SmallVector<int64_t> numThreads;
  SmallVector<int64_t> numWarps;

  /// Common values based on derived quantities.
  int64_t totalNumThreads() const {
    int64_t res = 1;
    for (auto v : numThreads)
      res *= v;
    return res;
  }

  int64_t totalNumWarps() const {
    int64_t res = 1;
    for (auto v : numWarps)
      res *= v;
    return res;
  }

  int64_t blockTileH() const {
    assert(blockTileSizes.size() >= 2 && "need at least 2 tile sizes");
    return blockTileSizes[0];
  }
  int64_t blockTileW() const {
    assert(blockTileSizes.size() >= 2 && "need at least 2 tile sizes");
    return blockTileSizes[1];
  }

  // int64_t numWarpsX() const {
  //   assert(numWarps.size() >= 2 && "need at least 2 warp sizes");
  //   return numWarps[0];
  // }
  // int64_t numWarpsY() const {
  //   assert(numWarps.size() >= 2 && "need at least 2 warp sizes");
  //   return numWarps[1];
  // }

  MappingInfo getBlockMapping() const {
    SmallVector<int64_t> tileSizes;
    SmallVector<Attribute> threadMapping = {blockY(ctx), blockX(ctx)};
    // Outer output channel.
    if (captures.convolutionDims.outputChannel.size() == 2) {
      tileSizes.push_back(1);
      threadMapping = {blockZ(ctx), blockY(ctx), blockX(ctx)};
    }
    // Image height.
    tileSizes.push_back(blockTileH());
    // Image width.
    tileSizes.push_back(blockTileW());
    return MappingInfo{/*numThreads=*/{},
                       /*tileSizes=*/tileSizes,
                       /*threadMapping=*/threadMapping,
                       /*vectorSize=*/std::nullopt};
  }

  MappingInfo computeMapping() const {
    int64_t innerOcTileSize =
        captures
            .convolutionOpSizes[captures.convolutionDims.outputChannel.back()];
    MappingInfo mapping = CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/innerOcTileSize,
        {blockTileH(), blockTileW(), innerOcTileSize});
    if (captures.convolutionDims.outputChannel.size() == 2) {
      mapping.tileSizes.insert(mapping.tileSizes.begin(), 1);
      mapping.numThreads.insert(mapping.numThreads.begin(), 0);
    }
    return mapping;
    // return MappingInfo{
    //     /*numThreads=*/captures.convolutionDims.outputChannel.size() == 2
    //         ? SmallVector<int64_t>{0, 0, numWarpsY(), numWarpsX()}
    //         : SmallVector<int64_t>{0, numWarpsY(), numWarpsX()},
    //     /*tileSizes=*/{},
    //     /*threadMapping=*/{warpY(ctx), warpX(ctx)},
    //     /*vectorSize=*/std::nullopt};
  }

  void print(llvm::raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

} // namespace gpu
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_CONVOLUTION_STRATEGY_H_
