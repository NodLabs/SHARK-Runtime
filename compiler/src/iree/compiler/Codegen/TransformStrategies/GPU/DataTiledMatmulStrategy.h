// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_DATA_TILED_MATMUL_STRATEGY_H_
#define IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_DATA_TILED_MATMUL_STRATEGY_H_

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

class DataTiledMatmulStrategy : public AbstractGemmLikeStrategy {
public:
  DataTiledMatmulStrategy(MLIRContext *context,
                          const transform_ext::MatchedMatmulCaptures &captures,
                          const GPUModel &gpuModel)
      : AbstractGemmLikeStrategy(gpuModel), ctx(context), captures(captures) {
    initDefaultValues(gpuModel);
  }

  DataTiledMatmulStrategy(const DataTiledMatmulStrategy &) = default;
  DataTiledMatmulStrategy &operator=(const DataTiledMatmulStrategy &) = default;

  /// Constructor quantities.
  MLIRContext *ctx;
  transform_ext::MatchedMatmulCaptures captures;

  /// Initialize values from the CLI. Set cliOptionsSpecified to true if the
  /// default CLI values have been overriden.
  void initDefaultValues(const GPUModel &gpuModel) override;

  LogicalResult validate(const GPUModel &gpuModel) const override;

  int64_t m() const override {
    int64_t mElements = 1;
    for (auto i : captures.contractionDims.m) {
      mElements *= captures.matmulOpSizes[i];
    }
    return mElements;
  }
  int64_t n() const override {
    int64_t nElements = 1;
    for (auto i : captures.contractionDims.n) {
      nElements *= captures.matmulOpSizes[i];
    }
    return nElements;
  }
  int64_t k() const override {
    int64_t kElements = 1;
    for (auto i : captures.contractionDims.k) {
      kElements *= captures.matmulOpSizes[i];
    }
    return kElements;
  }

  int64_t blockTileM() const override { return blockTileSizes[0]; }
  int64_t blockTileN() const override {
    return captures.matmulOpSizes[captures.contractionDims.n.back()];
  }

  int64_t numWarpsX() const override { return numWarps[0]; }
  int64_t numWarpsY() const override { return 1; }

  Type getLhsElementalType() const override { return captures.lhsElementType; }
  Type getRhsElementalType() const override { return captures.rhsElementType; }
  Type getResElementalType() const override {
    return captures.outputElementType;
  }

  MappingInfo getBlockMapping() const override {
    SmallVector<int64_t> tileSizes;
    SmallVector<Attribute> threadMapping = {blockX(ctx)};
    // Outer output channel.
    if (captures.contractionDims.n.size() == 2) {
      tileSizes.push_back(1);
      threadMapping = {blockY(ctx), blockX(ctx)};
    }
    tileSizes.push_back(blockTileM());
    return MappingInfo{/*numThreads=*/{},
                       /*tileSizes=*/tileSizes,
                       /*threadMapping=*/threadMapping,
                       /*vectorSize=*/std::nullopt};
  }

  // LHS copy is of size mxk.
  MappingInfo lhsCopyMapping() const override {
    int64_t kInnerTileSize =
        captures.matmulOpSizes[captures.contractionDims.k.back()];
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/k(),
        /*copySizes=*/captures.contractionDims.k.size() == 2
            ? ArrayRef<int64_t>{1, blockTileM(), kInnerTileSize}
            : ArrayRef<int64_t>{blockTileM(), kInnerTileSize},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/lhsElementalBitWidth());
  }
  // TODO: Implement validator.
  LogicalResult validateLhsCopyMapping() const override { return success(); }

  // RHS copy is of size kxn.
  MappingInfo rhsCopyMapping() const override {
    int64_t kInnerTileSize =
        captures.matmulOpSizes[captures.contractionDims.k.back()];
    int64_t nInnerTileSize =
        captures.matmulOpSizes[captures.contractionDims.n.back()];
    MappingInfo mapping = CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/k(),
        /*copySizes=*/ArrayRef<int64_t>{nInnerTileSize, kInnerTileSize},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/rhsElementalBitWidth());
    if (captures.contractionDims.n.size() == 2) {
      mapping.tileSizes.insert(mapping.tileSizes.begin(), 1);
      mapping.numThreads.insert(mapping.numThreads.begin(), 0);
    }
    if (captures.contractionDims.k.size() == 2) {
      mapping.tileSizes.insert(mapping.tileSizes.begin(), 1);
      mapping.numThreads.insert(mapping.numThreads.begin(), 0);
    }
    return mapping;
  }
  // TODO: Implement validator.
  LogicalResult validateRhsCopyMapping() const override { return success(); }

  // RES copy is of size mxn.
  MappingInfo resCopyMapping() const override {
    int64_t nInnerTileSize =
        captures.matmulOpSizes[captures.contractionDims.n.back()];
    return CopyMapping::getMappingInfo(
        ctx, totalNumThreads(),
        /*alignment=*/n(),
        /*copySizes=*/captures.contractionDims.n.size() == 2
            ? ArrayRef<int64_t>{1, blockTileM(), nInnerTileSize}
            : ArrayRef<int64_t>{blockTileM(), nInnerTileSize},
        /*favorPredication=*/false,
        /*elementalBitWidth=*/resElementalBitWidth());
  }
  // TODO: Implement validator.
  LogicalResult validateResCopyMapping() const override { return success(); }

  // COMPUTE is of size mxn.
  MappingInfo computeMapping() const override {
    if (useFma) {
      // When using FMA we don't need to map to warps, instead just match what
      // the copy does.
      return resCopyMapping();
    }
    return MappingInfo{/*numThreads=*/captures.contractionDims.n.size() == 2
                           ? SmallVector<int64_t>{0, numWarpsX()}
                           : SmallVector<int64_t>{numWarpsX()},
                       /*tileSizes=*/{},
                       /*threadMapping=*/{warpX(ctx)},
                       /*vectorSize=*/std::nullopt};
  }

  void print(llvm::raw_ostream &os) const override;
  LLVM_DUMP_METHOD void dump() const override;
};

} // namespace gpu
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_CODEGEN_TRANSFORM_DIALECT_STRATEGIES_GPU_DATA_TILED_MATMUL_STRATEGY_H_
