//===- SPIRVMapMemRefStorageCLassPass.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to map numeric MemRef memory spaces to
// symbolic ones defined in the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

/// Identical to upstream's MapMemRefStorageClassPass without initializeOptions.
namespace {
class SPIRVMapMemRefStorageClassPass final
    : public SPIRVMapMemRefStorageClassBase<SPIRVMapMemRefStorageClassPass> {
public:
  explicit SPIRVMapMemRefStorageClassPass() {
    memorySpaceMap = spirv::mapMemorySpaceToVulkanStorageClass;
  }
  explicit SPIRVMapMemRefStorageClassPass(
      const spirv::MemorySpaceToStorageClassMap &memorySpaceMap)
      : memorySpaceMap(memorySpaceMap) {}

  void runOnOperation() override;

private:
  spirv::MemorySpaceToStorageClassMap memorySpaceMap;
};
} // namespace

/// Identical to upstream's MapMemRefStorageClassPass execpt we are using getSPIRVTargetEnvAttr instead of spirv::lookupTargetEnv.
void SPIRVMapMemRefStorageClassPass::runOnOperation() {
  MLIRContext *context = &getContext();
  Operation *op = getOperation();

  if (spirv::TargetEnvAttr attr = getSPIRVTargetEnvAttr(op)) {
    spirv::TargetEnv targetEnv(attr);
    if (targetEnv.allows(spirv::Capability::Kernel)) {
      memorySpaceMap = spirv::mapMemorySpaceToOpenCLStorageClass;
    } else if (targetEnv.allows(spirv::Capability::Shader)) {
      memorySpaceMap = spirv::mapMemorySpaceToVulkanStorageClass;
    }
  }

  auto target = spirv::getMemorySpaceToStorageClassTarget(*context);
  spirv::MemorySpaceToStorageClassConverter converter(memorySpaceMap);

  RewritePatternSet patterns(context);
  spirv::populateMemorySpaceToStorageClassPatterns(converter, patterns);

  if (failed(applyFullConversion(op, *target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<>> createSPIRVMapMemRefStorageClassPass() {
  return std::make_unique<SPIRVMapMemRefStorageClassPass>();
}

}  // namespace iree_compiler
}  // namespace mlir