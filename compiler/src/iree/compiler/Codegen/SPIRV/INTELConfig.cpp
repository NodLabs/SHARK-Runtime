// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- INTELConfig.h - INTEL CodeGen Configurations ---------------------===//
//
// This file contains CodeGen configurations for INTEL GPUs.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include <iostream>

#define DEBUG_TYPE "iree-spirv-intel-config"

namespace mlir {
namespace iree_compiler {
namespace detail {

struct JointMatrixSize {
  int64_t m;
  int64_t n;
  int64_t k;
};

struct TileWorkgroupSizePair {
  // How many scalar elements each workgroup should handle along each dimension.
  std::array<int64_t, 3> tileSize;
  std::array<int64_t, 3> workgroupSize;
};

/// Returns the joint matrix (M, N, K) sizes that are supported by the
/// target environment and match the given parameters.
static Optional<JointMatrixSize> getJointMatrixSize(
    spirv::ResourceLimitsAttr resourceLimits, Type lhsType, Type rhsType,
    Type resultType, int64_t m, int64_t n, int64_t k) {
    /*std::cout<<"resourceLimits.getCooperativeMatrixPropertiesNv() is:\n";
  resourceLimits.getCooperativeMatrixPropertiesNv().dump();
  auto properties = resourceLimits.getCooperativeMatrixPropertiesNv()
                        .getAsRange<spirv::CooperativeMatrixPropertiesNVAttr>();
  for (auto property : properties) {
    if (property.getAType() == lhsType && property.getBType() == rhsType &&
        property.getCType() == resultType &&
        property.getResultType() == resultType &&
        property.getScope().getValue() == spirv::Scope::Subgroup) {
      int matmulM = property.getMSize();
      int matmulN = property.getNSize();
      int matmulK = property.getKSize();*/
      if (m % 8 == 0 && n % 8 == 0 && k % 16 == 0) {
        return JointMatrixSize{8, 8, 16};
      }
    /*}
  }*/
  return llvm::None;
}

static LogicalResult setOpConfig(const spirv::TargetEnv &targetEnv,
                                 linalg::MatmulOp op) {
  // This configuration is only for joint matrix.
  std::cout<<"In the set op config\n";
  if (!targetEnv.allows(spirv::Capability::JointMatrixINTEL) ||
      !targetEnv.allows(spirv::Extension::SPV_INTEL_joint_matrix)) {
    std::cout<<"Taking the early return\n";
    return success();
  }

  Value lhs = op.getInputs()[0], rhs = op.getInputs()[1],
        init = op.getOutputs()[0];

  ArrayRef<int64_t> lhsShape = lhs.getType().cast<ShapedType>().getShape();
  ArrayRef<int64_t> rhsShape = rhs.getType().cast<ShapedType>().getShape();
  if (llvm::any_of(lhsShape, ShapedType::isDynamic)) return success();
  if (llvm::any_of(rhsShape, ShapedType::isDynamic)) return success();

  // TODO: Joint matrix support is fairly restricted. We can only have
  // a curated list of fused element wise ops as defined in the extension
  // SPV_INTEL_joint_matrix. Check that once we move bufferization after
  // vectorization.

  auto getElementType = [](Value v) {
    return v.getType().cast<ShapedType>().getElementType();
  };

  auto resourceLimits = targetEnv.getResourceLimits();
  auto coopMatSize = getJointMatrixSize(
      resourceLimits, getElementType(lhs), getElementType(rhs),
      getElementType(init), lhsShape[0], rhsShape[1], lhsShape[1]);
  std::cout<<"Last check before assigning pipeline\n";
  if (!coopMatSize) return success();
  std::cout<<"We do dispatch to joinstops\n";
  auto pipeline = IREE::Codegen::DispatchLoweringPassPipeline::
      SPIRVVectorizeToJointOps;

  // For now only support one subgroup per workgroup because in the above
  // configuration deduction step we only consider whether the input workload is
  // perfectly divisible by some native joint matrix size.
  //
  // TODO: Use some heuristics to deduce how many subgroups should be used and
  // the tile sizes for each subgroup, considering the input workload size and
  // native joint matrix size choices.
  int subgroupSize = 8;//resourceLimits.getSubgroupSize();
  std::array<int64_t, 3> workgroupSize = {2*subgroupSize, 2, 1};

  TileSizesListType tileSizes;
  // Again because we only consider whether the input workload is perfectly
  // divisible by some native joint matrix size, not some multiples of it,
  // need to make sure the subgroup tile sizes are the same as the workgroup
  // one.
  // Taking a different approach to tiling configuration

  /*tileSizes.push_back({coopMatSize->m, coopMatSize->n, coopMatSize->k});
  tileSizes.push_back({coopMatSize->m, coopMatSize->n, coopMatSize->k});*/
  tileSizes.push_back({16, 16, 16});
  tileSizes.push_back({8, 8, 16});


  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<func::FuncOp>(), op, tileSizes, pipeline,
      workgroupSize);
}

LogicalResult setINTELCodeGenConfig(const spirv::TargetEnv &targetEnv,
                                     Operation *rootOp) {
  std::cout<<"In INTEL config\n";
  int subgroupSize = 8;//targetEnv.getResourceLimits().getSubgroupSize();
  std::cout<<"subgroup size"<<subgroupSize<<std::endl;

  // First try to see if we can use tensor cores.
  if (auto matmulOp = dyn_cast<linalg::MatmulOp>(rootOp)) {
    std::cout<<"We have found a matmul\n";
    if (failed(setOpConfig(targetEnv, matmulOp))) { std::cout<<"It failed on us\n"; return failure();}
    if (getLoweringConfig(rootOp)) { std::cout<<"It succeeded on us\n"; return success();}
  }
  std::cout<<"It is here and it shouldnt be?\n";
  if (isa<linalg::BatchMatmulOp, linalg::MatmulOp>(rootOp)) {
    std::array<int64_t, 2> workgroupXY = {subgroupSize, 8};
    std::array<int64_t, 3> threadMNK = {4, 4, 32};
    return setMatmulOpConfig(rootOp, subgroupSize, workgroupXY, threadMNK,
                             /*useWorkgroupMemory=*/true);
  }

  return success();
}

}  // namespace detail
}  // namespace iree_compiler
}  // namespace mlir
