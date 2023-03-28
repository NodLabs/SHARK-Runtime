// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iterator>
#include <memory>

#include "iree/split_mlir/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree {
namespace split_mlir {

#define GEN_PASS_DEF_RENAMEFUNCTIONS
#include "iree/split_mlir/Passes.h.inc"  // IWYU pragma: export

namespace {

void renameDeclarations(
    ModuleOp moduleOp,
    const llvm::SmallDenseMap<StringRef, StringRef>& nameMap) {
  moduleOp.walk([&nameMap](func::FuncOp funcOp) {
    auto nameMapIt = nameMap.find(funcOp.getSymName());
    if (nameMapIt != nameMap.end()) {
      funcOp.setSymName(nameMapIt->second);
    }
  });
}

void renameCallSites(ModuleOp moduleOp,
                     const llvm::SmallDenseMap<StringRef, StringRef>& nameMap) {
  moduleOp.walk([moduleOp, &nameMap](func::CallOp callOp) {
    auto nameMapIt = nameMap.find(callOp.getCalleeAttr().getValue());
    if (nameMapIt != nameMap.end()) {
      callOp.setCalleeAttr(
          FlatSymbolRefAttr::get(moduleOp->getContext(), nameMapIt->second));
    }
  });
}

struct RenameFunctionsPass
    : public impl::RenameFunctionsBase<RenameFunctionsPass> {
  using RenameFunctionsBase::RenameFunctionsBase;

  LogicalResult initialize(MLIRContext* context) override {
    if (functions.size() != newNames.size()) {
      emitError(UnknownLoc::get(context),
                "Pass options functions and new-names must have the same "
                "number of elements.");
      return LogicalResult::failure();
    }

    auto functionsIt = functions.begin();
    auto newNamesIt = newNames.begin();
    for (; functionsIt != functions.end(); ++functionsIt, ++newNamesIt) {
      nameMap.insert({*functionsIt, *newNamesIt});
    }

    return LogicalResult::success();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    renameDeclarations(moduleOp, nameMap);
    renameCallSites(moduleOp, nameMap);
  }

 private:
  llvm::SmallDenseMap<StringRef, StringRef> nameMap;
};
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createRenameFunctionsPass() {
  return std::make_unique<RenameFunctionsPass>();
}

}  // namespace split_mlir
}  // namespace iree
}  // namespace mlir
