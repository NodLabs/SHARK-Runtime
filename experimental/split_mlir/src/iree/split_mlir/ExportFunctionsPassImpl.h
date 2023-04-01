// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <fstream>
#include <iterator>
#include <regex>

#include "iree/split_mlir/Passes.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/LocationSnapshot.h"

namespace mlir {
namespace iree {
namespace split_mlir {

#define GEN_PASS_DEF_EXPORTFUNCTIONS
#include "iree/split_mlir/Passes.h.inc"  // IWYU pragma: export

namespace {

template <typename RegexIt>
bool matchFunction(func::FuncOp op, std::cmatch& m,
                   iterator_range<RegexIt> regexRange) {
  StringRef name = op.getSymName();
  for (const std::regex& regex : regexRange) {
    if (std::regex_match(name.begin(), name.end(), m, regex)) {
      return true;
    }
  }

  return false;
}

std::string& makeExportedFunctionFilePath(StringRef prefix,
                                          StringRef functionName,
                                          std::string& result) {
  result.clear();
  result += prefix;
  result += functionName;
  result += ".mlir";
  return result;
}

void clearFunctionBody(func::FuncOp op) {
  op.getFunctionBody().getBlocks().clear();
  SymbolTable::setSymbolVisibility(op.getOperation(),
                                   SymbolTable::Visibility::Nested);
}

struct ExportFunctionsPass
    : public impl::ExportFunctionsBase<ExportFunctionsPass> {
  using ExportFunctionsBase::ExportFunctionsBase;

  LogicalResult initialize(MLIRContext* context) override {
    std::transform(functions.begin(), functions.end(),
                   std::back_inserter(regexList),
                   [](const std::string& val) { return std::regex(val); });
    return LogicalResult::success();
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    std::cmatch m;
    std::string filePath;
    moduleOp.walk([&regexList = regexList, &pathPrefix = pathPrefix, &m,
                   &filePath](func::FuncOp op) {
      if (!matchFunction(
              op, m, llvm::make_range(regexList.begin(), regexList.end()))) {
        return;
      }
      StringRef name = op.getSymName();
      makeExportedFunctionFilePath(pathPrefix, name, filePath);
      std::error_code error_code;
      llvm::raw_fd_ostream stream(filePath, error_code);
      if (error_code) {
        report_fatal_error("Error writing to file \"" + StringRef(filePath) +
                           "\". " + error_code.message());
      }
      generateLocationsFromIR(stream, filePath, op.getOperation(),
                              OpPrintingFlags());
      clearFunctionBody(op);
    });
  }

 private:
  llvm::SmallVector<std::regex, 3> regexList;
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createExportFunctionsPass() {
  return std::make_unique<ExportFunctionsPass>();
}

}  // namespace split_mlir
}  // namespace iree
}  // namespace mlir
