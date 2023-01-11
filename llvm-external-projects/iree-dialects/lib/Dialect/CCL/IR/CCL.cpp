#include "mlir/Dialect/CCL/IR/CCL.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/CCL/IR/CCLInterfaces.cpp.inc"

#include "mlir/Dialect/CCL/IR/CCLOpsEnums.cpp.inc"

#include "mlir/Dialect/CCL/IR/CCLOpsDialect.cpp.inc"

namespace mlir {
namespace ccl {

void CCLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/CCL/IR/CCLOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/CCL/IR/CCLOpsTypes.cpp.inc"
      >();
}

} // namespace ccl
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/CCL/IR/CCLOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/CCL/IR/CCLOpsTypes.cpp.inc"
