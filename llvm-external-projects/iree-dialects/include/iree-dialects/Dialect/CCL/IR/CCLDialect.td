//===- CCLDialect.td - CCL dialect definition --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the CCL (Collective Communications Library) dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CCL_IR_CCLDIALECT
#define MLIR_DIALECT_CCL_IR_CCLDIALECT

include "mlir/IR/OpBase.td"

def CCL_Dialect : Dialect {
  let summary = "Collective Communications Library dialect";
  let description = [{
    CCL is a dialect that describes distributed communication very similar to
    MPI (Message Passing Interface) and
    NCCL (NVIDIA Collective Communications Library).
    This dialect differs from them in that it works with tensors instead of continous
    buffers.
    ```mlir
    %communicator = ccl.com_world : !ccl.communicator
    %rank = ccl.rank %communicator : i64
    %in_tensor = tensor.from_elements %rank :  tensor<1xi64>
    %out_tensor = linalg.init_tensor [1] :  tensor<1xi64>
    %in_chain = ccl.create_chain : !ccl.chain
    %reduced, %out_chain = ccl.allreduce sum, %in_tensor, %out_tensor, %communicator, %in_chain
      : (tensor<1xi64>, tensor<1xi64>, !ccl.communicator, !ccl.chain) -> (tensor<1xi64>, !ccl.chain)
    ```
  }];

  let name = "ccl";
  let cppNamespace = "::mlir::ccl";

  let useDefaultTypePrinterParser = 1;
}

#endif // MLIR_DIALECT_CCL_IR_CCLDIALECT
