//===- CCLOps.td - CCL dialect definition ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the CCL (Collective Communications Library) dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CCL_IR_CCLOPS
#define MLIR_DIALECT_CCL_IR_CCLOPS

include "mlir/IR/OpAsmInterface.td"
include "iree-dialects/Dialect/CCL/IR/CCLInterfaces.td"
include "iree-dialects/Dialect/CCL/IR/CCLTypes.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"

def CCL_TensorLike : AnyTypeOf<[AnyTensor, AnyMemRef, AnyUnrankedMemRef]>;

class CCL_Op<string mnemonic, list<Trait> traits = []>
    : Op<CCL_Dialect, mnemonic, traits # [CCL_OpInterface]> {
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

def CCL_CommWorldOp : CCL_Op<"comm_world"> {
  let summary = "Get the global/world communicator.";
  let results = (outs CCL_Communicator:$result);
  let assemblyFormat = "attr-dict";
}

def CCL_CommDupOp : CCL_Op<"comm_dup"> {
  let summary = "Clones a communicator.";
	let arguments = (ins CCL_Communicator:$communicator);
  let results = (outs CCL_Communicator:$result);
  let assemblyFormat = "$communicator attr-dict `:` type($result)";
}

def CCL_CommCreateOp : CCL_Op<"comm_create"> {
  let summary = "Create a communicator from a subset of ranks.";
  let arguments = (ins CCL_Communicator:$communicator,
      Variadic<CCL_Rank>:$ranks);
  let results = (outs CCL_Communicator:$result);
}

def CCL_CommDestroyOp : CCL_Op<"comm_destroy"> {
  let summary = "Destroy a communicator.";
  let arguments = (ins CCL_Communicator:$communicator);
}

def CCL_RankOp : CCL_Op<"rank"> {
  let summary = "Get the rank of this process.";
  let arguments = (ins CCL_Communicator:$communicator);
  let results = (outs CCL_Rank:$result);
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def CCL_CreateChainOp : CCL_Op<"create_chain"> {
  let results = (outs CCL_Chain:$result);
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def CCL_MergeChainsOp : CCL_Op<"merge_chains"> {
  let arguments = (ins Variadic<CCL_Chain>:$in_chains);
  let results = (outs CCL_Chain:$result_chain);
  let assemblyFormat = "$in_chains attr-dict `:` type($result_chain)";
}

def CCL_SendOp : CCL_Op<"send"> {
  let arguments = (ins
      CCL_TensorLike:$in_tensor,
      CCL_Rank:$destination,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain);
  let results = (outs CCL_Chain:$result_chain);
}

def CCL_RecvOp : CCL_Op<"recv",
    [TypesMatchWith<"result_tensor type matches type of out_tensor",
                    "out_tensor", "result_tensor",
                    "$_self.cast<ShapedType>()">]> {
  let arguments = (ins
      CCL_TensorLike:$out_tensor,
      CCL_Rank:$source,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain);
  let results = (outs
    CCL_TensorLike:$result_tensor,
    CCL_Chain:$result_chain);
}

def CCL_BcastOp : CCL_Op<"bcast",
    [TypesMatchWith<"result_tensor type matches type of io_tensor",
                    "io_tensor", "result_tensor",
                    "$_self.cast<ShapedType>()">]> {
  let arguments = (ins
      CCL_TensorLike:$io_tensor,
      CCL_Rank:$root,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain);
  let results = (outs
      CCL_TensorLike:$result_tensor,
      CCL_Chain:$result_chain);
}

def CCL_ReduceOp : CCL_Op<"reduce"> {
  let description = [{
    Returns a non-empty optional value only when the current rank = `root`.
  }];
  let arguments = (ins
      CCL_TensorLike:$in_tensor,
      CCL_TensorLike:$out_tensor,
      CCL_Rank:$root,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain,
      DefaultValuedAttr<CCL_OpEnum, "OpEnum::SUM">:$reduction_op);
  let results = (outs
      CCL_TensorLike: $result_tensor,
      CCL_Chain:$result_chain);
  let assemblyFormat = "$reduction_op `,` operands attr-dict `:` functional-type(operands, results)";
}

def CCL_AllReduceOp : CCL_Op<"allreduce",
    [TypesMatchWith<"result_tensor type matches type of out_tensor",
                    "out_tensor", "result_tensor",
                    "$_self.cast<ShapedType>()">]> {
  let arguments = (ins
      CCL_TensorLike:$in_tensor,
      CCL_TensorLike:$out_tensor,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain,
      DefaultValuedAttr<CCL_OpEnum, "OpEnum::SUM">:$reduction_op);
  let results = (outs
      CCL_TensorLike:$result_tensor,
      CCL_Chain:$result_chain);
  let assemblyFormat = "$reduction_op `,` operands attr-dict `:` functional-type(operands, results)";
}

def CCL_ReduceScatterOp : CCL_Op<"reduce_scatter"> {
  let arguments = (ins
      CCL_TensorLike:$in_tensor,
      CCL_TensorLike:$out_tensor,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain,
      DefaultValuedAttr<CCL_OpEnum, "OpEnum::SUM">:$reduction_op);
  let results = (outs
      CCL_TensorLike:$result_tensor,
      CCL_Chain:$result_chain);
  let assemblyFormat = "$reduction_op `,` operands attr-dict `:` functional-type(operands, results)";
}

def CCL_AllGatherOp : CCL_Op<"allgather"> {
  let description = [{
    Stacks along a new dimension `dim`, analogous to `numpy.stack`.
  }];
  let arguments = (ins
      CCL_TensorLike:$in_tensor,
      CCL_TensorLike:$out_tensor,
      CCL_Communicator:$communicator,
      CCL_Chain:$in_chain,
      DefaultValuedAttr<I32Attr, "0">:$dim);
  let results = (outs
      CCL_TensorLike:$result_tensor,
      CCL_Chain:$result_chain);
}

#endif // MLIR_DIALECT_CCL_IR_CCLOPS
