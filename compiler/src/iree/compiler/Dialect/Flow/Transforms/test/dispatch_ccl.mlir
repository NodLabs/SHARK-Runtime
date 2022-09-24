// RUN: iree-opt --split-input-file --verify-diagnostics --iree-flow-enable-multi-result-dispatches --pass-pipeline="func.func(iree-flow-dispatch-ccl-pass), cse, canonicalize, cse" %s | FileCheck %s

func.func @receive(
    %rank : index,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain) -> tensor<1x2xf32> {
  %res, %chain_out = ccl.recv %rank, %communicator, %chain :
      (index, !ccl.communicator, !ccl.chain) -> (tensor<1x2xf32>, !ccl.chain)
  return %res : tensor<1x2xf32>
}
//      CHECK: func.func @receive
// CHECK-SAME:   %[[rank:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[communicator:[a-zA-Z0-9_]+]]: !ccl.communicator
// CHECK-SAME:   %[[chain:[a-zA-Z0-9_]+]]: !ccl.chain
//      CHECK:     %[[res:[a-zA-Z0-9_]+]]:2 = flow.dispatch.collectives
// CHECK-SAME:     (%[[rank]], %[[communicator]], %[[chain]])
// CHECK-NEXT:     %[[rank0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:     %[[communicator0:[a-zA-Z0-9_]+]]: !ccl.communicator
// CHECK-SAME:     %[[chain0:[a-zA-Z0-9_]+]]: !ccl.chain
// CHECK-SAME:     %[[res0:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:1x2xf32>
// CHECK-SAME:     %[[out_chain0:[a-zA-Z0-9_]+]]: !ccl.chain
// CHECK-NEXT:     %[[res1:[a-zA-Z0-9_]+]], %[[out_chain1:[a-zA-Z0-9_]+]] = ccl.recv
// CHECK-SAME:     %[[rank0]], %[[communicator0]], %[[chain0]]
// CHECK-NEXT:     flow.dispatch.tensor.store %[[res1]], %[[res0]]
// CHECK-SAME:     offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : tensor<1x2xf32> -> !flow.dispatch.tensor<writeonly:1x2xf32>
// CHECK-NEXT:     flow.return
//      CHECK:   return %[[res]]#0 : tensor<1x2xf32>

// -----

func.func @allreduce(
    %in_tensor : tensor<1x2xf32>,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain) -> (tensor<1x2xf32>, !ccl.chain) {
   %res, %chain_out = ccl.allreduce sum, %in_tensor, %communicator, %chain :
       (tensor<1x2xf32>, !ccl.communicator, !ccl.chain) -> (tensor<1x2xf32>, !ccl.chain)
  return %res, %chain_out : tensor<1x2xf32>, !ccl.chain
}
//      CHECK: func.func @allreduce
// CHECK-SAME:   %[[in_tensor:[a-zA-Z0-9_]+]]: tensor<1x2xf32>
// CHECK-SAME:   %[[communicator:[a-zA-Z0-9_]+]]: !ccl.communicator
// CHECK-SAME:   %[[chain:[a-zA-Z0-9_]+]]: !ccl.chain
//      CHECK:     %[[res:[a-zA-Z0-9_]+]]:2 = flow.dispatch.collectives
// CHECK-SAME:     (%[[in_tensor]], %[[communicator]], %[[chain]])
// CHECK-NEXT:     %[[in_tensor0:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:1x2xf32>
// CHECK-SAME:     %[[communicator0:[a-zA-Z0-9_]+]]: !ccl.communicator
// CHECK-SAME:     %[[chain0:[a-zA-Z0-9_]+]]: !ccl.chain
// CHECK-SAME:     %[[out_tensor0:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<writeonly:1x2xf32>
// CHECK-SAME:     %[[out_chain0:[a-zA-Z0-9_]+]]: !ccl.chain
// CHECK-NEXT:     %[[in_tensor1:[a-zA-Z0-9_]+]] = flow.dispatch.tensor.load %[[in_tensor0]]
// CHECK-SAME:     offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : !flow.dispatch.tensor<readonly:1x2xf32> -> tensor<1x2xf32>
// CHECK-NEXT:     %[[out_tensor1:[a-zA-Z0-9_]+]], %[[out_chain1:[a-zA-Z0-9_]+]] = ccl.allreduce
// CHECK-SAME:     sum, %[[in_tensor1]], %[[communicator0]], %[[chain0]]
// CHECK-NEXT:     flow.dispatch.tensor.store %[[out_tensor1]], %[[out_tensor0]]
// CHECK-SAME:     offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : tensor<1x2xf32> -> !flow.dispatch.tensor<writeonly:1x2xf32>
// CHECK-NEXT:     flow.return
//      CHECK:   return %[[res]]#0, %[[res]]#1 : tensor<1x2xf32>, !ccl.chain
