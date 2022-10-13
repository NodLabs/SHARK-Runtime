// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-flow-outline-dispatch-regions %s | FileCheck %s

func.func @allreduce(%arg0: tensor<1x2xf32>, %arg1: !ccl.communicator, %arg2: !ccl.chain) -> (tensor<1x2xf32>, !ccl.chain) {
  %c1 = arith.constant 1 : index
  %0:2 = flow.dispatch.collectives[%c1](%arg0, %arg1, %arg2) : (tensor<1x2xf32>, !ccl.communicator, !ccl.chain) -> (tensor<1x2xf32>, !ccl.chain) =
      (%arg3: !flow.dispatch.tensor<readonly:1x2xf32>, %arg4: !ccl.communicator, %arg5: !ccl.chain, %arg6: !flow.dispatch.tensor<writeonly:1x2xf32>, %arg7: !ccl.chain) {
    %1 = flow.dispatch.tensor.load %arg3, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : !flow.dispatch.tensor<readonly:1x2xf32> -> tensor<1x2xf32>
    %result, %out_chain = ccl.allreduce sum, %1, %arg4, %arg5 : (tensor<1x2xf32>, !ccl.communicator, !ccl.chain) -> (tensor<1x2xf32>, !ccl.chain)
    flow.dispatch.tensor.store %result, %arg6, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : tensor<1x2xf32> -> !flow.dispatch.tensor<writeonly:1x2xf32>
    flow.return
  }
  return %0#0, %0#1 : tensor<1x2xf32>, !ccl.chain
}
//      CHECK: flow.executable private @allreduce_dispatch_collectives_0
// CHECK-NEXT:   flow.executable.export public @allreduce_dispatch_collectives_0
//      CHECK:     func.func @allreduce_dispatch_collectives_0(
// CHECK-SAME:     %[[in_tensor:[a-zA-Z0-9_]+]]: !flow.dispatch.tensor<readonly:1x2xf32>
// CHECK-SAME:     %[[communicator:[a-zA-Z0-9_]+]]: !ccl.communicator
// CHECK-SAME:     %[[chain:[a-zA-Z0-9_]+]]: !ccl.chain
//      CHECK: func.func @allreduce(
// CHECK-SAME: %[[in_tensor:[a-zA-Z0-9_]+]]: tensor<1x2xf32>
// CHECK-SAME: %[[communicator:[a-zA-Z0-9_]+]]: !ccl.communicator
// CHECK-SAME: %[[in_chain:[a-zA-Z0-9_]+]]: !ccl.chain
// CHECK-SAME: -> (tensor<1x2xf32>, !ccl.chain)
//      CHECK:   %[[c1:[a-zA-Z0-9_]+]] = arith.constant 1 : index
//      CHECK:   %[[res:[a-zA-Z0-9_]+]]:2 = flow.dispatch @allreduce_dispatch_collectives_0::@allreduce_dispatch_collectives_0
// CHECK-SAME:   [%[[c1]]](%[[in_tensor]], %[[communicator]], %[[in_chain]])
// CHECK-SAME:   : (tensor<1x2xf32>, !ccl.communicator, !ccl.chain) -> (tensor<1x2xf32>, !ccl.chain)
//      CHECK:   return %[[res]]#0, %[[res]]#1 : tensor<1x2xf32>, !ccl.chain
