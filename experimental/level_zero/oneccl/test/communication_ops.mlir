// RUN: mpirun -n 2 iree-run-mlir --iree-hal-target-backends=opencl-spirv %s --device=level_zero

//      CHECK: EXEC @main
// func.func @main() -> tensor<2x2xi32> {
//   //%input = util.unfoldable_constant dense<[1, 2]> : tensor<2xf32>
//   //%output = util.unfoldable_constant dense<[[0, 0], [0, 0]]> : tensor<2x2xf32>
//   %input = stream.tensor.constant : tensor<2xi32> in !stream.resource<constant> = dense<[101, 102]> : tensor<2xi32>
//   %fill_val = arith.constant -1 : i32
//   %c0 = arith.constant 0 : index
//   %c2 = arith.constant 2 : index
//   %c8 = arith.constant 8 : index
//   %c16 = arith.constant 16 : index
//   %output = stream.tensor.splat %fill_val :
//     i32 -> tensor<2x2xi32> in !stream.resource<external>{%c16}
//   %channel = stream.channel.create : !stream.channel
//   %0 = stream.cmd.execute with(%input as %input_arg: !stream.resource<constant>{%c8},
//                                %output as %output_arg: !stream.resource<external>{%c16}) {
//     stream.cmd.collective<all_gather : si32>[%c2] channel(%channel) {
//       ro %input_arg[%c0 for %c8] : !stream.resource<constant>{%c8},
//       rw %output_arg[%c0 for %c16] : !stream.resource<external>{%c16}
//     }
//   } => !stream.timepoint
//   %result = stream.tensor.export %output : tensor<2x2xi32> in !stream.resource<external>{%c16} -> tensor<2x2xi32>
//   return %result : tensor<2x2xi32>
// }

//      CHECK: EXEC @main
func.func @main() -> !hal.buffer_view {
  %input = stream.tensor.constant : tensor<2xi32> in !stream.resource<constant> =
    dense<[101, 102]> : tensor<2xi32>
  %fill_val = arith.constant -1 : i32
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %output = stream.tensor.splat %fill_val :
    i32 -> tensor<2x2xi32> in !stream.resource<external>{%c16}
  %channel = stream.channel.create : !stream.channel
  %time_point = stream.cmd.execute with(
      %input as %input_arg: !stream.resource<constant>{%c8},
      %output as %output_arg: !stream.resource<external>{%c16}) {
    stream.cmd.collective<all_gather : si32>[%c2] channel(%channel) {
      ro %input_arg[%c0 for %c8] : !stream.resource<constant>{%c8},
      rw %output_arg[%c0 for %c16] : !stream.resource<external>{%c16}
    }
  } => !stream.timepoint
  %output2 = stream.timepoint.await %time_point => %output : !stream.resource<external>{%c16}
  %result = stream.tensor.export %output2 :
    tensor<2x2xi32> in !stream.resource<external>{%c16} -> !hal.buffer_view
  return %result : !hal.buffer_view
}
