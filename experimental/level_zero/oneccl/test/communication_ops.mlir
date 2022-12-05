// RUN: mpirun -n 2 iree-run-mlir --iree-hal-target-backends=opencl-spirv %s --device=level_zero

//      CHECK: EXEC @main
func.func @main() -> !hal.buffer_view {
  %input = stream.tensor.constant : tensor<2xi32> in !stream.resource<constant> =
    dense<[101, 102]> : tensor<2xi32>
  %fill_val = arith.constant -1 : i32
  %c0 = arith.constant 0 : index
  %element_count = arith.constant 2 : index
  %src_buff_size = arith.constant 8 : index
  %dst_buff_size = arith.constant 16 : index
  %output = stream.tensor.splat %fill_val :
    i32 -> tensor<2x2xi32> in !stream.resource<external>{%dst_buff_size}
  %channel = stream.channel.default : !stream.channel
  %output2 = stream.async.collective<all_gather : f32>[%element_count]
      on(#hal.affinity.queue<[0]>) channel(%channel)
      %input[%c0 to %src_buff_size for %src_buff_size],
      %output[%c0 to %dst_buff_size for %dst_buff_size] :
      !stream.resource<constant>{%src_buff_size} ->
      %output as !stream.resource<external>{%dst_buff_size}
  %result = stream.tensor.export %output2 :
    tensor<2x2xi32> in !stream.resource<external>{%dst_buff_size} -> !hal.buffer_view
  return %result : !hal.buffer_view
}
