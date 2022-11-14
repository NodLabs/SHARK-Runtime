// RUN: mpirun -n 1 iree-run-mlir --iree-hal-target-backends=opencl-spirv %s --device=level_zero

//      CHECK: EXEC @main
// CHECK-NEXT: result[0]: i64=0
// CHECK-NEXT: result[1]: i64=1
func.func @main() -> (index, index) {
  %channel = stream.channel.create on(#hal.affinity.queue<[0, 1]>) : !stream.channel
  %rank = stream.channel.rank %channel : index
  %count = stream.channel.count %channel : index
  return %rank, %count : index, index
}
