# 'ccl' Dialect

## Lowering to a CCL backend
The CCL dialect is an abstraction and can be lowered to
specific runtime implementations like OpenMPI, NCCL, oneCCL, etc.

### Ensuring continuous memory
Most collective communication libraries have no notion of stride and
work with continuous chunks of memory.
The CCL dialect operates on ranked tensors and memrefs.
These tensors have to be first copied into new values if they are not continous.

### Lowering to specific CCL library function calls
```mlir
func.func @my_allgather(
    %in_memref : memref<?xf32>,
    %out_memref : memref<?x?xf32>,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain)
    -> (memref<?x?xf32>, !ccl.chain) {
  %res, %chain_out = ccl.allgather
      %in_memref, %out_memref, %communicator, %chain :
      (memref<?xf32>, memref<?x?xf32>, !ccl.communicator, !ccl.chain)
      -> (memref<?x?xf32>, !ccl.chain)
  return %res, %chain_out : memref<?x?xf32>, !ccl.chain
}
```

The above MLIR gets lowered to calls to external functions.
These functions are provided by the runtime implementation.
They are mostly thin wrappers around the actual library underneath.

Example when lowering to OpenMPI.
```mlir

func.func private @ccl_open_mpi_allgather(
    memref<*xf32>, memref<*xf32>, !ccl.communicator, !ccl.chain)
    -> (memref<*xf32>, !ccl.chain) attributes { llvm.emit_c_interface }
func.func @my_allgather(
    %in_memref : memref<?xf32>,
    %out_memref : memref<?x?xf32>,
    %communicator : !ccl.communicator,
    %chain : !ccl.chain)
    -> (memref<?xf32>, !ccl.chain) {
  %unraked_in_memref = memref.cast %in_memref : memref<?xf32> to memref<*xf32>
  %unraked_out_memref = memref.cast %in_memref : memref<?x?xf32> to memref<*xf32>
  %res_unrakned_memref, %chain_out = func.call @ccl_open_mpi_allgather
      %unraked_in_memref, %unraked_out_memref, %communicator, %chain :
      (memref<*xf32>, memref<*xf32>, !ccl.communicator, !ccl.chain)
      -> (memref<*xf32>, !ccl.chain)
  %res_memref = memref.cast %res_unrakned_memref : memref<*xf32> to memref<?x?xf32>
  return %res_memref, %chain_out : memref<?x?xf32>, !ccl.chain
}
```

### Lowering to LLVM dialect
After converting to CCL backend specific function calls the IR is ready to be converted to the LLVM dialect.

The special types like `!ccl.communicator` are converted to LLVM pointers.
```mlir
llvm.func @ccl_open_mpi_allgather(
    memref<*xf32>, memref<*xf32>, !llvm.ptr<i8>, !llvm.ptr<i8>)
    -> (memref<*xf32>, !llvm.ptr<i8>) attributes {sym_visibility = "private"}
```
