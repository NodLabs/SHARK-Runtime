#!/bin/bash

set -xeu

if (( $# != 2 )); then
  echo "usage: $0 <target-chip> <input-mlir>"
  exit 1
fi

iree-compile $2 --iree-preprocessing-pass-pipeline="builtin.module(iree-preprocessing-transpose-convolution-pipeline, iree-global-opt-raise-special-ops, util.func(iree-preprocessing-pad-to-intrinsics))"   --iree-codegen-llvmgpu-use-vector-distribution --iree-input-type=auto --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=rocm --iree-vulkan-target-triple=rdna3-unknown-linux --iree-stream-resource-max-allocation-size=4294967296 --mlir-print-op-on-diagnostic=false --iree-input-type=torch --mlir-print-op-on-diagnostic=false --iree-llvmcpu-target-cpu-features=host --iree-llvmcpu-target-triple=x86_64-linux-gnu --iree-stream-resource-index-bits=64 --iree-vm-target-index-bits=64 --iree-rocm-target-chip=$1 --iree-vm-bytecode-module-strip-source-map=true --iree-opt-strip-assertions=true --iree-vm-target-truncate-unsupported-floats --iree-codegen-llvmgpu-enable-transform-dialect-jit=false --iree-llvmgpu-enable-prefetch --iree-codegen-transform-dialect-library=attention_$1.spec.mlir --iree-global-opt-propagate-transposes=true --iree-opt-aggressively-propagate-transposes=true --iree-opt-data-tiling=false --iree-opt-const-eval=false --iree-opt-outer-dim-concat=true --iree-flow-enable-aggressive-fusion --iree-global-opt-enable-fuse-horizontal-contractions=true --iree-opt-aggressively-propagate-transposes=true -o unet.vmfb
