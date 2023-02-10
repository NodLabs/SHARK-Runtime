# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import random
import jax.core
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
from jax.tree_util import tree_flatten
from iree.jax import (
    like,
    kernel,
    Program,
)
import numpy as np
import argparse
from typing import Callable, Any
from tempfile import TemporaryDirectory
import os
import tarfile


def set_seed(seed: int):
  random.seed(seed)
  np.random.seed(seed)


INPUT_SIZE = 28 * 28
OUPUT_SIZE = 10


def get_example_input():
  batch_size = 128
  return np.random.rand(batch_size, INPUT_SIZE).astype(np.float32)


def get_model():
  num_layers = 3
  layer_size = 256
  layers = []
  for i in range(num_layers):
    layers.append(Dense(layer_size))
    layers.append(Relu)
  layers.append(Dense(OUPUT_SIZE))
  layers.append(LogSoftmax)
  init_random_params, forward = stax.serial(*layers)
  return init_random_params, forward


class ModelContext:

  def __init__(self, model_forward_fn: Callable[[Any, Any], Any], model_params,
               example_input):
    self.model_forward_fn = model_forward_fn
    self.model_params = model_params
    self.example_input = example_input


def make_model_context(rng) -> ModelContext:
  init_random_params, forward = get_model()
  example_input = get_example_input()
  _, params = init_random_params(rng, example_input.shape)
  return ModelContext(model_forward_fn=forward,
                      model_params=params,
                      example_input=example_input)


def create_iree_jax_program(model_context: ModelContext) -> Program:

  class IreeJaxProgram(Program):
    _params = model_context.model_params

    def set_params(self, params=like(model_context.model_params)):
      self._params = params

    def forward(self, inputs=like(model_context.example_input)):
      return self._forward(self._params, inputs)

    @kernel
    def _forward(params, inputs):
      return model_context.model_forward_fn(params, inputs)

  return IreeJaxProgram()


def create_iree_jax_module(model_context: ModelContext):

  class JaxModule:
    _params = model_context.model_params

    def get_params(self):
      return self._params

    def forward(self, inputs):
      return model_context.model_forward_fn(self._params, inputs)

  return JaxModule()


def build_mlir_module(module: Program, path: str):
  with open(path, "wb") as f:
    ir_module = Program.get_mlir_module(module)
    ir_module.operation.write_bytecode(f)


def generate_test_data(output_filepath: str,):
  seed = 876543210
  set_seed(seed)
  rng = jax.random.PRNGKey(seed)
  jax_model_context = make_model_context(rng)
  iree_model_context = make_model_context(rng)
  iree_jax_program = create_iree_jax_program(iree_model_context)
  jax_module = create_iree_jax_module(jax_model_context)
  expected_forward_output = jax_module.forward(jax_model_context.example_input)
  with TemporaryDirectory() as tmp_dir:
    data_dir = os.path.join(tmp_dir, "tests/e2e/models/stateful_model_test")
    os.makedirs(data_dir)
    build_mlir_module(iree_jax_program,
                      os.path.join(data_dir, "stateful_model.mlirbc"))
    np.savez_compressed(os.path.join(data_dir, "forward_input.npz"),
                        jax_model_context.example_input)
    np.savez_compressed(os.path.join(data_dir, "params.npz"),
                        *tree_flatten(jax_model_context.model_params)[0])
    np.savez_compressed(os.path.join(data_dir, "expected_forward_output.npz"),
                        expected_forward_output)
    with tarfile.open(output_filepath, "w") as tar:
      tar.add(os.path.join(tmp_dir, "tests"), arcname="tests")


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--output_filepath",
                      help="TAR file with all test data.",
                      type=str,
                      default="stateful_model_test_data.tar")

  return parser.parse_args()


def generate_test_data_cli():
  kwargs = vars(parse_args())
  generate_test_data(**kwargs)


if __name__ == "__main__":
  generate_test_data_cli()
