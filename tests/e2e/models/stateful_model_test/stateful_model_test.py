# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.runtime import load_vm_flatbuffer_file
from iree.compiler.tools import compile_file, InputType
import os
import sys
import tempfile
import numpy as np
import unittest
import argparse
from typing import List, TypeVar
from urllib.request import urlretrieve
import tarfile

Tensor = TypeVar('Tensor')
args = None


def build_module(artifacts_dir: str):
  vmfb_file = os.path.join(artifacts_dir, "stateful_model.vmfb")
  compile_file(input_file=os.path.join(
      artifacts_dir,
      "tests/e2e/models/stateful_model_test/stateful_model.mlirbc"),
               output_file=vmfb_file,
               target_backends=[args.target_backend],
               input_type=InputType.MHLO)
  return load_vm_flatbuffer_file(vmfb_file, driver=args.driver)


def download_test_data(filepath: str) -> str:
  return urlretrieve(
      "https://storage.googleapis.com/shark-public/boian/stateful_model_test_data.tar",
      filename=filepath)[0]


def extract_test_data(archive_path: str, out_dir: str):
  with tarfile.open(archive_path) as tar:
    tar.extractall(out_dir)


def load_data(data_dir: str):
  with np.load(os.path.join(data_dir, "forward_input.npz")) as data:
    forward_input = list(data.values())[0]
  with np.load(os.path.join(data_dir, "expected_forward_output.npz")) as data:
    expected_forward_output = list(data.values())[0]
  with np.load(os.path.join(data_dir, "params.npz")) as data:
    params = list(data.values())

  return (forward_input, expected_forward_output, params)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--test_data_filepath",
                      type=str,
                      help="If not specified will download it.",
                      default=None)
  parser.add_argument("--target_backend", type=str, default="llvm-cpu")
  parser.add_argument("--driver", type=str, default="local-task")
  return parser.parse_known_args()


DEFAULT_REL_TOLERANCE = 1e-3
DEFAULT_ABS_TOLERANCE = 1e-5


def allclose(a: Tensor,
             b: Tensor,
             rtol=DEFAULT_REL_TOLERANCE,
             atol=DEFAULT_ABS_TOLERANCE):
  return np.allclose(np.asarray(a), np.asarray(b), rtol, atol)


def assert_array_list_compare(array_compare_fn, a: Tensor, b: Tensor):
  assert (len(a) == len(b))
  for x, y in zip(a, b):
    np.testing.assert_array_compare(array_compare_fn, x, y)


def assert_array_list_allclose(a: List[Tensor],
                               b: List[Tensor],
                               rtol=DEFAULT_REL_TOLERANCE,
                               atol=DEFAULT_ABS_TOLERANCE):
  assert_array_list_compare(lambda x, y: allclose(x, y, rtol, atol), a, b)


class StatefulModelTest(unittest.TestCase):

  def test_stateful_model(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      test_data_filepath = download_test_data(
          os.path.join(tmp_dir, "stateful_model_test_data.tar")
      ) if args.test_data_filepath is None else args.test_data_filepath
      extract_test_data(test_data_filepath, tmp_dir)
      module = build_module(tmp_dir)
      (forward_input, expected_forward_output, params) = load_data(
          os.path.join(tmp_dir, "tests/e2e/models/stateful_model_test"))
    for i in range(10):
      module.set_params(*params)
      forward_output = np.asarray(module.forward(forward_input))
      assert_array_list_allclose(forward_output, expected_forward_output)


if __name__ == '__main__':
  args, remaining_args = parse_args()
  unittest.main(argv=[sys.argv[0]] + remaining_args)
