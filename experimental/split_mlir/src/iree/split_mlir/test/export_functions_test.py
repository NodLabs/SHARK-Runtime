# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import subprocess
import os
import tempfile
import sys
import argparse
import types
import copy

args = types.SimpleNamespace(path=[])


def updateEnv(env):
  for path in args.path:
    env["PATH"] += os.pathsep
    env["PATH"] += path


class ExportFunctionsTest(unittest.TestCase):

  def test_export_finctions(self):
    env = copy.deepcopy(os.environ)
    updateEnv(env)

    with tempfile.TemporaryDirectory() as temp_dir:
      mlir_file_path = os.path.join(os.path.dirname(__file__),
                                    "export_functions.mlir")

      # Run pass
      subprocess.check_call(
          [("iree-opt "
            "--split-input-file "
            "--iree-plugin=split_mlir "
            "'--pass-pipeline=builtin.module(iree-export-functions{"
            f"functions=function_to_export.+,path-prefix=\"{temp_dir}/\"}})' "
            f"\"{mlir_file_path}\" "
            f"| FileCheck \"{mlir_file_path}\"")],
          shell=True,
          cwd=temp_dir,
          env=env)

      # Check exported function MLIR file.
      function_to_export_sin_file_path = os.path.join(
          temp_dir, "function_to_export_sin.mlir")
      function_to_export_sin_check_file_path = os.path.join(
          os.path.dirname(__file__), "function_to_export_sin.filecheck")
      subprocess.check_call([
          f"cat \"{function_to_export_sin_file_path}\""
          f" | FileCheck \"{function_to_export_sin_check_file_path}\""
      ],
                            shell=True,
                            env=env)

      # Check exported function MLIR file.
      function_to_export_cos_file_path = os.path.join(
          temp_dir, "function_to_export_sin.mlir")
      function_to_export_cos_check_file_path = os.path.join(
          os.path.dirname(__file__), "function_to_export_sin.filecheck")
      subprocess.check_call([
          f"cat \"{function_to_export_cos_file_path}\""
          f" | FileCheck \"{function_to_export_cos_check_file_path}\""
      ],
                            shell=True,
                            env=env)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--path",
      type=str,
      nargs="*",
      default=args.path,
      help="List of directories to add to the PATH environment variable")
  args, remaining_args = parser.parse_known_args(args=sys.argv[1:])
  unittest.main(argv=sys.argv[:1] + remaining_args)
