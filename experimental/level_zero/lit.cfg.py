# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Lit config for IREE Level Zero Driver."""

# Lint for undefined variables is disabled as config is not defined inside this
# file, instead config is injected by way of evaluating runlit.cfg.py from
# runlit.site.cfg.py which in turn is evaluated by lit.py.
# pylint: disable=undefined-variable

import os
import tempfile

import lit.formats

config.name = "IREE Level Zero Driver"
config.suffixes = [".mlir"]
config.test_format = lit.formats.ShTest(execute_external=True)

# Use the most preferred temp directory.
config.test_exec_root = (os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or
                         os.environ.get("TEST_TMPDIR") or
                         os.path.join(tempfile.gettempdir(), "lit"))

# The propagation of the environment is required
# because the programs must run in a context after sourcing oneAPI's setvars.sh.
config.environment.update(os.environ)
