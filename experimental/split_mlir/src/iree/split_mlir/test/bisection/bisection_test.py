# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
from iree.split_mlir import bisection_search
from typing import List, Optional, Tuple
import numpy as np

class BisectionTest(unittest.TestCase):

  def test_bisection_search(self):
    """In a range of numbers from [0, 100) find a subrange
    that contains 42 and has 5 elements."""
    np.random.seed(12345)

    def has_property(x: List[int]) -> bool:
      return 42 in x and len(x) >= 5

    def bisect(x: List[int], attempt: int) -> Optional[Tuple[List[int], List[int]]]:
      if attempt > 99:
        return None
      cut_index = np.random.randint(len(x))
      return x[:cut_index], x[cut_index:]

    piece = bisection_search(object=np.arange(100, dtype=int), bisect=bisect, has_property=has_property)
    assert len(piece) == 5
    assert 42 in piece

if __name__ == "__main__":
  unittest.main()
