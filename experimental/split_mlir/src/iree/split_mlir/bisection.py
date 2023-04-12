# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Tuple, Optional
from numbers import Integral

Piece = Any
AttemptCount = Integral
Bisect = Callable[[Piece, AttemptCount], Optional[Tuple[Piece, Piece]]]
HasProperty = Callable[[Piece], bool]

def bisection_search(
        object: Piece,
        bisect: Bisect,
        has_property: HasProperty) -> Piece:
    """Searches with the bisection search alogirithm for a small piece that has a property."""
    attempts = 1
    while True:
        pieces = bisect(object, attempts)
        if pieces is None:
            return object
        for piece in pieces:
            if has_property(piece):
                return bisection_search(piece, bisect, has_property)
        attempts += 1
