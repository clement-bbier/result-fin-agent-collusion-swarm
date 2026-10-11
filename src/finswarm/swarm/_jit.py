"""Optional numba compilation.

Set ``FINSWARM_NO_NUMBA=1`` to run every kernel as plain Python/numpy (same code, same
results up to libm rounding). Kernels are ``@njit(cache=True, nogil=True)`` with pure array
signatures; they never draw random numbers themselves.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

NUMBA_ENABLED: bool = os.environ.get("FINSWARM_NO_NUMBA", "").strip().lower() in ("", "0", "false")

if NUMBA_ENABLED:
    from numba import njit as _numba_njit

    def jit(fn: Callable[..., Any]) -> Callable[..., Any]:
        return _numba_njit(cache=True, nogil=True)(fn)

else:

    def jit(fn: Callable[..., Any]) -> Callable[..., Any]:
        return fn
