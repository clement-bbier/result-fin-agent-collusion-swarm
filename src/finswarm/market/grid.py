"""Price-grid arithmetic shared by the markets and the runner (re-grids, lever L3).

A grid is ``p_min + k * tick`` for ``k = 0 .. n_levels - 1`` with ``n_levels =
round((p_max - p_min) / tick) + 1``. A re-grid keeps ``p_min`` and ``p_max`` and maps every
standing price onto the new grid: **asks round UP, bids round DOWN** (to the next grid point),
so a re-grid never makes a quote more aggressive and never crosses a book. A price whose
rounded level falls outside the new grid has no image (index -1, price NaN): the quote is
withdrawn rather than moved inside. Historical prices (trades) map to the nearest level.
"""

from __future__ import annotations

import math

import numpy as np

#: slack when snapping onto a grid (grid units): exact grid points never move by rounding
GRID_EPS = 1e-9


def n_levels_for(p_min: float, p_max: float, tick: float) -> int:
    """Number of levels of the grid ``[p_min, p_max]`` with step ``tick``."""
    if not (math.isfinite(tick) and tick > 0.0):
        raise ValueError(f"tick size must be finite and > 0, got {tick!r}")
    n = int(round((p_max - p_min) / tick)) + 1
    if n < 2:
        raise ValueError(f"tick {tick} leaves fewer than 2 levels on [{p_min}, {p_max}]")
    return n


def regrid_indices(
    idx: np.ndarray, old_tick: float, new_tick: float, n_levels: int, *, up: bool
) -> np.ndarray:
    """Map old-grid levels to new-grid levels (``up``: asks, else bids); -1 stays -1 and
    levels that fall off the new grid become -1."""
    idx = np.asarray(idx, dtype=np.int64)
    x = idx * float(old_tick) / float(new_tick)
    new = np.ceil(x - GRID_EPS) if up else np.floor(x + GRID_EPS)
    out = new.astype(np.int64)
    bad = (idx < 0) | (out < 0) | (out >= n_levels)
    out[bad] = -1
    return out


def regrid_index(idx: int, old_tick: float, new_tick: float, n_levels: int, *, up: bool) -> int:
    return int(regrid_indices(np.array([idx]), old_tick, new_tick, n_levels, up=up)[0])


def nearest_indices(idx: np.ndarray, old_tick: float, new_tick: float, n_levels: int) -> np.ndarray:
    """Historical prices (trades) on the new grid: nearest level, clipped to the grid."""
    idx = np.asarray(idx, dtype=np.int64)
    out = np.rint(idx * float(old_tick) / float(new_tick)).astype(np.int64)
    out = np.clip(out, 0, n_levels - 1)
    out[idx < 0] = -1
    return out


def regrid_price(
    price: float, p_min: float, old_tick: float, new_tick: float, n_levels: int, *, up: bool
) -> float:
    """A standing price on the old grid mapped onto the new grid (NaN when it has no image).

    The price is first snapped to its old-grid level so that the result is exactly the
    market's own re-quantised quote.
    """
    if price is None or not math.isfinite(price):
        return float("nan")
    old = int(round((price - p_min) / old_tick))
    new = regrid_index(old, old_tick, new_tick, n_levels, up=up)
    return float("nan") if new < 0 else p_min + new * new_tick
