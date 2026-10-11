import math

import numpy as np
import polars as pl
import pytest

from finswarm.metrics import dynamics as Y


def test_time_to_collude() -> None:
    s = np.r_[np.zeros(10), np.full(5, 0.5), np.zeros(3), np.full(30, 0.6)]
    assert Y.time_to_collude(s, thr=0.3, persist=20) == 18.0
    assert Y.time_to_collude(s, thr=0.3, persist=5) == 10.0
    assert math.isnan(Y.time_to_collude(np.zeros(50)))  # censored
    assert Y.time_to_collude(pl.Series(s), persist=20) == 18.0


def test_time_to_break_and_recover() -> None:
    after = np.r_[np.full(3, 0.6), np.full(4, 0.2), np.full(10, 0.65)]
    assert Y.time_to_break(after, baseline=0.6) == 3.0
    assert Y.time_to_recover(after, baseline=0.6, frac=0.9, persist=5) == 7.0
    assert Y.time_to_recover(after, target=0.5, after_break=False, persist=3) == 0.0
    assert math.isnan(Y.time_to_recover(np.full(10, 0.6), baseline=0.6))  # never broke


def test_hysteresis() -> None:
    on = np.full(30, 0.2)
    off_full = np.full(30, 0.6)
    off_none = np.full(30, 0.2)
    assert Y.hysteresis(on, off_full, baseline=0.6) == pytest.approx(0.0)
    assert Y.hysteresis(on, off_none, baseline=0.6) == pytest.approx(1.0)
    assert Y.hysteresis(on, np.full(30, 0.4), baseline=0.6) == pytest.approx(0.5)
    assert Y.hysteresis(on, off_full) == pytest.approx(0.4)


def test_mttr() -> None:
    s = np.r_[
        np.zeros(5),
        np.full(5, 0.5),
        np.zeros(4),
        np.full(5, 0.5),
        np.zeros(2),
        np.full(3, 0.5),
        np.zeros(3),
    ]
    assert Y.mttr(s, thr=0.3) == pytest.approx(3.0)  # episodes of 4 and 2; last censored
    assert math.isnan(Y.mttr(np.zeros(10)))
