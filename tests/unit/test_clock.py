import numpy as np
import pytest

from finswarm.config.schema import ClockConfig
from finswarm.core.clock import EpochClock
from finswarm.core.rng import RNGRegistry
from finswarm.sim.integration import (
    CLOCK_SETTABLE,
    UnsupportedIntervention,
    check_clock_param,
    speed_bump_jitter,
)


def test_sync_schedule() -> None:
    clk = EpochClock(ClockConfig(mode="sync", epoch_ticks=5), 3, RNGRegistry(0), start_tick=10)
    eps = clk.epochs_until(30)
    assert [e.tick for e in eps] == [10, 15, 20, 25]
    assert all(e.slots == (0, 1, 2) for e in eps)
    assert [e.index for e in eps] == [0, 1, 2, 3]
    assert all(e.decision_ticks == (e.tick,) * 3 for e in eps)


def test_sync_realigns_at_every_period_end() -> None:
    """ceil(T / E) epochs per period of T ticks, the last one shorter (runner semantics)."""
    clk = EpochClock(ClockConfig(mode="sync", epoch_ticks=3), 2, RNGRegistry(0))
    ticks = []
    for p in range(3):
        while (b := clk.next_batch(10 * (p + 1))) is not None:
            ticks.append(b.tick)
    assert ticks == [0, 3, 6, 9, 10, 13, 16, 19, 20, 23, 26, 29]
    empty = EpochClock(ClockConfig(mode="sync", epoch_ticks=4), 0, RNGRegistry(0))
    assert [b.tick for b in iter(lambda: empty.next_batch(10), None)] == [0, 4, 8]


def test_poisson_gaps_are_geometric_with_mean_epoch_ticks() -> None:
    clk = EpochClock(ClockConfig(mode="poisson", epoch_ticks=4), 2, RNGRegistry(1))
    times: dict[int, list[int]] = {0: [], 1: []}
    for e in clk.epochs_until(40_000):
        for s in e.slots:
            times[s].append(e.tick)
    for s in (0, 1):
        assert times[s][0] == 0  # initial quote at the start tick
        gaps = np.diff(times[s])
        assert gaps.min() >= 1
        assert gaps.mean() == pytest.approx(4.0, rel=0.03)
    one = EpochClock(ClockConfig(mode="poisson", epoch_ticks=1), 2, RNGRegistry(1))
    assert [e.tick for e in one.epochs_until(5)] == [0, 1, 2, 3, 4]


def test_building_never_draws_and_state_round_trips() -> None:
    rng = RNGRegistry(2)
    before = rng.capture()
    clk = EpochClock(ClockConfig(mode="poisson", epoch_ticks=3, speed_bump_max=2), 3, rng)
    assert rng.capture() == before
    clk.epochs_until(50)
    clk.set_jitter(4)
    clk.apply_tick(60, 1)
    snap_rng, snap = rng.capture(), clk.state()
    a = [(e.tick, e.slots, e.decision_ticks) for e in clk.epochs_until(400)]
    a_bump = [clk.apply_tick(t, t % 3) for t in range(400, 420)]
    other = EpochClock(ClockConfig(), 0, rng)  # restore overrides config and slot count
    rng.restore(snap_rng)
    other.restore(snap)
    assert other.cfg == clk.cfg and other.n_slots == 3
    b = [(e.tick, e.slots, e.decision_ticks) for e in other.epochs_until(400)]
    assert a == b and a_bump == [other.apply_tick(t, t % 3) for t in range(400, 420)]


def test_staleness_window_batches_slots_on_one_observation() -> None:
    cfg = ClockConfig(mode="poisson", epoch_ticks=6, staleness_window=4)
    clk = EpochClock(cfg, 5, RNGRegistry(3))
    clk.next_batch()  # initial quotes, everyone at tick 0
    seen_multi = False
    for _ in range(300):
        b = clk.next_batch(end_tick=10**9)
        assert all(b.tick <= d < b.tick + 4 for d in b.decision_ticks)
        assert len(set(b.slots)) == len(b.slots)
        seen_multi |= len(set(b.decision_ticks)) > 1
    assert seen_multi
    cut = EpochClock(cfg, 5, RNGRegistry(3))
    cut.next_batch()
    for _ in range(50):  # the window never crosses the period end
        b = cut.next_batch(end_tick=cut.peek() + 2)
        assert all(d < b.tick + 2 for d in b.decision_ticks)


def test_speed_bump_is_a_fifo_constant_delay() -> None:
    rng = RNGRegistry(4)
    before = rng.capture()
    clk = EpochClock(ClockConfig(speed_bump_max=3), 2, rng)
    assert [clk.apply_tick(t, 0) for t in (0, 1, 5)] == [3, 4, 8]
    assert clk.delay(1) == 3 and rng.capture() == before  # no draw without jitter
    clk.reconfigure(ClockConfig(speed_bump_max=0), 6)
    assert clk.apply_tick(6, 0) == 8  # floored at the previous apply tick: FIFO
    assert clk.apply_tick(9, 0) == 9
    assert clk.apply_tick(7, 1) == 7  # FIFO is per slot


def test_jitter_is_per_slot_constant_and_order_preserving() -> None:
    clk = EpochClock(ClockConfig(speed_bump_max=2), 40, RNGRegistry(5))
    clk.set_jitter(3)
    delays = [clk.delay(s) for s in range(40)]
    assert set(delays) == {2, 3, 4, 5}
    for s in range(40):
        ats = [clk.apply_tick(t, s) for t in range(0, 30, 2)]
        assert ats == [t + delays[s] for t in range(0, 30, 2)]
    slot = clk.add_slot(30)
    assert 2 <= clk.delay(slot) <= 5
    assert speed_bump_jitter(None) == 0 and speed_bump_jitter("3") == 3
    with pytest.raises(UnsupportedIntervention):
        speed_bump_jitter("-1")


def test_speed_bump_jitter_field_drives_the_clock() -> None:
    """``ClockConfig.speed_bump_jitter`` (schema field) replaces the ``tags`` jitter."""
    from finswarm.core.clock import jitter_of

    assert jitter_of(ClockConfig()) == 0
    cfg = ClockConfig(speed_bump_max=2, speed_bump_jitter=3)
    assert jitter_of(cfg) == 3
    rng = RNGRegistry(9)
    clk = EpochClock(ClockConfig(speed_bump_max=2), 40, rng)  # building never draws
    before = rng.capture()
    clk.reconfigure(cfg, 100)  # the field changed: the per-slot offsets are drawn now
    assert rng.capture() != before and clk.jitter_max == 3
    assert set(clk.delay(s) for s in range(40)) == {2, 3, 4, 5}
    snap = rng.capture()
    clk.reconfigure(ClockConfig(speed_bump_max=4, speed_bump_jitter=3), 200)
    assert rng.capture() == snap  # same jitter: no re-draw, the bump alone changes
    assert set(clk.delay(s) for s in range(40)) == {4, 5, 6, 7}
    assert check_clock_param("clock.speed_bump_jitter", 2, ClockConfig()).speed_bump_jitter == 2
    with pytest.raises(UnsupportedIntervention):
        check_clock_param("clock.speed_bump_jitter", -1, ClockConfig())
    with pytest.raises(ValueError):
        ClockConfig(speed_bump_jitter=-1)


def test_reconfigure_semantics() -> None:
    rng = RNGRegistry(6)
    clk = EpochClock(ClockConfig(mode="sync", epoch_ticks=10), 3, rng)
    clk.next_batch(100)
    snap = rng.capture()
    clk.reconfigure(ClockConfig(mode="sync", epoch_ticks=10, speed_bump_max=5), 100)
    assert rng.capture() == snap and clk.peek() == 10  # bump only: schedule untouched
    clk.reconfigure(ClockConfig(mode="poisson", epoch_ticks=10), 100)
    assert rng.capture() != snap and clk.next_ticks.min() >= 100  # memoryless restart
    clk.reconfigure(ClockConfig(mode="sync", epoch_ticks=4), 200)
    assert clk.peek() == 200 and list(clk.next_ticks) == [200, 200, 200]
    assert clk.add_slot(200) == 3 and clk.next_ticks[3] == 200


def test_settable_clock_paths_are_validated() -> None:
    cur = ClockConfig()
    assert set(CLOCK_SETTABLE) == {
        "clock.mode",
        "clock.epoch_ticks",
        "clock.speed_bump_max",
        "clock.staleness_window",
        "clock.speed_bump_jitter",  # live once ClockConfig carries the field (schema diff to lead)
    }
    assert check_clock_param("clock.mode", "poisson", cur).mode == "poisson"
    assert check_clock_param("clock.speed_bump_max", 7, cur).speed_bump_max == 7
    for path, value in (
        ("clock.mode", "batch"),
        ("clock.epoch_ticks", 0),
        ("clock.speed_bump_max", -1),
        ("clock.staleness_window", -2),
        ("clock.foo", 1),
    ):
        with pytest.raises(UnsupportedIntervention):
            check_clock_param(path, value, cur)
