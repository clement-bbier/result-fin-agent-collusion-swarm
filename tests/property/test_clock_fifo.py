"""Speed bump invariants: orders never apply before their decision tick and never overtake an
earlier order of the same slot, whatever the clock, delays, jitter and mid-run changes."""

from hypothesis import given, settings
from hypothesis import strategies as st

from finswarm.config.schema import ClockConfig
from finswarm.core.clock import EpochClock
from finswarm.core.rng import RNGRegistry

changes = st.lists(
    st.tuples(
        st.sampled_from(["sync", "poisson"]),
        st.integers(1, 6),  # epoch_ticks
        st.integers(0, 12),  # speed_bump_max
        st.integers(0, 4),  # staleness_window
        st.integers(0, 3),  # jitter
    ),
    min_size=1,
    max_size=4,
)


@settings(max_examples=60, deadline=None)
@given(changes=changes, n_slots=st.integers(1, 4), seed=st.integers(0, 2**16))
def test_apply_ticks_are_fifo_and_never_early(changes, n_slots, seed) -> None:
    mode, e, d, w, j = changes[0]
    clk = EpochClock(
        ClockConfig(mode=mode, epoch_ticks=e, speed_bump_max=d, staleness_window=w),
        n_slots,
        RNGRegistry(seed),
    )
    clk.set_jitter(j)
    last: dict[int, int] = {}
    last_decision: dict[int, int] = {}
    period, tpp = 0, 10
    for k, (mode, e, d, w, j) in enumerate(changes):
        if k:  # a lever applied at a period boundary
            clk.reconfigure(
                ClockConfig(mode=mode, epoch_ticks=e, speed_bump_max=d, staleness_window=w),
                period * tpp,
            )
            clk.set_jitter(j)
        for _ in range(3):
            end = (period + 1) * tpp
            while (b := clk.next_batch(end)) is not None:
                assert period * tpp <= b.tick < end
                for s in b.slots:
                    dt = b.decision_tick(s)
                    assert dt >= last_decision.get(s, -1) + 1  # one decision per slot per tick
                    at = clk.apply_tick(dt, s)
                    assert at >= dt
                    assert at >= last.get(s, at)
                    last[s], last_decision[s] = at, dt
            period += 1
