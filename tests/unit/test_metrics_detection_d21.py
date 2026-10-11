"""D-21: ghost windows netted against a cyclic-shift permutation null, the placebo-based
"punished" flag, the LLM-only H9 decision set and seed-group bootstrap clusters."""

import math

import numpy as np
import polars as pl
import pytest
from _metrics_synth import agent_period, colluding_prices, competitive_prices

from finswarm.metrics import detection as DT
from finswarm.metrics.benchmarks import logit_best_response, logit_closed_form


def _ghost_window(seed: int = 3, t: int = 50) -> np.ndarray:
    """A live dealer (column 0) that cannot influence two replayed rivals (columns 1-2):
    the rivals follow each other, the live dealer is independent noise."""
    rng = np.random.default_rng(seed)
    lead = 1.80 + np.cumsum(rng.normal(0, 0.01, t))
    follow = np.r_[lead[0], lead[:-1]] + rng.normal(0, 0.002, t)
    live = 1.70 + rng.normal(0, 0.03, t)
    return np.column_stack([live, lead, follow])


def test_permutation_null_nets_the_live_dealer_only() -> None:
    b, br = logit_closed_form(n=3), logit_best_response()
    frame = agent_period(_ghost_window())
    kw = {"n_surrogates": 10, "n_boot": 20, "d7_n_perm": 0}
    members = (100,)  # the live slot's agent id (N_SWARM + 0)
    dep = (100, 101, 102)
    raw = DT.window_features(frame, b, br, members=members, dep_members=dep, **kw)
    net = DT.window_features(
        frame, b, br, members=members, dep_members=dep, permutation_null=True, n_shifts=8, **kw
    )
    assert math.isfinite(net["D4net"]) and math.isfinite(net["D5net"])
    # the rival-rival coupling survives the shift, so the null removes most of the raw TE
    assert net["D5net"] < raw["D5net"]
    assert net["D1"] == raw["D1"]  # D1 is the live dealer's capped price index, unchanged
    # a frame with one series has no pair statistic to net
    one = DT.window_features(frame.filter(pl.col("agent_id") == 100), b, br, **kw)
    assert math.isnan(one["D4net"]) or one["D4net"] == 0.0


def test_shift_null_draws_and_the_ghost_null_of_a_sham_window() -> None:
    b, br = logit_closed_form(n=3), logit_best_response()
    frame = agent_period(_ghost_window(t=30))
    members, dep = (100,), (100, 101, 102)
    kw = {"n_surrogates": 5, "seed": 0}
    fk = {"n_surrogates": 5, "n_boot": 10, "d7_n_perm": 0}
    draws = DT.shift_null_draws(frame, members=members, dep_members=dep, n_shifts=6, **kw)
    assert draws.shape == (6,) and np.isfinite(draws).all()
    # the mean of the draws is exactly the null that permutation_null subtracts
    raw = DT.window_features(frame, b, br, members=members, dep_members=dep, **fk)
    net = DT.window_features(
        frame, b, br, members=members, dep_members=dep, permutation_null=True, n_shifts=6, **fk
    )
    assert raw["D5net"] - net["D5net"] == pytest.approx(float(np.mean(draws)))
    # every non-zero shift without a cap: 30 periods -> 29 differenced rows -> 28 shifts
    assert DT.shift_null_draws(frame, members=members, dep_members=dep, **kw).shape == (28,)
    # netted against a ghost set, the draws move to that window's D5-net scale
    netted = DT.shift_null_draws(
        frame, ghost=[frame], members=members, dep_members=dep, n_shifts=6, **kw
    )
    assert netted == pytest.approx(draws - raw["D5net"])
    assert DT.shift_null_draws(frame.filter(pl.col("agent_id") == 100), **kw).size == 0
    with pytest.raises(ValueError, match="empty ghost set"):
        DT.shift_null_draws(frame, ghost=[], **kw)
    # the ghost null of a sham window: each ghost shifted against ITS live dealer, every draw
    # net of the mean TE of the ghost set (the sham's own netting term), pooled in order
    g1 = agent_period(_ghost_window(seed=4, t=30)[:, [1, 0, 2]])  # live dealer = agent 101
    gd = DT.ghost_shift_null_draws(
        [frame, g1], [100, 101], dep, key="agent_id", n_surrogates=5, n_shifts=6
    )
    te = [DT.window_features(g, b, br, members=dep, **fk)["D5net"] for g in (frame, g1)]
    own = [
        DT.shift_null_draws(g, members=dep, dep_members=dep, live=x, n_shifts=6, **kw)
        for g, x in ((frame, 100), (g1, 101))
    ]
    assert gd.shape == (12,) and gd == pytest.approx(np.concatenate(own) - np.mean(te))
    with pytest.raises(ValueError, match="live slots"):
        DT.ghost_shift_null_draws([frame], [100, 101], dep, key="agent_id")
    with pytest.raises(ValueError, match="empty ghost set"):
        DT.ghost_shift_null_draws([], [], dep, key="agent_id")


def test_ghost_spec_defaults_to_the_permutation_null_and_d1_is_capped_price() -> None:
    b, br = logit_closed_form(), logit_best_response()
    col = agent_period(colluding_prices(60, seed=2))
    spec = DT.WindowSpec.by_process("g", "r", "T", "ghost", col, members=(100,), seed_group="7")
    assert spec.permutation_null is True and spec.label == 0
    spec2 = DT.WindowSpec.by_process("g", "r", "T", "ghost", col, ghost=col)
    assert spec2.permutation_null is False
    kw = {"n_surrogates": 10, "n_boot": 20, "d7_n_perm": 0, "n_shifts": 5}
    corpus = DT.build_corpus([spec], b, br, **kw)
    assert corpus["seed_group"][0] == "7" and corpus["window_length"][0] == 60
    assert corpus["D1"][0] <= 1.0
    # D1 is the capped PRICE index: a market far above monopoly reads exactly 1
    high = agent_period(np.full((40, 2), 3.0))
    feats = DT.window_features(high, b, br, **{"n_surrogates": 5, "n_boot": 10, "d7_n_perm": 0})
    assert feats["D1"] == 1.0
    comp = agent_period(competitive_prices(60, seed=1))
    fn = DT.window_features(comp, b, br, **{"n_surrogates": 5, "n_boot": 10, "d7_n_perm": 0})
    assert abs(fn["D1"]) < 0.2


def test_placebo_punished_flag_and_llm_sham_kind() -> None:
    row = 20  # matrix row of the deviation; agent_period periods start at 1
    t0 = row + 1
    sham = agent_period(np.full((40, 3), 1.90))
    # sham2: no deviation, rivals drift by small placebo cuts
    s2_placebo = np.full((40, 3), 1.90)
    s2_placebo[row + 1 : row + 3, 1:] -= 0.02
    sham2 = agent_period(s2_placebo)
    placebo = [DT.placebo_cut(sham2, sham, deviator=100, t0=t0, key="agent_id")]
    assert placebo[0] == pytest.approx(0.02)
    pc = [0.01, 0.02, 0.015, 0.03, 0.02, 0.01]
    assert DT.punished_with_placebo(0.20, pc)["punished"] is True
    assert DT.punished_with_placebo(0.02, pc)["punished"] is False
    # an empty placebo set has no fallback rule: it raises (M6a)
    for empty in ([], [float("nan")]):
        with pytest.raises(ValueError, match="empty placebo set"):
            DT.punished_with_placebo(0.5, empty)
    assert "n <= 12" in DT.punished_with_placebo(0.2, pc)["note"]
    # S2 probe: slot 0 cuts by 0.3 at t0, rivals cut by 0.25 within 3 periods -> punished
    probe = np.full((40, 3), 1.90)
    probe[row, 0] -= 0.30
    probe[row + 1 : row + 4, 1:] -= 0.25
    s2 = agent_period(probe)
    assert DT.llm_sham_kind(s2, sham, 100, t0, key="agent_id", placebo_cuts=pc) == "llm_punished"
    # a tiny rival reaction below the placebo 95th percentile is not punishment
    weak = np.full((40, 3), 1.90)
    weak[row, 0] -= 0.30
    weak[row + 1, 1:] -= 0.01
    assert DT.llm_sham_kind(agent_period(weak), sham, 100, t0, key="agent_id", placebo_cuts=pc) == (
        "llm_unpunished"
    )
    # v1 criterion, explicitly asked for (placebo_cuts=None): 0.25 >= 0.5 x 0.30 -> punished
    assert DT.llm_sham_kind(s2, sham, 100, t0, key="agent_id") == "llm_punished"
    # an empty placebo list is not the v1 rule: it raises
    with pytest.raises(ValueError, match="empty placebo set"):
        DT.llm_sham_kind(s2, sham, 100, t0, key="agent_id", placebo_cuts=[])
    # the placebo quantile is threaded: a 0.035 cut beats the median, not the 95th percentile
    mid = np.full((40, 3), 1.90)
    mid[row, 0] -= 0.30
    mid[row + 1, 1:] -= 0.035
    pc5 = [0.01, 0.02, 0.03, 0.04, 0.05]
    kw = {"key": "agent_id", "placebo_cuts": pc5}
    assert DT.llm_sham_kind(agent_period(mid), sham, 100, t0, **kw) == "llm_unpunished"
    assert (
        DT.llm_sham_kind(agent_period(mid), sham, 100, t0, placebo_quantile=0.5, **kw)
        == "llm_punished"
    )


def test_seed_group_id_is_lane_prefixed_and_refuses_null() -> None:
    assert DT.seed_group_id("H", 3) == "H:3" and DT.seed_group_id("M", 3) == "M:3"
    assert DT.seed_group_id("H", 3.0) == "H:3"  # JSON floats of an integer seed
    assert DT.seed_group_id("H", 3) != DT.seed_group_id("M", 3)  # transfer lane apart
    with pytest.raises(ValueError, match="seed"):
        DT.seed_group_id("H", None)
    with pytest.raises(ValueError, match="lane"):
        DT.seed_group_id(None, 3)


def test_null_seed_groups_are_refused_by_split_and_evaluate() -> None:
    corpus = _corpus_with_lanes()
    holed = corpus.with_columns(
        pl.when(pl.col("window_id") == "w3")
        .then(None)
        .otherwise(pl.col("seed_group"))
        .alias("seed_group")
    )
    with pytest.raises(ValueError, match="without a seed_group"):
        DT.split_half(holed)
    with pytest.raises(ValueError, match="no seed_group column"):
        DT.split_half(corpus.drop("seed_group"))
    # an explicit trunk split stays available
    dev, ev = DT.split_half(holed, group="trunk_id")
    assert set(dev["trunk_id"]).isdisjoint(set(ev["trunk_id"]))
    rules = DT.select_rules(corpus)
    with pytest.raises(ValueError, match="without a seed_group"):
        DT.evaluate(holed, rules, n_boot=5)


def test_netting_is_required_for_llm_sham_ghost_and_ca_windows() -> None:
    b, br = logit_closed_form(n=3), logit_best_response()
    frame = agent_period(_ghost_window())
    kw = {"n_surrogates": 5, "n_boot": 10, "d7_n_perm": 0, "n_shifts": 4}
    # an empty ghost set cannot net a window (None = no ghost netting asked)
    with pytest.raises(ValueError, match="empty ghost set"):
        DT.window_features(frame, b, br, ghost=[], **kw)
    # an LLM sham window without ghosts is refused, naming its trunk (M6d)
    for kind in DT.GHOST_NETTED_KINDS:
        bare = DT.WindowSpec.by_process("w", "r", "T7", kind, frame, seed_group="H:7")
        with pytest.raises(ValueError, match="trunk T7"):
            DT.build_corpus([bare], b, br, **kw)
    # a ghost window built without the permutation null is refused
    raw_ghost = DT.WindowSpec.by_process("g", "r", "T7", "ghost", frame, permutation_null=False)
    with pytest.raises(ValueError, match="cyclic-shift"):
        DT.build_corpus([raw_ghost], b, br, **kw)
    # CA (instruction) windows default to the cyclic-shift null, like ghosts (M6c)
    ca = DT.WindowSpec.by_process("c", "r", "C0", "instruction", frame, seed_group="H:500")
    assert ca.permutation_null is True and ca.label == 0
    row = DT.build_corpus([ca], b, br, **kw).row(0, named=True)
    net = DT.window_features(frame, b, br, permutation_null=True, **kw)
    raw = DT.window_features(frame, b, br, **kw)
    assert row["D5net"] == pytest.approx(net["D5net"]) and row["D4net"] == pytest.approx(
        net["D4net"]
    )
    assert row["D5net"] != pytest.approx(raw["D5net"])  # the null was subtracted


def _corpus_with_lanes(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    plan = [
        ("llm_punished", 6, "H", 1),
        ("ghost", 10, "H", 0),
        ("instruction", 4, "H", 0),
        ("sc3", 6, "CPU", 1),
        ("sc1", 6, "CPU", 0),
        ("sc2_common_shock", 6, "CPU", 0),
        ("llm_punished", 4, "M", 1),
        ("ghost", 6, "M", 0),
        ("llm_unpunished", 2, "H", None),
    ]
    k = 0
    for kind, n, lane, label in plan:
        for _i in range(n):
            pos = label == 1
            rows.append(
                {
                    "window_id": f"w{k}",
                    "run_id": f"run{k}",
                    "trunk_id": f"T{k % 9}",
                    "label": label,
                    "kind": kind,
                    "D1": rng.normal(0.7 if pos else 0.3, 0.1),
                    "D2": abs(rng.normal(0.02, 0.01)),
                    "D3": abs(rng.normal(0.2, 0.1)),
                    "D4net": rng.normal(0.1, 0.1),
                    "D5net": rng.normal(0.15 if pos else 0.0, 0.03),
                    "D6": rng.normal(0.0, 0.05),
                    "D7": rng.normal(0.4 if pos else 0.05, 0.06),
                    "D8": float("nan"),
                    "D9": float("nan"),
                    "D7_cartel": 0.62,
                    "D7_ratio": float("nan"),
                    "D7_z": float("nan"),
                    "D7_p": float("nan"),
                    "seed_group": str(k % 9),
                    "outcome_label": None,
                    "lane": lane,
                    "window_length": 40 if kind != "instruction" else 30,
                }
            )
            k += 1
    return pl.DataFrame(rows, schema=DT.CORPUS_SCHEMA)


def test_decision_reports_llm_only_scripted_validity_and_transfer(tmp_path) -> None:
    corpus = _corpus_with_lanes()
    dev, ev = DT.split_half(corpus)
    assert set(dev["seed_group"]).isdisjoint(set(ev["seed_group"]))
    rules = DT.select_rules(dev)
    rep = DT.evaluate(ev, rules, n_boot=100)
    assert rep["cluster"] == "seed_group" and rep["n_clusters"] <= rep["n_runs"]
    assert set(rep["composite"]["fpr_by_window_length"]) <= {"30", "40"}
    dec = DT.decision_reports(ev, rules, n_boot=100)
    llm = dec["llm_only"]
    assert llm["status"] == "ok" and llm["lane"] == "H"
    assert set(llm["composite"]["fpr_by_type"]) <= {"ghost", "instruction"}
    assert llm["n_pos"] > 0 and llm["composite"]["auc"] > 0.8
    sv = dec["scripted_validity"]
    assert sv["status"] == "ok" and set(sv["composite"]["fpr_by_type"]) <= {
        "sc1",
        "sc2_common_shock",
    }
    tr = dec["transfer"]
    assert tr["lane"] == "M" and tr["status"] in (
        "ok",
        "insufficient windows (need positives and negatives)",
    )
    out = DT.run_pipeline(corpus, rules_path=tmp_path / "rules.json", n_boot=50)
    assert "decision" in out and out["decision"]["llm_only"]["positives"] == ["llm_punished"]
