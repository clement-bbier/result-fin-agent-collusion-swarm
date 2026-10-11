"""Registered-arm campaign analysis: whitelist, ghosts, pinned benchmarks, D-3' inference."""

import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from _metrics_synth import market_frame, write_frame_run

from finswarm.analysis import stats as S
from finswarm.analysis.arms import (
    PinError,
    check_lever_params,
    load_registered_arms,
    resolve_arm_benchmarks,
)
from finswarm.analysis.benchmarks_file import build_benchmark_file, dumps, solve_market
from finswarm.analysis.campaign import analyze_campaign
from finswarm.config.load import load_campaign_config

ROOT = Path(__file__).resolve().parents[2]
ARMS = ROOT / "configs" / "analysis" / "registered_arms.yaml"
SPEC = load_registered_arms(ARMS)
AB = resolve_arm_benchmarks(SPEC, ROOT)
TRUNK_M = AB["sham"].primary_model
L1_M, L2_M, L6_M = (AB[k].window_model for k in ("L1", "L2", "L6"))
TB = AB["sham"].primary
SCALE = TB.p_mono - TB.p_nash
N_TRUNKS = 8
TRH_MARKET = load_campaign_config(ROOT / "configs/sprint/tier0/TR-H.yaml").base.model_dump(
    mode="json"
)["market"]


def manifest(arm, label, trunk, seed, branch=True, market=None, **tags) -> dict:
    m = {
        "arm": arm,
        "label": label,
        "trunk_id": trunk,
        "seed": seed,
        "branch_period": 100 if branch else None,
        "resolved_config": {"tags": {"arm": arm, "label": label, **tags}},
    }
    if market is not None:
        m["resolved_config"]["market"] = market
    return m


@pytest.fixture(scope="module")
def camp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    c = tmp_path_factory.mktemp("camp") / "TR-H"
    rng = np.random.default_rng(0)
    for k in range(N_TRUNKS + 1):  # T8 does not qualify (Gate 2)
        t = f"T{k}"
        level = 1.40 if k == N_TRUNKS else 1.90
        tr = market_frame(np.full((30, 4), level), TRUNK_M, start=71)
        write_frame_run(c, f"trunk-{t}", tr, manifest("trunk", "TR-H", t, k, branch=False))
        sham = 1.85 + 0.02 * (k % 8) + rng.normal(0, 0.01, (40, 4))
        write_frame_run(
            c,
            f"sham-{t}",
            market_frame(sham, TRUNK_M),
            manifest("sham", "SH-H", t, k, market=TRH_MARKET),
        )
        l1 = sham - (0.25 + 0.01 * k)  # a behavioural cut under the new routing
        write_frame_run(c, f"L1-{t}", market_frame(l1, L1_M), manifest("L1", "L1-H", t, k))
        l2 = sham - (0.15 + 0.005 * k)
        write_frame_run(c, f"L2-{t}", market_frame(l2, L2_M), manifest("L2", "L2-H", t, k))
        l6 = market_frame(sham - 0.05, L6_M, entrant=np.full(40, 1.40))  # entrant = slot 4
        write_frame_run(c, f"L6-{t}", l6, manifest("L6", "L6-H", t, k))
    # the ghosts run under the SH-H sham as in the store (A-013): <sham run>/ghost/<slot>,
    # manifest carrying the parent's arm and label; they are the registered ghost arm and are
    # never pooled into the sham arm (only the ghost slot is analysed)
    for g in range(4):
        p = np.full((40, 4), 1.50)
        p[:, g] = 1.70  # the live slot g prices at 1.70, replayed rivals at 1.50
        write_frame_run(
            c / "sham-T1" / "ghost",
            f"sham-T1--ghost{g}",
            market_frame(p, TRUNK_M),
            manifest("sham", "SH-H", "T1", 1, ghost_of="sham-T1", ghost_slot=str(g)),
            subdir=str(g),
        )
    # a ghost labelled for the planned GH-H parent: no longer registered (listed, never pooled)
    write_frame_run(
        c / "sham-T0" / "ghost",
        "sham-T0--ghost0",
        market_frame(np.full((40, 4), 1.10), TRUNK_M),
        manifest("ghost_sham", "GH-H", "T0", 0, ghost_of="sham-T0", ghost_slot="0"),
        subdir="0",
    )
    # unregistered arm, failed decisions, missing incumbent, wrong market in the manifest
    flat = market_frame(np.full((40, 4), 1.9), TRUNK_M)
    write_frame_run(c, "weird-T0", flat, manifest("weird", "X", "T0", 0))
    l1flat = market_frame(np.full((40, 4), 1.6), L1_M)
    write_frame_run(c, "L1-bad", l1flat, manifest("L1", "L1-H", "T9", 9), fail_every=5)
    three = market_frame(np.full((40, 3), 1.6), replace(L1_M, n_incumbents=3))
    write_frame_run(c, "L1-missing", three, manifest("L1", "L1-H", "T10", 10))
    beta1 = json.loads(json.dumps(TRH_MARKET))
    beta1["demand"]["beta"] = 1.0  # the old CLI default
    write_frame_run(c, "L1-beta1", l1flat, manifest("L1", "L1-H", "T11", 11, market=beta1))
    return c


# the MQC document carries the 95 % upper bound (m11: no upper bound, no eligibility) and the
# SC1 best responders' price response (A-003: the L2 rule needs the SC1 ask fall)
MQC = {
    "levers": {
        "L1": {"status": "ok", "mean": 1.0, "bca_hi": 1.5, "sc1_price_response": -0.01},
        "L2": {"status": "ok", "mean": 3.0, "bca_hi": 4.0, "sc1_price_response": -0.02},
        "L6": {"status": "ok", "mean": 2.0, "bca_hi": 3.0, "sc1_price_response": -0.01},
    }
}


def _run(camp: Path, out: Path, mqc=MQC, **kw):
    kw = {"B": 400, "n_perm": 0, **kw}
    return analyze_campaign(camp, ARMS, ROOT, out, mqc=mqc, **kw)


@pytest.fixture(scope="module")
def analysed(camp: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    """One full analysis of the campaign, shared by the read-only tests (speed)."""
    out = tmp_path_factory.mktemp("analysis")
    return _run(camp, out, n_perm=20), out


def mini_campaign(camp: Path, dest: Path) -> Path:
    """A small copy of the campaign (two trunks: trunk, sham, L1, L2) for refusal paths."""
    c = dest / "TR-H"
    for t in ("T0", "T1"):
        for a in ("trunk", "sham", "L1", "L2"):
            shutil.copytree(camp / f"{a}-{t}", c / f"{a}-{t}")
    return c


def test_registered_arms_whitelist_ghosts_and_exclusions(analysed: tuple[dict, Path]) -> None:
    res, out = analysed
    arms = res["arms"]
    assert arms["sham"]["n"] == N_TRUNKS + 1  # the ghost under sham-T0 is not a sham run
    assert arms["ghost"]["n"] == 4 and "ghost_sham" not in arms
    ghosts = [r for r in res["runs"] if r["registered_arm"] == "ghost"]
    assert sorted(r["members"][0] for r in ghosts) == [0, 1, 2, 3]
    expect_ghost = (1.70 - TB.p_nash) / SCALE  # the live slot only, never the replayed rivals
    assert all(r["delta_p"] == pytest.approx(expect_ghost) for r in ghosts)
    l6 = [r for r in res["runs"] if r["registered_arm"] == "L6"]
    assert all(r["slots_excluded"] == [4] and r["members"] == [0, 1, 2, 3] for r in l6)
    assert all(np.isfinite(r["d7"]["d7"]) for r in l6)
    ign = {i["run_id"]: i for i in res["ignored_runs"]}
    assert set(ign) == {"weird-T0", "sham-T0--ghost0"}  # listed, never pooled
    assert ign["sham-T0--ghost0"]["ghost"] and not ign["weird-T0"]["ghost"]
    out = out / "TR-H"
    exc = {e["run_id"]: e["reason"] for e in json.loads((out / "exclusions.json").read_text())}
    assert set(exc) == {"L1-bad", "L1-missing", "L1-beta1"}
    assert "failed decisions" in exc["L1-bad"] and "incumbents missing" in exc["L1-missing"]
    assert "config mismatch" in exc["L1-beta1"] and "beta=1.0" in exc["L1-beta1"]
    shams = [r for r in res["runs"] if r["registered_arm"] == "sham"]
    assert all(r["config_checked"] for r in shams)  # the real TR-H market passes the check
    # per-arm pinned benchmarks: L1's window market is its own, its primary is the trunk's
    assert res["benchmarks"]["L1"]["primary"] == res["benchmarks"]["sham"]["primary"]
    assert res["benchmarks"]["L1"]["window"]["p_nash"] < TB.p_nash
    assert res["benchmarks"]["L1"]["sha256"] == SPEC.benchmark_files["TR-H"].sha256
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "Mechanical vs behavioural" in md and "L1-missing" in md and "descriptive" in md


def test_gate2_qualification_and_mechanical_decomposition(analysed: tuple[dict, Path]) -> None:
    res, _ = analysed
    q = res["inference"]["qualification"]
    prov = AB["L2"].window.method.endswith("flow_diversion")  # until the sweep re-pin
    assert ("L2" in res["provisional_benchmarks"]) == prov
    assert "L1" not in res["provisional_benchmarks"]
    assert q["available"] and q["not_qualified"] == ["T8"] and len(q["qualified"]) == N_TRUNKS
    c = res["contrasts"]["L1"]
    assert c["unqualified_trunks_dropped"] == ["T8"] and c["n_pairs"] == N_TRUNKS
    expect = np.mean([(0.25 + 0.01 * k) / SCALE for k in range(N_TRUNKS)])
    assert c["delta_p"]["mean"] == pytest.approx(expect, rel=1e-9)
    assert c["delta_p"]["alternative"] == "greater" and c["delta_p"]["t"]["p"] < 1e-6
    # frozen-quote decomposition: the primary estimand has no mechanical part
    assert c["mechanical"]["delta_p"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert c["behavioural"]["delta_p"]["mean"] == pytest.approx(expect, rel=1e-9)
    tot = c["delta_pi_window"]["mean"]
    mech = c["mechanical"]["delta_pi_window"]["mean"]
    beh = c["behavioural"]["delta_pi_window"]["mean"]
    assert beh == pytest.approx(tot - mech, abs=1e-9) and abs(mech) > 0.01
    # L6: the entrant's realised quotes enter the mechanical re-scoring
    assert res["contrasts"]["L6"]["mechanical"]["delta_pi"]["n"] == N_TRUNKS
    # L2 (fringe flow diversion): profits fall mechanically, prices do not
    l2 = res["contrasts"]["L2"]
    assert l2["mechanical"]["delta_p"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert l2["mechanical"]["delta_pi"]["mean"] > 0.1


def test_h1_single_pre_specified_lever(analysed: tuple[dict, Path]) -> None:
    res, out = analysed
    h1 = res["h1"]
    d = [(0.15 + 0.005 * k) / SCALE for k in range(N_TRUNKS)]
    assert h1["lever"] == SPEC.primary_lever == "L2" and h1["n"] == N_TRUNKS
    assert h1["mean"] == pytest.approx(np.mean(d), rel=1e-9)
    se = np.std(d, ddof=1) / np.sqrt(N_TRUNKS)
    from scipy import stats as sps

    assert h1["lower_bound_one_sided_95"] == pytest.approx(
        np.mean(d) - sps.t.ppf(0.95, N_TRUNKS - 1) * se, rel=1e-9
    )
    assert h1["p_one_sided"] < 1e-6 and h1["wilcoxon_sensitivity"]["alternative"] == "greater"
    assert h1["bca_lo"] <= h1["mean"] <= h1["bca_hi"] and h1["mqc"] == 3.0
    assert h1["h1_supported"] and h1["h1_practically_significant"] and h1["tost"] is None
    assert h1["rule"]["multiplicity"].startswith("none")
    st = json.loads((out / "TR-H" / "stats.json").read_text(encoding="utf-8"))
    assert st["h1"]["h1_supported"] is True and st["inference"]["primary_lever"] == "L2"
    assert st["inference"]["sidedness"].startswith("one-sided")
    # an MQC above 5 pp: not supported, TOST reported instead
    inf = SPEC.inference
    diffs = list(res["contrasts"]["L2"]["delta_p"]["diffs"].values())

    def h1_with(mqc):
        return S.primary_h1_test(
            diffs,
            mqc,
            inf.alpha,
            inf.h1.mqc_max_pp,
            inf.h1.practical_lower_bound_gt,
            inf.tost_margin,
            500,
            inf.seed,
        )

    h1b = h1_with(6.0)
    assert not h1b["h1_supported"] and not h1b["h1_practically_significant"]
    core = {k: h1b["checks"][k] for k in ("p_lt_alpha", "mqc_ok", "lower_bound_gt")}
    assert core == {"p_lt_alpha": True, "mqc_ok": False, "lower_bound_gt": False}  # D-17 adds p_own
    # D-19: TOST is reported only when neither scoring is significant; here the trunk test is
    assert h1b["tost"] is None and h1b["checks"]["p_lt_alpha"]
    flat = S.primary_h1_test([0.0, 0.01, -0.01, 0.02, 0.0, -0.02, 0.01, 0.0], 2.0, B=300)
    assert isinstance(flat["tost"], dict) and "equivalent" in flat["tost"]
    # no MQC estimate at all: never supported
    assert not h1_with(None)["h1_supported"]


def test_h1r_family_holm_and_descriptive_best_lever(analysed: tuple[dict, Path]) -> None:
    res, _ = analysed
    h1r, inf = res["h1r"], res["inference"]
    assert h1r["m"] == inf["m"] == 8 and h1r["family"] == SPEC.lever_family
    assert inf["missing_levers_at_p1"] == ["L3", "L4a", "L5", "L7", "L8"]
    lv = h1r["levers"]
    assert all(lv[k]["p"] == 1.0 and lv[k]["placeholder"] for k in inf["missing_levers_at_p1"])
    raw = {k: lv[k]["p"] for k in ("L1", "L2", "L6")}
    adj, rej = S.holm([raw.get(k, 1.0) for k in SPEC.lever_family])
    for i, k in enumerate(SPEC.lever_family):
        assert lv[k]["p_holm"] == pytest.approx(adj[i]) and lv[k]["reject"] == bool(rej[i])
    assert lv["L1"]["reject"] and lv["L2"]["reject"]
    best = h1r["best_lever_split_sample"]
    assert best["label"].startswith("descriptive")
    assert best["selection_trunks"] == ["T1", "T3", "T5", "T7"]
    assert best["estimation_trunks"] == ["T0", "T2", "T4", "T6"]
    assert best["best_lever"] == "L1"
    diffs = res["contrasts"]["L1"]["delta_p"]["diffs"]
    assert best["estimate"]["mean"] == pytest.approx(
        np.mean([diffs[t] for t in best["estimation_trunks"]])
    )
    # sigma_d: pooled SD of the realised lever-minus-sham differences (not the sham pairs)
    ds = [np.array(list(res["contrasts"][k]["delta_p"]["diffs"].values())) for k in raw]
    pooled = np.sqrt(sum(np.sum((d - d.mean()) ** 2) for d in ds) / sum(d.size - 1 for d in ds))
    assert inf["sigma_d"]["estimate"] == pytest.approx(pooled, rel=1e-9)
    assert 0 <= inf["power"]["h1"] <= 1


def test_pin_mismatch_and_stale_configs_are_refused(camp: Path, tmp_path: Path) -> None:
    camp = mini_campaign(camp, tmp_path / "mini")
    text = ARMS.read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text.replace(SPEC.benchmark_files["TR-H"].sha256, "0" * 64), encoding="utf-8")
    with pytest.raises(PinError, match="sha256"):
        analyze_campaign(camp, bad, ROOT, tmp_path / "o", mqc=MQC, B=200, n_perm=0)
    # a lever config edited after pinning: the pinned file is stale
    src = (ROOT / "configs/sprint/tier0/L1-H.yaml").read_text(encoding="utf-8")
    cfg = tmp_path / "L1-H.yaml"
    cfg.write_text(
        re.sub(r"(path: market\.routing\.tau\n\s+value: )\S+", r"\g<1>0.05", src), "utf-8"
    )
    stale = tmp_path / "stale.yaml"
    stale.write_text(
        text.replace("configs/sprint/tier0/L1-H.yaml, branch", cfg.as_posix() + ", branch"),
        encoding="utf-8",
    )
    with pytest.raises(PinError, match="stale"):
        analyze_campaign(camp, stale, ROOT, tmp_path / "o", mqc=MQC, B=200, n_perm=0)
    res = analyze_campaign(
        camp, stale, ROOT, tmp_path / "o", mqc=MQC, B=200, n_perm=0, allow_stale_benchmarks=True
    )
    assert res["stale_benchmarks"] == ["TR-H:L1"]


def test_committed_benchmarks_match_builder_and_pin(monkeypatch) -> None:
    from finswarm.analysis import benchmarks_file as BF

    # closed forms only: the replicated fringe sweep (about 1 h) is never re-run in tests
    monkeypatch.setattr(BF, "_sweep_module", lambda: None)
    doc = build_benchmark_file(SPEC, "TR-H", ROOT)
    raw = (ROOT / SPEC.benchmark_files["TR-H"].file).read_bytes().replace(b"\r\n", b"\n")
    on_disk = json.loads(raw)
    assert dumps(on_disk).encode() == raw  # canonical layout (LF, sorted keys)
    assert set(on_disk["markets"]) == set(doc["markets"])
    for name, entry in on_disk["markets"].items():
        built = doc["markets"][name]
        if str(entry["method"]).startswith("sweep"):  # pinned sweep: configuration identity
            assert entry["market_sha256"] == built["market_sha256"], name
            assert entry["model"] == built["model"] and not entry["provisional"], name
        else:
            assert entry == built, f"{name}: regenerate with scripts/analyze.py --make-benchmarks"
    m = doc["markets"]
    assert m["trunk"]["model"]["n_incumbents"] == 4 and m["L6"]["model"]["n_entrants"] == 1
    assert m["trunk"]["benchmarks"]["pi_nash"] == pytest.approx(8.146069, rel=1e-6)  # beta 100
    assert m["trunk"]["d7_cartel"] == pytest.approx(0.6213, abs=1e-4)
    assert m["GR-s4"]["model"]["params"]["tau"] == 0.02


class FakeSweep:
    """Stands in for ``finswarm.market.sweep``: closed-form expected profits per seed."""

    seeds: list = []

    @staticmethod
    def profit_fn(cfg, *, periods=50, warmup_periods=None, seed=None):
        from finswarm.metrics.benchmarks import MarketModel

        mm = MarketModel.from_config(cfg.market)
        FakeSweep.seeds.append(seed)

        def fn(batch):
            b = np.atleast_2d(batch)
            m2 = replace(mm, n_incumbents=b.shape[1])
            return np.where(np.isfinite(b), (b - 1.0) * m2.quantities(b), 0.0)

        return fn


def test_fringe_market_uses_replicated_sweep_when_available(monkeypatch) -> None:
    from finswarm.analysis import benchmarks_file as BF

    res = BF.resolve_market(SPEC.benchmark_files["TR-H"].markets["L2"], ROOT)
    closed = solve_market(res, "closed_form")
    assert closed["provisional"] and closed["method"] == "closed_form:flow_diversion"
    FakeSweep.seeds = []
    monkeypatch.setattr(BF, "_sweep_module", lambda: FakeSweep)
    sw = solve_market(res, "auto", {"n_starts": 2, "periods": 5})
    assert sw["method"].startswith("sweep") and not sw["provisional"]
    assert FakeSweep.seeds == list(range(10))  # R = 10 seed replications (SAP)
    info = sw["sweep"]
    assert info["replications"] == 10 and info["periods_per_evaluation"] == 5
    assert "nash_spread" in info and len(info["nash_points"]) == info["n_converged"]
    assert sw["benchmarks"]["pi_nash"] == pytest.approx(closed["benchmarks"]["pi_nash"], rel=0.1)
    # without the fringe the closed form is kept even when the sweep is importable
    trunk = BF.resolve_market(SPEC.benchmark_files["TR-H"].markets["trunk"], ROOT)
    assert solve_market(trunk, "auto")["method"] == "closed_form:routed_logit"


def test_lever_values_are_checked_against_the_registry(tmp_path: Path) -> None:
    spec = SPEC.model_copy(deep=True)
    for name, a in spec.arms.items():
        if name != "L1":
            a.lever = None
    src = (ROOT / "configs/sprint/tier0/L1-H.yaml").read_text(encoding="utf-8")

    def variant(value: str, extra: str = "") -> str:
        p = tmp_path / f"L1-{value}{extra}.yaml"
        txt = re.sub(r"(path: market\.routing\.tau\n\s+value: )\S+", rf"\g<1>{value}", src)
        if extra:
            txt += "  - type: set_param\n    path: market.cost\n    value: 1.1\n"
        p.write_text(txt, encoding="utf-8")
        return p.as_posix()

    spec.arms["L1"].lever.config = variant("0.0625")
    assert check_lever_params(ROOT, spec) == []
    assert check_lever_params(ROOT, ARMS.relative_to(ROOT)) == check_lever_params(
        ROOT, SPEC.model_dump(mode="json")
    )
    spec.arms["L1"].lever.config = variant("0.03125")  # not the registered L1 level
    (msg,) = check_lever_params(ROOT, spec)
    assert "market.routing.tau = 0.03125, registered 0.0625" in msg
    spec.arms["L1"].lever.config = variant("0.0625", "x")
    assert any("market.cost" in m and "not registered" in m for m in check_lever_params(ROOT, spec))


def test_holm_fixed_placeholders_and_helpers() -> None:
    fam = ["L1", "L2", "L3", "L4a", "L5", "L6", "L7", "L8"]
    h = S.holm_fixed({"L1": 0.004, "L6": 0.03, "X": 0.0}, fam)
    assert list(h) == fam and "X" not in h
    assert h["L1"]["p_holm"] == pytest.approx(0.032) and h["L1"]["reject"]
    assert h["L6"]["p_holm"] == pytest.approx(0.21) and not h["L6"]["reject"]
    assert all(h[k]["p"] == 1.0 and h[k]["placeholder"] for k in fam if k not in {"L1", "L6"})
    # dropping the missing levers would have made L6 significant: the placeholders prevent it
    adj, _ = S.holm([0.004, 0.03])
    assert adj[1] < 0.05
    sel, est = S.split_by_seed_parity({"a": 1, "b": 2, "c": 3, "d": 4})
    assert sel == ["a", "c"] and est == ["b", "d"]
    eff = {"L1": 0.3, "L2": 0.5, "L6": 0.2}
    assert S.select_best(eff, fam) == "L2"
    assert S.select_best(eff, fam, {"L1": 1.0, "L2": 7.0, "L6": 0.5}, mqc_max_pp=5.0) == "L1"
    t = S.paired_t([1.0, 2.0, 3.0], [0.5, 1.0, 2.9], alternative="greater")
    t2 = S.paired_t([1.0, 2.0, 3.0], [0.5, 1.0, 2.9])
    assert t["p"] == pytest.approx(t2["p"] / 2)
    assert S.power_one_sided(12, 0.20, 0.25) > 0.9 > S.power_one_sided(12, 0.20, 0.25, 0.05 / 8)


def test_primary_h1_test_and_descriptive_split_on_synthetic_data() -> None:
    rng = np.random.default_rng(1)
    d = 0.30 + rng.normal(0, 0.15, 12)
    r = S.primary_h1_test(d, mqc=2.0, B=2000, d_own=d)
    assert r["h1_supported"] and r["h1_practically_significant"] and r["tost"] is None
    assert r["practical_verdict"] == "practically significant"  # D-19 as its own field (m9)
    assert not S.primary_h1_test(d, mqc=2.0, B=2000)["h1_supported"]  # m9: no own scoring
    w = 0.06 + rng.normal(0, 0.02, 12)
    weak = S.primary_h1_test(w, mqc=2.0, B=2000, d_own=w)
    assert weak["h1_supported"] and not weak["h1_practically_significant"]
    null = S.primary_h1_test(rng.normal(0, 0.05, 12), mqc=2.0, B=2000)
    assert not null["h1_supported"] and null["tost"]["equivalent"]
    assert not S.primary_h1_test([0.5], mqc=1.0, B=200)["h1_supported"]  # n < 2
    fam = ["L1", "L2", "L3", "L4a", "L5", "L6", "L7", "L8"]
    trunks = [f"T{k}" for k in range(12)]
    seeds = {t: k for k, t in enumerate(trunks)}
    diffs = {
        lev: {t: mu + rng.normal(0, 0.1) for t in trunks}
        for lev, mu in {"L1": 0.40, "L2": 0.05}.items()
    }
    # a lever that looks great on the estimation half only cannot be selected
    diffs["L6"] = {t: (0.9 if seeds[t] % 2 == 0 else -0.2) for t in trunks}
    out = S.split_sample_best(diffs, seeds, fam, B=2000)
    assert out["best_lever"] == "L1" and out["label"].startswith("descriptive")
    assert set(out["selection_trunks"]).isdisjoint(out["estimation_trunks"])
    xe = [diffs["L1"][t] for t in out["estimation_trunks"]]
    assert out["estimate"]["mean"] == pytest.approx(np.mean(xe))
    assert out["estimate"]["bca_lo"] <= out["estimate"]["mean"] <= out["estimate"]["bca_hi"]


def test_analyze_script_cli(camp: Path, tmp_path: Path) -> None:
    camp = mini_campaign(camp, tmp_path / "mini")
    import importlib.util

    spec = importlib.util.spec_from_file_location("_an", ROOT / "scripts" / "analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mqc = tmp_path / "mqc.json"
    mqc.write_text(json.dumps(MQC), encoding="utf-8")
    common = ["--arms", str(ARMS), "--root", str(ROOT), "--out-root", str(tmp_path / "a")]
    rc = mod.main([str(camp), *common, "--mqc", str(mqc), "--B", "500", "--n-perm", "0"])
    assert rc == 0 and (tmp_path / "a" / "TR-H" / "stats.json").exists()
    bad = tmp_path / "bad.yaml"
    shutil.copy(ARMS, bad)
    bad.write_text(
        bad.read_text(encoding="utf-8").replace(SPEC.benchmark_files["TR-H"].sha256, "f" * 64),
        encoding="utf-8",
    )
    common[1] = str(bad)
    assert mod.main([str(camp), *common]) == 3


def side1_ok(fig: Path) -> bool:
    side = json.loads((fig / "fig1_positive_control.json").read_text(encoding="utf-8"))
    d = side["data"]
    return d["thresholds"] == [0.25, 0.10] and len(d["delta_pi_sham"]) == len(d["delta_p_sham"])


def test_frontier_figure_from_stats_json(analysed: tuple[dict, Path], tmp_path: Path) -> None:
    import importlib.util

    from finswarm.analysis.figures import frontier_from_stats

    res, out = analysed
    st = json.loads((out / "TR-H" / "stats.json").read_text(encoding="utf-8"))
    pts, band, ids, not_run = frontier_from_stats(st)
    by = {p["label"]: p for p in pts}
    assert set(by) == {"L1", "L2", "L6"} and by["L2"]["primary"] and not by["L1"]["primary"]
    assert set(not_run) == set(st["h1r"]["family"]) - {"L1", "L2", "L6"}
    for lev, p in by.items():
        c = st["contrasts"][lev]
        assert p["dc"] == pytest.approx(c["delta_p"]["mean"])  # total, trunk scoring
        assert p["lo"] == c["delta_p"]["bca_lo"] and p["hi"] == c["delta_p"]["bca_hi"]
        assert p["mechanical"] == pytest.approx(c["mechanical"]["delta_p"]["mean"], abs=1e-12)
        assert p["behavioural"] == pytest.approx(c["behavioural"]["delta_p"]["mean"], abs=1e-12)
        assert p["own"] == pytest.approx(c["delta_p_own"]["mean"])
        assert p["dual_label"] == c["dual_benchmark"]["label"]
        assert p["mqc"] == MQC["levers"][lev]["mean"]
    # no sham2 arm in this campaign: the sham band is 0 (never 1.96 sigma_d / sqrt(n))
    assert "sham2" not in st["contrasts"] and band == 0.0
    spec = importlib.util.spec_from_file_location("_mf", ROOT / "scripts" / "make_figures.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fig = tmp_path / "figs"
    argv = [str(out / "TR-H"), "--stats", str(out / "TR-H" / "stats.json"), "--out", str(fig)]
    more = ["--gradient-arms", "trunk,L1,L2", "--ablation-arms", "sham,L1", "--t0", "100"]
    assert mod.main([*argv, "--pc-arms", "sham,L1", *more]) == 0
    names = {p.name for p in fig.iterdir()}
    for stem in ("fig1_positive_control", "fig2_gradient", "fig3_frontier", "fig4_action_carrier"):
        assert any(n.startswith(stem) for n in names), (stem, names)
    side = json.loads((fig / "fig2_gradient.json").read_text(encoding="utf-8"))
    assert side["data"]["trunk"]["marker"] == "s" and side["data"]["L1"]["marker"] == "o"
    assert not [k for k in side["data"] if k.startswith("ref_")]  # Fish values are unverified: off
    assert side["role"] == "confirmatory" and len(side["inputs"]) == 1
    assert len(side["inputs"][0]["sha256"]) == 64 and "code_sha" in side
    # several stats files merge, a level can pool arms (trunk+L1), the Fish mark is opt-in
    stats = str(out / "TR-H" / "stats.json")
    fig2 = tmp_path / "figs2"
    argv2 = [str(out / "TR-H"), "--stats", stats, stats, "--out", str(fig2), "--fish-reference"]
    assert mod.main([*argv2, "--pc-arms", "sham,L1", "--gradient-arms", "trunk+L1,L2"]) == 0
    side2 = json.loads((fig2 / "fig2_gradient.json").read_text(encoding="utf-8"))
    assert side2["data"]["ref_Fish et al. (P1), unverified"] == [0.7, 0.7]
    assert side2["data"]["trunk+L1"]["marker"] == "s" and len(side2["inputs"]) == 2
    pooled_n = len(side2["data"]["trunk+L1"]["values"])
    assert pooled_n == len(side["data"]["trunk"]["values"]) + len(side["data"]["L1"]["values"])
    side3 = json.loads((fig / "fig3_frontier.json").read_text(encoding="utf-8"))
    cp = side3["data"]["ceiling_panel"]  # the ceiling caveat is in the figure (lead ruling)
    by_lev = {r["label"]: r for r in cp["levers"]}
    assert set(by_lev) == {"L1", "L2", "L6"}
    for lev, r in by_lev.items():
        assert r["raw"] == pytest.approx(st["contrasts"][lev]["delta_p_raw"]["mean"])
        assert r["share_arm"] == pytest.approx(st["contrasts"][lev]["share_quotes_above_pm"]["arm"])
    assert side3["data"]["sham_band"] == 0.0 and sorted(side3["data"]["not_run"]) == sorted(not_run)
    side4 = json.loads((fig / "fig4_action_carrier.json").read_text(encoding="utf-8"))
    sham = side4["data"]["sham"]
    assert max(sham["mean"]) <= 1.0 + 1e-9  # capped price index, not the profit Delta
    assert min(sham["t"]) >= 80 and side1_ok(fig)


@pytest.mark.slow
def test_fringe_sweep_builder_on_sim_core_reduced_setting() -> None:
    """The builder's replicated profit function on the real sim-core sweep, reduced setting
    (the registered pin uses R = 10 seeds, 50 periods, 10 starts: about 1 h, run by the
    integrator with --make-benchmarks; the solver path is covered with FakeSweep above)."""
    from finswarm.analysis import benchmarks_file as BF
    from finswarm.config.schema import RunConfig

    sw = BF._sweep_module()
    if sw is None:
        pytest.skip("finswarm.market.sweep not available")
    res = BF.resolve_market(SPEC.benchmark_files["TR-H"].markets["L2"], ROOT)
    run = RunConfig.model_validate(res["run"])
    model = L2_M
    assert model.fringe_share > 0
    closed = replace(model, fringe_share=0.0).benchmarks()  # no fringe: all flow to dealers
    p = np.full((2, model.n_sellers), closed.p_nash)
    p[1] = closed.p_mono
    fn = BF.replicated_profit_fn(sw, run, [int(run.seed), int(run.seed) + 1], periods=3)
    prof = np.asarray(fn(p))
    assert prof.shape == p.shape and np.isfinite(prof).all() and (prof > 0).all()
    assert prof[0].mean() < closed.pi_nash  # the fringe diverts flow at the Nash quotes


def test_gradient_levels_match_by_the_level_tag_not_the_resolved_tau() -> None:
    """A-015: the engine resolves s1 to tau null (the default 0.25); the level tag
    (tags.level s1..s4) is what the manifests carry, so all four levels match (16 runs)."""
    from finswarm.analysis.campaign import match_arm

    def man(level: str, tau) -> dict:
        routing = {} if tau is None else {"routing": {"tau": tau}}
        return {
            "arm": "gradient",
            "label": "GR-H",
            "resolved_config": {"tags": {"level": level}, "market": routing},
        }

    got = {}
    for level, tau in (("s1", None), ("s2", 0.125), ("s3", 0.0625), ("s4", 0.02)):
        for seed in range(4):
            info = {"run_id": f"{level}-{seed}", "arm": "gradient", "label": "GR-H"}
            got.setdefault(match_arm(SPEC, info, man(level, tau)), []).append(seed)
    assert {k: len(v) for k, v in got.items()} == {f"GR-s{i}": 4 for i in (1, 2, 3, 4)}
    assert sum(len(v) for v in got.values()) == 16


def test_s1c_llm_runs_and_r3_match_registered_arms() -> None:
    """A-016: the competitive-sting runs carry S1c-LLM-H (CA-H) and S1c-LLM-TR-H (TR-H);
    R3-H is registered on the H lane."""
    from finswarm.analysis.campaign import match_arm

    def hit(arm: str, label: str) -> str | None:
        info = {"run_id": "x", "arm": arm, "label": label}
        return match_arm(SPEC, info, {"resolved_config": {"tags": {"label": label}}})

    assert hit("S1c", "S1c-LLM-H") == "S1c_LLM" and hit("S1c", "S1c-LLM-TR-H") == "S1c_LLM_TR"
    assert hit("S1c", "S1c-LLM") is None and hit("R3", "R3-H") == "R3"
