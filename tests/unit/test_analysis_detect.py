"""Detection corpus from campaign results, process labels via probe + placebo, rule freeze and
the single evaluation (A2, DETECTION_PROTOCOL sections 1, 4, 6; D-21)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from _arms_fixture import ROOT, temp_registry
from _metrics_synth import market_frame, write_frame_run

from finswarm.analysis import detect as DX
from finswarm.analysis.arms import load_registered_arms, resolve_arm_benchmarks
from finswarm.config.load import load_campaign_config
from finswarm.metrics import detection as DT

TRH_MARKET = load_campaign_config(ROOT / "configs/sprint/tier0/TR-H.yaml").base.model_dump(
    mode="json"
)["market"]
N_TRUNKS = 6


def _load_analyze():
    spec = importlib.util.spec_from_file_location(
        "_script_analyze_detect", ROOT / "scripts" / "analyze.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def man(arm, label, trunk, seed, branch=True, status="ok", **tags) -> dict:
    m = {
        "arm": arm,
        "label": label,
        "trunk_id": trunk,
        "seed": seed,
        "status": status,
        "branch_period": 100 if branch else None,
        "resolved_config": {"tags": {"arm": arm, "label": label, **tags}},
    }
    if arm == "sham":
        m["resolved_config"]["market"] = TRH_MARKET
    return m


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory):
    """Registry + campaigns: 6 trunks with sham / sham2 / S2 / ghosts (4 punished, 2 not),
    4 CA runs, 5 SC-3 cartels and 5 SC-1 competitors."""
    arms = temp_registry(tmp_path_factory.mktemp("arms"))
    spec = load_registered_arms(arms)
    ab = resolve_arm_benchmarks(spec, ROOT)
    tm = ab["sham"].primary_model
    croot = tmp_path_factory.mktemp("campaigns")
    tr = croot / "TR-H"
    rng = np.random.default_rng(0)
    for k in range(N_TRUNKS):
        t = f"T{k}"
        trunk = market_frame(np.full((30, 4), 1.90), tm, start=71)
        write_frame_run(tr, f"trunk-{t}", trunk, man("trunk", "TR-H", t, k, branch=False))
        # branch runs cover 101-150 so the probe window (deviation at 101) is inside
        lead = 1.85 + np.cumsum(rng.normal(0, 0.01, 50))
        sham_p = np.column_stack([lead + rng.normal(0, 0.003, 50) for _ in range(4)])
        sham = market_frame(sham_p, tm, start=101)
        write_frame_run(tr, f"sham-{t}", sham, man("sham", "SH-H", t, k))
        sham2 = market_frame(sham_p + rng.normal(0, 0.01, (50, 4)), tm, start=101)
        write_frame_run(tr, f"sham2-{t}", sham2, man("sham2", "SH2-H", t, k))
        probe = sham_p.copy()
        probe[0, 0] -= 0.30  # forced undercut by slot 0 at period 101
        if k < 4:
            probe[1:4, 1:] -= 0.25  # punished: rivals cut within 3 periods
        write_frame_run(tr, f"S2-{t}", market_frame(probe, tm, start=101), man("S2", "S2-H", t, k))
        for g in range(4):  # the ghosts run under the SH-H sham, as in the store (A-013)
            gp = sham_p.copy()
            gp[:, g] = 1.80 + rng.normal(0, 0.03, 50)  # the live dealer: independent noise
            write_frame_run(
                tr / f"sham-{t}" / "ghost",
                f"sham-{t}--ghost{g}",
                market_frame(gp, tm, start=101),
                man("sham", "SH-H", t, k, ghost_of=f"sham-{t}", ghost_slot=str(g)),
                subdir=str(g),
            )
    ca = croot / "CA-H"
    for k in range(4):
        comp = 1.40 + rng.normal(0, 0.03, (30, 4))
        write_frame_run(
            ca,
            f"ca-{k}",
            market_frame(comp, tm, start=71),
            man("CA", "CA-H", f"C{k}", 500 + k, branch=False),
        )
    for label, level, n in (("SC-3", 1.95, 5), ("SC-1", 1.35, 5)):
        c = croot / label
        for k in range(n):
            p = level + rng.normal(0, 0.01, (40, 4))
            write_frame_run(
                c,
                f"{label}-{k}",
                market_frame(p, tm, start=111),
                man(label, label, f"{label}-{k}", 900 + k, branch=False),
            )
    cspec = yaml.safe_load((ROOT / "configs/detection/corpus.yaml").read_text(encoding="utf-8"))
    cspec["features"] = {"n_surrogates": 5, "n_boot": 10, "d7_n_perm": 0, "n_shifts": 3}
    cspec["sources"] = [
        {
            "campaign": "TR-H",
            "arm": "sham",
            "kind": "llm_sham",
            "probe": "S2",
            "placebo": "sham2",
            "ghosts": "ghost",
        },
        {"campaign": "TR-H", "arm": "ghost", "kind": "ghost"},
        {"campaign": "TR-H", "arm": "trunk", "kind": "outcome", "ghosts": "ghost"},
        {"campaign": "CA-H", "arm": "CA", "kind": "instruction"},
        {"campaign": "SC-3", "label": "SC-3", "kind": "sc3", "window": "branch"},
        {"campaign": "SC-1", "label": "SC-1", "kind": "sc1", "window": "branch"},
    ]
    cpath = croot / "corpus.yaml"
    cpath.write_text(yaml.safe_dump(cspec, sort_keys=False), encoding="utf-8")
    return {"arms": arms, "spec": spec, "croot": croot, "corpus_path": cpath, "tm": tm}


def test_corpus_labels_by_process_with_probe_placebo_and_ghost_netting(world) -> None:
    cs = DX.load_corpus_spec(world["corpus_path"])
    corpus, meta = DX.build_corpus_from_results(world["croot"], cs, world["spec"], ROOT)
    kinds = dict(corpus.group_by("kind").len().iter_rows())
    assert kinds["llm_punished"] == 4 and kinds["llm_unpunished"] == 2
    assert kinds["ghost"] == 4 * N_TRUNKS and kinds["instruction"] == 4
    assert kinds["sc3"] == 5 and kinds["sc1"] == 5 and kinds["outcome"] == N_TRUNKS
    assert set(corpus.filter(corpus["kind"] == "outcome")["label"].to_list()) == {None}
    assert corpus.filter(corpus["kind"] == "ghost")["label"].unique().to_list() == [0]
    assert corpus["seed_group"].null_count() == 0
    # seed group = "<lane>:<seed>" on every window (M7)
    assert all(
        sg.startswith(f"{ln}:") for sg, ln in zip(corpus["seed_group"], corpus["lane"], strict=True)
    )
    assert set(corpus.filter(corpus["kind"] == "sc3")["seed_group"]) == {
        f"CPU:{900 + k}" for k in range(5)
    }
    assert set(corpus.filter(corpus["kind"] == "ghost")["seed_group"]) == {
        f"H:{k}" for k in range(N_TRUNKS)
    }
    # no qualification list given: every trunk enters the placebo, and the meta says so
    assert meta["qualified_filter"] is False and meta["qualified"] is None
    assert meta["ca_netting"] == "cyclic_shift" and meta["ghost_netting"] == "cyclic_shift"
    by_src = {(s["campaign"], s["kind"]): s for s in meta["sources"]}
    assert by_src[("CA-H", "instruction")]["netting"] == "cyclic_shift"
    assert by_src[("TR-H", "llm_sham")]["netting"] == "ghost_set"
    assert by_src[("TR-H", "llm_sham")]["n_unnetted"] == 0
    assert len(meta["probe_flags"]) == len(meta["probe_flags_v1"]) == N_TRUNKS
    # ghost cyclic-shift null of every sham window's D5-net: 4 ghosts x 3 shifts per trunk
    gn = meta["ghost_null"]
    assert all(gn[f"T{k}"]["n"] == 12 and gn[f"T{k}"]["lane"] == "H" for k in range(N_TRUNKS))
    pooled = [x for k in range(N_TRUNKS) for x in gn[f"T{k}"]["d5net_draws"]]
    assert gn["sham_pooled"]["n"] == 12 * N_TRUNKS and gn["sham_pooled"]["lane"] == "H"
    assert gn["sham_pooled"]["q95"] == pytest.approx(float(np.quantile(pooled, 0.95)))
    assert set(corpus.filter(corpus["kind"] == "sc3")["lane"].to_list()) == {"CPU"}
    # outcome strata are set from the capped price index, never used as labels
    out = corpus.filter(corpus["kind"] == "outcome")
    assert set(out["outcome_label"].to_list()) <= {1, 0, None} and out["outcome_label"].max() == 1
    pl_ = meta["placebo"]["TR-H:sham"]
    assert pl_["n"] == N_TRUNKS and pl_["quantile"] == 0.95
    assert pl_["trunks"] == [f"T{k}" for k in range(N_TRUNKS)]
    assert pl_["t0"] == {f"T{k}": 101 for k in range(N_TRUNKS)}
    assert all(np.isfinite(corpus["D5net"].to_numpy()))
    assert (
        corpus.filter(corpus["kind"] == "sc3")["D7"].mean()
        > corpus.filter(corpus["kind"] == "sc1")["D7"].mean()
    )


def test_stub_freeze_and_single_evaluation(world, tmp_path: Path) -> None:
    rules = tmp_path / "frozen_rules.json"
    out = tmp_path / "detection"
    cs = DX.load_corpus_spec(world["corpus_path"])
    sha0 = DX.write_stub(cs, rules, world["corpus_path"])
    stub = DT.load_rules(rules)  # the stub carries a valid hash
    assert stub["status"] == "unfrozen" and stub["sha256"] == sha0
    with pytest.raises(ValueError):
        DX.evaluate_frozen(rules, out)  # not frozen yet
    res = DX.run_detect(
        "build", world["croot"], world["corpus_path"], rules, world["arms"], ROOT, out
    )
    assert res["n_windows"] > 30 and (out / "corpus.parquet").exists()
    sp = DX.run_detect(
        "split", world["croot"], world["corpus_path"], rules, world["arms"], ROOT, out
    )
    split = json.loads(rules.read_text(encoding="utf-8"))
    assert split["status"] == "split" and split["split"]["seed"] == cs.seed
    assert split["split"]["dev_groups"] == sp["dev_groups"] and "thresholds" not in split
    assert set(split["split"]["dev_groups"]).isdisjoint(split["split"]["eval_groups"])
    sp2 = DX.run_detect(  # idempotent: the same split is rewritten, never moved
        "split", world["croot"], world["corpus_path"], rules, world["arms"], ROOT, out
    )
    assert sp2["dev_groups"] == sp["dev_groups"]
    fr = DX.run_detect(
        "freeze", world["croot"], world["corpus_path"], rules, world["arms"], ROOT, out
    )
    frozen = DT.load_rules(rules)
    assert frozen["status"] == "frozen" and frozen["sha256"] == fr["rules_sha256"]
    assert frozen["split"]["dev_groups"] == sp["dev_groups"]  # freeze reproduces the split
    with pytest.raises(FileExistsError):
        DX.run_detect(
            "split", world["croot"], world["corpus_path"], rules, world["arms"], ROOT, out
        )
    assert set(frozen["split"]["dev_groups"]).isdisjoint(frozen["split"]["eval_groups"])
    assert (out / "eval.parquet").exists() and (out / "freeze.json").exists()
    assert frozen["features"]["D7"]["calibration_kinds"] == sorted(cs.theta7_kinds)
    with pytest.raises(FileExistsError):
        DX.run_detect(
            "freeze", world["croot"], world["corpus_path"], rules, world["arms"], ROOT, out
        )
    rep = DX.run_detect(
        "evaluate", world["croot"], world["corpus_path"], rules, world["arms"], ROOT, out, n_boot=50
    )
    assert rep["rules_sha256"] == fr["rules_sha256"]
    assert rep["report"]["cluster"] == "seed_group"
    dec = rep["decision"]
    assert dec["llm_only"]["positives"] == ["llm_punished"] and dec["llm_only"]["lane"] == "H"
    assert dec["scripted_validity"]["status"] in (
        "ok",
        "insufficient windows (need positives and negatives)",
    )
    assert "auc_ge_0.85" in rep["decision_rules"] or rep["decision_rules"]["status"] != "ok"
    assert (out / "report.json").exists() and "H9 decision set" in (out / "report.md").read_text(
        encoding="utf-8"
    )
    # a tampered frozen file is refused
    body = json.loads(rules.read_text(encoding="utf-8"))
    body["max_fpr"] = 0.5
    rules.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError):
        DX.evaluate_frozen(rules, out)


def test_cli_detect_steps(world, tmp_path: Path) -> None:
    analyze = _load_analyze()
    rules = tmp_path / "frozen_rules.json"
    common = [
        "--campaigns-root",
        str(world["croot"]),
        "--corpus-spec",
        str(world["corpus_path"]),
        "--rules",
        str(rules),
        "--arms",
        str(world["arms"]),
        "--root",
        str(ROOT),
        "--out-root",
        str(tmp_path / "analysis"),
        "--n-boot",
        "20",
    ]
    assert analyze.main(["--detect", "stub", *common]) == 0
    assert analyze.main(["--detect", "build", *common]) == 0
    assert (tmp_path / "analysis" / "detection" / "corpus.parquet").exists()
    assert analyze.main(["--detect", "freeze", *common]) == 0
    assert analyze.main(["--detect", "evaluate", *common]) == 0
    assert (tmp_path / "analysis" / "detection" / "report.json").exists()


# --------------------------------------------------------------------------- M6 / M7 guards


def _bp_man(arm, label, trunk, seed, bp, **tags) -> dict:
    return {**man(arm, label, trunk, seed, **tags), "branch_period": bp}


def _trunk(
    tr: Path,
    t: str,
    seed,
    tm,
    rng_seed: int,
    bp: int = 100,
    placebo: float = 0.02,
    punish: bool = True,
    sham: bool = True,
    sham2: bool = True,
    probe: bool = True,
    ghosts: int = 2,
) -> None:
    """Branch runs of one TR-H trunk over periods bp + 1 .. 150: the sham, the sham2 whose
    rivals sit exactly ``placebo`` below the sham for the 3 periods after bp + 1 (placebo cut
    = ``placebo`` at t0 = bp + 1, undefined at any other t0 reaching back before bp + 1),
    the S2 probe (slot 0 undercuts by 0.30 at bp + 1, rivals cut by 0.25 if ``punish``) and
    ``ghosts`` fresh-draw ghost sub-runs (live slots 0..ghosts-1)."""
    rng = np.random.default_rng(rng_seed)
    n, start = 150 - bp, bp + 1
    lead = 1.85 + np.cumsum(rng.normal(0, 0.01, n))
    sham_p = np.column_stack([lead + rng.normal(0, 0.003, n) for _ in range(4)])
    if sham:
        f = market_frame(sham_p, tm, start=start)
        write_frame_run(tr, f"sham-{t}", f, _bp_man("sham", "SH-H", t, seed, bp))
    if sham2:
        p2 = sham_p.copy()
        p2[1:4, 1:] -= placebo
        f = market_frame(p2, tm, start=start)
        write_frame_run(tr, f"sham2-{t}", f, _bp_man("sham2", "SH2-H", t, seed, bp))
    if probe:
        pp = sham_p.copy()
        pp[0, 0] -= 0.30
        if punish:
            pp[1:4, 1:] -= 0.25
        f = market_frame(pp, tm, start=start)
        write_frame_run(tr, f"S2-{t}", f, _bp_man("S2", "S2-H", t, seed, bp))
    if ghosts and sham:  # the ghosts run under the SH-H sham, as in the store (A-013)
        for g in range(ghosts):
            gp = sham_p.copy()
            gp[:, g] = 1.80 + rng.normal(0, 0.03, n)
            write_frame_run(
                tr / f"sham-{t}" / "ghost",
                f"sham-{t}--ghost{g}",
                market_frame(gp, tm, start=start),
                _bp_man("sham", "SH-H", t, seed, bp, ghost_of=f"sham-{t}", ghost_slot=str(g)),
                subdir=str(g),
            )


def _sham_spec(**over) -> DX.CorpusSpec:
    cspec = yaml.safe_load((ROOT / "configs/detection/corpus.yaml").read_text(encoding="utf-8"))
    cspec["features"] = {"n_surrogates": 5, "n_boot": 10, "d7_n_perm": 0, "n_shifts": 3}
    src = {
        "campaign": "TR-H",
        "arm": "sham",
        "kind": "llm_sham",
        "probe": "S2",
        "placebo": "sham2",
        "ghosts": "ghost",
    }
    cspec["sources"] = [{**src, **over}]
    return DX.CorpusSpec.model_validate(cspec)


def _build(world, croot: Path, spec: DX.CorpusSpec | None = None, **kw):
    return DX.build_corpus_from_results(croot, spec or _sham_spec(), world["spec"], ROOT, **kw)


def _set_manifest(run_dir: Path, **fields) -> None:
    p = run_dir / "manifest.json"
    p.write_text(json.dumps({**json.loads(p.read_text(encoding="utf-8")), **fields}))


def _drop_slot0(run_dir: Path) -> None:
    part = run_dir / "agent_period" / "part-0000.parquet"
    pl.read_parquet(part).filter(pl.col("slot") != 0).write_parquet(part)


def test_placebo_per_trunk_t0_own_sham_qualified_filter_and_ghost_null(world, tmp_path) -> None:
    croot = tmp_path / "c"
    tr = croot / "TR-H"
    _trunk(tr, "A", 0, world["tm"], 1, bp=100, placebo=0.02)
    _trunk(tr, "B", 1, world["tm"], 2, bp=110, placebo=0.05, punish=False)
    _trunk(tr, "C", 2, world["tm"], 3, bp=120, placebo=0.03, probe=False)
    corpus, meta = _build(world, croot)
    pm = meta["placebo"]["TR-H:sham"]
    # every trunk at its own S2 deviation period (one shared t0 = 101 leaves B and C undefined)
    assert pm["t0"] == {"A": 101, "B": 111, "C": 121} and pm["probe_t0"] == {"A": 101, "B": 111}
    assert pm["cut_by_trunk"] == pytest.approx({"A": 0.02, "B": 0.05, "C": 0.03})
    assert pm["n"] == 3 and pm["trunks"] == ["A", "B", "C"]
    assert meta["qualified_filter"] is False and pm["qualified_filter"] is False
    kinds = dict(zip(corpus["trunk_id"].to_list(), corpus["kind"].to_list(), strict=True))
    assert kinds == {"A": "llm_punished", "B": "llm_unpunished", "C": "llm_unprobed"}
    assert meta["probe_flags_v1"] == {"sham-A": "llm_punished", "sham-B": "llm_unpunished"}
    assert sorted(corpus["seed_group"].to_list()) == ["H:0", "H:1", "H:2"]
    # ghost cyclic-shift null of A's sham D5-net, recomputed from its ghost sub-runs
    gn = meta["ghost_null"]
    frames = [
        DX._window_of(
            DX._frame(tr / "sham-A" / "ghost" / str(g), f"sham-A--ghost{g}"),
            (111, 150),
        )
        for g in range(2)
    ]
    exp = DT.ghost_shift_null_draws(frames, [0, 1], [0, 1, 2, 3], n_surrogates=5, n_shifts=3)
    assert exp.size == 6 and gn["A"]["d5net_draws"] == pytest.approx(exp.tolist())
    assert gn["A"]["n"] == 6 and gn["A"]["q95"] == pytest.approx(float(np.quantile(exp, 0.95)))
    pooled = [x for t in ("A", "B", "C") for x in gn[t]["d5net_draws"]]
    assert gn["sham_pooled"]["n"] == 18 and gn["sham_pooled"]["trunks"] == ["A", "B", "C"]
    assert gn["sham_pooled"]["q95"] == pytest.approx(float(np.quantile(pooled, 0.95)))
    json.dumps(DX._clean(meta))  # corpus_meta.json stays serialisable
    # the qualified filter: placebo cuts and the pooled null of the qualified trunks only
    _, mq = _build(world, croot, qualified={"A", "B"})
    pq, gq = mq["placebo"]["TR-H:sham"], mq["ghost_null"]["sham_pooled"]
    assert pq["trunks"] == ["A", "B"] and pq["n"] == 2 and pq["qualified_filter"] is True
    assert mq["qualified_filter"] is True and mq["qualified"] == ["A", "B"]
    assert gq["trunks"] == ["A", "B"] and gq["n"] == 12 and gq["n_mean_draws"] == 6
    mean = (np.asarray(gn["A"]["d5net_draws"]) + np.asarray(gn["B"]["d5net_draws"])) / 2
    assert gq["mean_q95"] == pytest.approx(float(np.quantile(mean, 0.95)))


def test_probe_without_its_own_sham_or_deviation_period_raises(world, tmp_path) -> None:
    croot = tmp_path / "c"
    _trunk(croot / "TR-H", "A", 0, world["tm"], 1)
    _trunk(croot / "TR-H", "D", 3, world["tm"], 4, sham=False)
    with pytest.raises(ValueError, match="trunk D has an S2 probe"):
        _build(world, croot)
    croot2 = tmp_path / "c2"
    _trunk(croot2 / "TR-H", "A", 0, world["tm"], 1)
    _set_manifest(croot2 / "TR-H" / "S2-A", branch_period=None)
    with pytest.raises(ValueError, match="no deviation period"):
        _build(world, croot2)
    croot3 = tmp_path / "c3"
    _trunk(croot3 / "TR-H", "A", 0, world["tm"], 1)
    _drop_slot0(croot3 / "TR-H" / "S2-A")
    with pytest.raises(ValueError, match="cannot be scored"):
        _build(world, croot3)


def test_placebo_trunk_without_its_own_sham_or_branch_period_raises(world, tmp_path) -> None:
    croot = tmp_path / "c"
    _trunk(croot / "TR-H", "A", 0, world["tm"], 1)
    _trunk(croot / "TR-H", "E", 4, world["tm"], 5, sham=False, probe=False)
    with pytest.raises(ValueError, match="trunk E has a sham2 placebo"):
        _build(world, croot)
    # an unqualified trunk is not a placebo trunk: the build goes through
    corpus, meta = _build(world, croot, qualified={"A"})
    assert corpus["trunk_id"].to_list() == ["A"] and meta["placebo"]["TR-H:sham"]["n"] == 1
    croot2 = tmp_path / "c2"
    _trunk(croot2 / "TR-H", "A", 0, world["tm"], 1)
    _trunk(croot2 / "TR-H", "C", 2, world["tm"], 3, probe=False)
    for run in ("sham-C", "sham2-C"):
        _set_manifest(croot2 / "TR-H" / run, branch_period=None)
    with pytest.raises(ValueError, match="placebo trunk C has no branch period"):
        _build(world, croot2)
    croot3 = tmp_path / "c3"
    _trunk(croot3 / "TR-H", "A", 0, world["tm"], 1)
    _drop_slot0(croot3 / "TR-H" / "sham2-A")
    with pytest.raises(ValueError, match="placebo cut of trunk A failed"):
        _build(world, croot3)


def test_empty_placebo_set_raises_instead_of_the_v1_cut(world, tmp_path) -> None:
    croot = tmp_path / "c"
    _trunk(croot / "TR-H", "A", 0, world["tm"], 1, sham2=False)
    with pytest.raises(ValueError, match="empty placebo set"):
        _build(world, croot)
    _trunk(croot / "TR-H", "B", 1, world["tm"], 2)  # B carries a sham2
    with pytest.raises(ValueError, match="empty placebo set"):
        _build(world, croot, spec=_sham_spec(placebo=None))  # no placebo arm (TR-M today)
    for qualified in (set(), {"A"}):  # the qualified trunks hold no sham2
        with pytest.raises(ValueError, match="empty placebo set"):
            _build(world, croot, qualified=qualified)


def test_sham_window_without_ghosts_of_its_seed_group_raises(world, tmp_path) -> None:
    croot = tmp_path / "c"
    _trunk(croot / "TR-H", "A", 0, world["tm"], 1)
    _trunk(croot / "TR-H", "F", 5, world["tm"], 6, ghosts=0)
    with pytest.raises(ValueError, match="sham window of trunk F .*no ghost sub-run"):
        _build(world, croot)
    with pytest.raises(ValueError, match="no ghost sub-run"):
        _build(world, croot, spec=_sham_spec(ghosts=None))


def test_run_without_a_manifest_seed_raises(world, tmp_path) -> None:
    croot = tmp_path / "c"
    _trunk(croot / "TR-H", "G", None, world["tm"], 7)
    with pytest.raises(ValueError, match="trunk G.*manifest seed"):
        _build(world, croot)


def _tiny_corpus(n_groups: int = 6) -> pl.DataFrame:
    rows = []
    for k in range(n_groups):
        for kind in ("llm_punished" if k < 2 else "llm_unpunished", "ghost", "instruction"):
            rows.append(
                {
                    "window_id": f"{kind}-{k}",
                    "run_id": f"{kind}-r{k}",
                    "trunk_id": f"T{k}",
                    "label": DT.process_label(kind),
                    "kind": kind,
                    **{f: 0.1 for f in DT.FEATURES},
                    "seed_group": DT.seed_group_id("H", k),
                    "lane": "H",
                }
            )
    return pl.DataFrame(rows, schema=DT.CORPUS_SCHEMA)


def test_freeze_refuses_null_seed_groups_and_groups_in_both_halves(
    world, tmp_path, monkeypatch
) -> None:
    cs = DX.load_corpus_spec(world["corpus_path"])
    rules, out = tmp_path / "rules.json", tmp_path / "det"
    corpus = _tiny_corpus()
    holed = corpus.with_columns(
        pl.when(pl.col("window_id") == "ghost-3")
        .then(None)
        .otherwise(pl.col("seed_group"))
        .alias("seed_group")
    )
    with pytest.raises(ValueError, match="without a seed_group"):
        DX.freeze(holed, cs, rules, out, world["corpus_path"])
    # the freeze re-checks the halves it receives: no trunk fallback, no shared seed group
    monkeypatch.setattr(DT, "split_half", lambda c, seed: (holed, holed.head(0)))
    with pytest.raises(ValueError, match="every window needs a seed_group"):
        DX.freeze(corpus, cs, rules, out, world["corpus_path"])
    monkeypatch.setattr(
        DT, "split_half", lambda c, seed: (c, c.filter(pl.col("seed_group") == "H:0"))
    )
    with pytest.raises(ValueError, match=r"both halves of the split: \['H:0'\]"):
        DX.freeze(corpus, cs, rules, out, world["corpus_path"])
    assert not rules.exists() and not out.exists()


def test_fpr_null_exploratory_rates_on_the_ghost_windows(world, tmp_path: Path) -> None:
    cs = DX.load_corpus_spec(world["corpus_path"])
    res = DX.run_detect(
        "fpr-null",
        world["croot"],
        world["corpus_path"],
        tmp_path / "r.json",
        world["arms"],
        ROOT,
        tmp_path / "out",
        fpr_shifts=4,
    )
    full = json.loads((tmp_path / "out" / "fpr_null.json").read_text(encoding="utf-8"))
    assert res["n_windows"] == full["n_windows"] == 4 * N_TRUNKS  # 4 ghosts per trunk
    assert full["status"].startswith("exploratory") and full["level"] == 0.05
    for k in ("plain_shift", "ghost_null"):
        r = full[k]
        assert r["n"] == full["n_windows"] and 0 <= r["flagged"] <= r["n"]
        assert r["fpr"] == pytest.approx(r["flagged"] / r["n"])
    row = full["rows"][0]
    assert {"run_id", "trunk_id", "live_slot", "te", "plain_q", "d5net", "ghost_q"} <= set(row)
    assert cs.seed == 20261010
