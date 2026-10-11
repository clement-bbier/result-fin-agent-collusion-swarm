"""Run the pre-registered analysis (registered arms, pinned benchmarks, revised SAP).

Usage::

    # 1. build the pinned benchmarks.json from the trunk/lever configs and re-pin its sha256
    uv run python scripts/analyze.py --make-benchmarks [--arms configs/analysis/registered_arms.yaml]
    # 2. market-quality cost per lever from the scripted MQC campaigns (0 LLM calls)
    uv run python scripts/analyze.py --mqc-only results/campaigns/SC-1
    # 3. the registered analysis of a campaign (only registered arms are read)
    uv run python scripts/analyze.py results/campaigns/TR-H [--mqc results/analysis/SC-1/mqc.json]
        [--mqc-campaign results/campaigns/SC-1] [--out-root results/analysis]
    # 3b. the same, BLIND (PREREG section 5): lever arms as letters, key kept apart
    uv run python scripts/analyze.py results/campaigns/TR-H --blind [--key-root DIR]
        [--controls sham,sham2,trunk]
    # 4. map a committed blinded stats.json back to the arm names
    uv run python scripts/analyze.py --unblind results/analysis/TR-H/stats.json
        --key results/analysis/_blind_keys/TR-H.key.json

Step 3 writes ``results/analysis/<campaign>/{summary.md, stats.json, exclusions.json}``. It
refuses to run when a pinned benchmarks file does not match its sha256 or when the configs
changed after pinning (``--allow-stale-benchmarks`` overrides the latter, recorded in
``stats.json``). With the public fringe on, ``--make-benchmarks`` uses
``finswarm.market.sweep`` (numerical sweep) when available. With ``--blind`` the open results
never touch the output folder: the letter key goes to ``<key-root>/_blind_keys/`` and only
its SHA-256 is recorded in ``stats.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_ARMS = Path("configs/analysis/registered_arms.yaml")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("campaign_dir", type=Path, nargs="?")
    p.add_argument("--arms", type=Path, default=DEFAULT_ARMS, help="registered arms YAML")
    p.add_argument("--root", type=Path, default=Path("."), help="repository root")
    p.add_argument("--out-root", type=Path, default=Path("results/analysis"))
    p.add_argument("--mqc", type=Path, default=None, help="mqc.json from --mqc-only")
    p.add_argument("--mqc-campaign", type=Path, action="append", default=None)
    p.add_argument("--B", type=int, default=None, help="bootstrap resamples (registry default)")
    p.add_argument("--seed", type=int, default=None, help="bootstrap seed (registry default)")
    p.add_argument("--n-perm", type=int, default=200, help="D7 permutation draws")
    p.add_argument("--allow-stale-benchmarks", action="store_true")
    p.add_argument(
        "--window-shift",
        type=int,
        default=0,
        help="trend-rule shift of the trunk/branch windows (+20 after an EXTEND; checked "
        "against --gates-json when given)",
    )
    p.add_argument("--make-benchmarks", action="store_true", help="build and re-pin benchmarks")
    p.add_argument("--no-repin", action="store_true", help="with --make-benchmarks: print only")
    p.add_argument("--sweep-periods", type=int, default=50, help="periods per sweep evaluation")
    p.add_argument("--sweep-replications", type=int, default=10, help="seeds per evaluation")
    p.add_argument("--sweep-starts", type=int, default=10, help="best-response starts")
    p.add_argument("--mqc-only", type=Path, action="append", default=None, metavar="DIR")
    blind = p.add_argument_group("blind analysis (PREREG section 5)")
    blind.add_argument("--blind", action="store_true", help="lever arms as letters")
    blind.add_argument(
        "--key-root",
        type=Path,
        default=None,
        help="where _blind_keys/<campaign>.key.json goes (default: --out-root)",
    )
    blind.add_argument(
        "--controls",
        default="sham,sham2,trunk",
        help="comma-separated arms that keep their names in blind mode",
    )
    blind.add_argument(
        "--unblind", type=Path, default=None, metavar="STATS", help="blinded stats.json to map back"
    )
    blind.add_argument("--key", type=Path, default=None, help="key file for --unblind")
    gates = p.add_argument_group("gate evaluation (PREREG section 1, D-16)")
    gates.add_argument("--gates", action="store_true", help="evaluate Gate 1 / 1b / 2 and AC-H")
    gates.add_argument("--campaigns-root", type=Path, default=Path("results/campaigns"))
    gates.add_argument(
        "--gate1", default="PC-H,PC-M", help="Gate 1 campaigns, one per lane ('none' to skip)"
    )
    gates.add_argument("--gate1b", default="PC-H-dealer", help="Gate 1b campaigns ('none' to skip)")
    gates.add_argument(
        "--gate2",
        default="TR-H",
        help="Gate 2 trunk campaigns, comma-separated, one per lane: the lane (and so the "
        "trunk arm, e.g. TR-S -> trunk_S) is read from the registry ('none' to skip)",
    )
    gates.add_argument(
        "--anchoring", default="AC-H", help="AC-H diagnostic campaign ('none' to skip)"
    )
    gates.add_argument("--gate2-lane", default=None, help="explicit lane for the --gate2 campaigns")
    gates.add_argument(
        "--gate1-lane",
        default=None,
        help="explicit lane(s) for the --gate1 campaigns (one, or one per campaign)",
    )
    gates.add_argument("--gates-out", type=Path, default=None, help="default: <out-root>/gates")
    det = p.add_argument_group("detection corpus, rule freeze and single evaluation (A2, D-21)")
    det.add_argument(
        "--detect",
        choices=["build", "split", "freeze", "evaluate", "stub", "fpr-null"],
        default=None,
        help="build the labelled corpus, freeze the rules, evaluate once, or write the stub",
    )
    det.add_argument("--corpus-spec", type=Path, default=Path("configs/detection/corpus.yaml"))
    det.add_argument("--rules", type=Path, default=Path("configs/detection/frozen_rules.json"))
    det.add_argument(
        "--fpr-shifts",
        type=int,
        default=None,
        help="--detect fpr-null: at most this many cyclic shifts per window (default all)",
    )
    det.add_argument(
        "--qualified-from",
        type=Path,
        default=None,
        help="stats.json whose inference.qualification.qualified restricts the placebo trunks "
        "(M6); default <out-root>/TR-H/stats.json when it exists",
    )
    det.add_argument("--detect-out", type=Path, default=None, help="default: <out-root>/detection")
    det.add_argument("--n-boot", type=int, default=2000, help="AUC bootstrap draws")
    fam = p.add_argument_group("family S (PREREG section 3, D-20)")
    fam.add_argument(
        "--family-s", action="store_true", help="Holm over H2, H4, H5, H6, H7, H8, H10"
    )
    fam.add_argument(
        "--stats", type=Path, default=None, help="TR-H stats.json (registered analysis)"
    )
    fam.add_argument("--gradient-stats", type=Path, default=None, help="GR-H stats.json")
    fam.add_argument("--corpus", type=Path, default=None, help="detection corpus.parquet (D5-net)")
    fam.add_argument("--probe-flags", type=Path, default=None, help="detection corpus_meta.json")
    fam.add_argument(
        "--sting-campaigns", default="", help="comma-separated campaign dirs holding S1 / S1c runs"
    )
    fam.add_argument("--h2-perm", type=int, default=10_000, help="H2 permutation draws")
    fam.add_argument(
        "--ghost-null",
        type=Path,
        default=None,
        help="ghost cyclic-shift null of the sham D5-net for the H6 guard (B3): the detection "
        "corpus_meta.json (ghost_null.sham_pooled) or a JSON list / {draws} / {q95}; "
        "without it H6 is never supported",
    )
    fam.add_argument(
        "--r1-replacement-period",
        type=int,
        default=None,
        help="first period played by the R1 replacement (default: the registry's T0)",
    )
    num = p.add_argument_group("paper/numbers.json producer (SAP section 12)")
    num.add_argument("--numbers", action="store_true", help="write the verify rows of every claim")
    num.add_argument("--claims", type=Path, default=Path("configs/verify/claims.yaml"))
    num.add_argument("--numbers-out", type=Path, default=Path("paper/numbers.json"))
    cv = p.add_argument_group("corpus v1 labelling sheet (DETECTION_PROTOCOL section 5)")
    cv.add_argument(
        "--corpus-v1",
        choices=["sample", "validate", "kappa"],
        default=None,
        help="sample the blind sheet (dev-half seed groups), validate a filled sheet, kappa",
    )
    cv.add_argument(
        "--d9-scores",
        type=Path,
        default=None,
        help="cached D9 scores (parquet/csv): sample draws among scored items; omitted = sampler-first (writes score_request.csv); kappa joins them",
    )
    cv.add_argument("--key-out", type=Path, default=None, help="sealed key path OUTSIDE the repo")
    cv.add_argument("--sheet", type=Path, default=None, help="filled labelling_sheet.csv")
    cv.add_argument("--sealed-key", type=Path, default=None, help="the sealed key (for kappa)")
    cv.add_argument("--n-items", type=int, default=60)
    cv.add_argument("--sampler-seed", type=int, default=20261011)
    cv.add_argument(
        "--cap-family",
        type=int,
        default=10,
        help="max items per seed group (trunk family); 6 dev groups x 10 = 60 items",
    )
    cv.add_argument("--salt", default=None, help="item-id salt (default random; key only)")
    cv.add_argument(
        "--corpus-campaigns",
        default=None,
        help="comma-separated campaign names under --campaigns-root (default: every TR-H* "
        "and CA-H* campaign: trunks, branches, ghosts, probes and CA of the Haiku lane)",
    )
    cv.add_argument("--corpus-lane", default="H", help="lane of the runs to sample (H)")
    cv.add_argument(
        "--detect-corpus",
        type=Path,
        default=None,
        help="corpus.parquet of --detect build (S2-punished trunks); default "
        "<out-root>/detection/corpus.parquet when it exists",
    )
    cv.add_argument(
        "--gates-json",
        type=Path,
        default=None,
        help="gates.json of --gates: EXTEND shifts the windows by the trend rule; default "
        "<out-root>/gates/gates.json when it exists",
    )
    blind.add_argument(
        "--unblind-out", type=Path, default=None, help="default: <stem>.unblinded.json"
    )
    return p


def make_benchmarks(args: argparse.Namespace) -> int:
    from finswarm.analysis.arms import load_registered_arms
    from finswarm.analysis.benchmarks_file import repin, write_benchmark_file

    spec = load_registered_arms(args.arms)
    for pin, bf in spec.benchmark_files.items():
        kw = {
            "periods": args.sweep_periods,
            "replications": args.sweep_replications,
            "n_starts": args.sweep_starts,
        }
        sha = write_benchmark_file(spec, pin, args.root, kw)
        print(f"{pin}: {bf.file} sha256 {sha}")
        if not args.no_repin:
            if repin(args.arms, pin, bf.file, sha):
                print(f"  re-pinned in {args.arms}")
            else:
                print(f"  could not re-pin {pin} in {args.arms}; edit sha256 by hand")
                return 1
    return 0


def mqc_only(args: argparse.Namespace) -> int:
    from finswarm.analysis.arms import load_registered_arms
    from finswarm.analysis.campaign import _clean
    from finswarm.analysis.mqc import analyze_mqc

    spec = load_registered_arms(args.arms)
    kw = {k: v for k, v in (("B", args.B), ("seed", args.seed)) if v is not None}
    doc = analyze_mqc(args.mqc_only, spec, **kw)
    out = args.out_root / args.mqc_only[0].name / "mqc.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_clean(doc), indent=2), encoding="utf-8")
    for lev, v in doc["levers"].items():
        print(f"{lev:>4}: {v.get('status')} n={v.get('n')} mean={v.get('mean')}")
    print(f"-> {out}")
    return 0


def unblind(args: argparse.Namespace) -> int:
    from finswarm.analysis.blind import unblind_stats

    if args.key is None:
        print("error: --unblind needs --key <_blind_keys/<campaign>.key.json>", file=sys.stderr)
        return 2
    opened = unblind_stats(args.unblind, args.key, args.unblind_out)
    dest = args.unblind_out or args.unblind.with_name(f"{args.unblind.stem}.unblinded.json")
    print(f"unblinded {len(opened['blind']['mapping'])} arms -> {dest}")
    return 0


def run_gates(args: argparse.Namespace) -> int:
    from finswarm.analysis.gates import evaluate_gates, print_report, write_report

    def names(s: str | None) -> list[str]:
        if s is None or not s.strip() or s.strip().lower() == "none":
            return []
        return [x.strip() for x in s.split(",") if x.strip()]

    lanes: dict[str, str] = {}
    if args.gate2_lane:
        lanes.update({c: args.gate2_lane for c in names(args.gate2)})
    if args.gate1_lane:
        g1, ls = names(args.gate1), names(args.gate1_lane)
        if len(ls) not in (1, len(g1)):
            print("error: --gate1-lane needs one lane or one per --gate1 campaign", file=sys.stderr)
            return 2
        lanes.update(dict(zip(g1, ls * len(g1) if len(ls) == 1 else ls, strict=False)))
    kw = {k: v for k, v in (("B", args.B), ("seed", args.seed)) if v is not None}
    report = evaluate_gates(
        args.campaigns_root,
        gate1_campaigns=names(args.gate1),
        gate1b_campaigns=names(args.gate1b),
        gate2_campaign=names(args.gate2),
        anchoring_campaign=",".join(names(args.anchoring)) or None,
        arms_path=args.arms,
        root=args.root,
        lanes=lanes,
        **kw,
    )
    print_report(report)
    out = write_report(report, args.gates_out or (args.out_root / "gates"))
    print(f"-> {out / 'gates.json'}, {out / 'gates.md'}")
    return int(report["exit_code"])


def run_detect_cli(args: argparse.Namespace) -> int:
    from finswarm.analysis.detect import run_detect

    qualified = None
    qpath = args.qualified_from or (args.out_root / "TR-H" / "stats.json")
    if args.detect == "build" and Path(qpath).exists():
        st = json.loads(Path(qpath).read_text(encoding="utf-8"))
        q = (st.get("inference") or {}).get("qualification") or {}
        qualified = set(q.get("qualified") or [])
        print(f"detect build: placebos restricted to {len(qualified)} qualified trunks ({qpath})")
    elif args.detect == "build":
        print("detect build: no stats.json for the qualified filter (every trunk used)")
    res = run_detect(
        args.detect,
        campaigns_root=args.campaigns_root,
        corpus_path=args.corpus_spec,
        rules_path=args.rules,
        arms_path=args.arms,
        root=args.root,
        out_dir=args.detect_out or (args.out_root / "detection"),
        n_boot=args.n_boot,
        qualified=qualified,
        fpr_shifts=args.fpr_shifts,
    )
    shown = {
        k: v
        for k, v in res.items()
        if k in ("n_windows", "kinds", "path", "rules_sha256", "n_dev", "n_eval")
    }
    print(f"detect {args.detect}: {json.dumps(shown, default=str)}")
    return 0


def _ghost_null_input(path: Path | None):
    """The H6 guard null from ``--ghost-null``: the detection ``corpus_meta.json`` (its
    ``ghost_null.sham_pooled`` block, null of the MEAN sham D5-net over the qualified trunks
    of lane H: the draws when present, else its q95), a bare list of draws, ``{"draws"}`` or
    ``{"q95"}``."""
    if path is None:
        return None
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, dict) and "ghost_null" in doc:
        doc = (doc["ghost_null"] or {}).get("sham_pooled") or {}
    if isinstance(doc, list):
        return [float(x) for x in doc]
    if isinstance(doc, dict):
        for key in ("mean_d5net_draws", "d5net_draws", "draws"):
            if doc.get(key):
                return [float(x) for x in doc[key]]
        for key in ("mean_q95", "q95"):
            if doc.get(key) is not None:
                return float(doc[key])
    raise ValueError(f"{path}: no ghost-null draws or q95 found")


def run_family_s(args: argparse.Namespace) -> int:
    from finswarm.analysis.campaign import _clean
    from finswarm.analysis.family_s import family_s_from_stats, family_s_md

    if args.stats is None or not args.stats.exists():
        print("error: --family-s needs --stats <TR-H stats.json>", file=sys.stderr)
        return 2
    res = family_s_from_stats(
        args.stats,
        args.gradient_stats,
        args.corpus,
        [Path(x.strip()) for x in args.sting_campaigns.split(",") if x.strip()],
        args.probe_flags,
        n_perm=args.h2_perm,
        seed=args.seed if args.seed is not None else 20261010,
        sham_d5net_ghost_null=_ghost_null_input(args.ghost_null),
        r1_replacement_period=args.r1_replacement_period,
        shift=args.window_shift or None,
        gates=args.gates_json if args.gates_json and Path(args.gates_json).exists() else None,
    )
    out = args.out_root / "family_s"
    out.mkdir(parents=True, exist_ok=True)
    (out / "family_s.json").write_text(json.dumps(_clean(res), indent=2), encoding="utf-8")
    (out / "family_s.md").write_text(family_s_md(_clean(res)), encoding="utf-8")
    for h in res["family"]:
        v = res["hypotheses"][h]
        print(
            f"{h:<11} p={v.get('p', float('nan')):.4f} p_holm={v['p_holm']:.4f} "
            f"reject={v['reject']} supported={v.get('supported')} {v['status']}"
        )
    print(f"-> {out}")
    return 0


def run_numbers(args: argparse.Namespace) -> int:
    from finswarm.analysis.numbers import build_numbers, write_numbers

    doc = build_numbers(args.claims, args.out_root, args.campaigns_root, args.arms, args.root)
    path = write_numbers(doc, args.numbers_out)
    m = doc["_meta"]
    print(f"numbers: {m['filled']} expected values filled, {m['pending']} pending -> {path}")
    return 0


def run_corpus_v1(args: argparse.Namespace) -> int:
    from finswarm.analysis.corpus_v1 import (
        build_corpus_v1,
        default_campaign_dirs,
        kappa_from_sheet,
        validate_sheet,
    )

    if args.corpus_v1 == "sample":
        if args.key_out is None:
            print("error: --corpus-v1 sample needs --key-out", file=sys.stderr)
            return 2
        if args.corpus_campaigns:
            dirs = [
                args.campaigns_root / c.strip()
                for c in args.corpus_campaigns.split(",")
                if c.strip()
            ]
        else:
            dirs = default_campaign_dirs(args.campaigns_root)
        if not dirs:
            print(f"error: no TR-H*/CA-H* campaign under {args.campaigns_root}", file=sys.stderr)
            return 2
        detect = args.detect_corpus or (args.out_root / "detection" / "corpus.parquet")
        gates = args.gates_json or (args.out_root / "gates" / "gates.json")
        res = build_corpus_v1(
            dirs,
            args.d9_scores,
            args.rules,
            args.out_root / "corpus_v1",
            args.key_out,
            args.root,
            args.n_items,
            args.sampler_seed,
            detect_corpus=detect if Path(detect).exists() else None,
            gates_json=gates if Path(gates).exists() else None,
            lane=args.corpus_lane,
            salt=args.salt,
            cap_family=args.cap_family,
        )
        print(
            f"corpus v1: {res['n_sampled']}/{res['n_requested']} items from "
            f"{res['n_candidates']} candidates (lane {res['lane']}, development half only); "
            f"by stratum {res['n_by_stratum']}; true positives {res['n_positive_true']}; "
            f"shortfall {res['shortfall'] or 'none'}; windows {res['windows']} "
            f"({res['windows_source']['source']}, shift {res['windows_source']['shift']}); "
            f"sheet {res['sheet']} sha256 {res['sheet_sha256']}; key {res['key']} (outside the "
            f"repo) sha256 {res['key_sha256']}"
        )
        if res.get("note"):
            print("note:", res["note"])
        if res.get("score_request"):
            print(f"sampler-first: score these decisions with D9 -> {res['score_request']}")
        return 0
    if args.corpus_v1 == "validate":
        if args.sheet is None:
            print("error: --corpus-v1 validate needs --sheet", file=sys.stderr)
            return 2
        v = validate_sheet(args.sheet)
        print(json.dumps(v, indent=2))
        return 0 if v["ok"] else 1
    if args.sheet is None or args.sealed_key is None:
        print("error: --corpus-v1 kappa needs --sheet and --sealed-key", file=sys.stderr)
        return 2
    print(
        json.dumps(
            kappa_from_sheet(args.sheet, args.sealed_key, scores_path=args.d9_scores), indent=2
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.gates:
        return run_gates(args)
    if args.corpus_v1:
        return run_corpus_v1(args)
    if args.numbers:
        return run_numbers(args)
    if args.family_s:
        return run_family_s(args)
    if args.detect:
        return run_detect_cli(args)
    if args.make_benchmarks:
        return make_benchmarks(args)
    if args.mqc_only:
        return mqc_only(args)
    if args.unblind is not None:
        return unblind(args)
    if args.campaign_dir is None or not args.campaign_dir.is_dir():
        print(f"error: {args.campaign_dir} is not a directory", file=sys.stderr)
        return 2
    from finswarm.analysis.arms import PinError

    kw = {
        "arms_path": args.arms,
        "root": args.root,
        "mqc": args.mqc,
        "mqc_campaigns": args.mqc_campaign,
        "B": args.B,
        "seed": args.seed,
        "allow_stale_benchmarks": args.allow_stale_benchmarks,
        "n_perm": args.n_perm,
        "gates_json": args.gates_json,
        "window_shift": args.window_shift,
    }
    out = args.out_root / args.campaign_dir.name
    try:
        if args.blind:
            from finswarm.analysis.blind import analyze_blind

            controls = tuple(c.strip() for c in args.controls.split(",") if c.strip())
            res = analyze_blind(
                args.campaign_dir,
                out_root=args.out_root,
                key_root=args.key_root,
                controls=controls,
                **kw,
            )
            b = res["blind"]
            print(
                f"analysed {res['n_runs']} runs ({res['n_excluded']} excluded, "
                f"{res['n_ignored']} unregistered ignored) -> {out} [BLIND: "
                f"{len(b['letters'])} arms as letters, key {b['key_path']}]"
            )
            return 0
        from finswarm.analysis.campaign import analyze_campaign

        res = analyze_campaign(args.campaign_dir, out_root=args.out_root, **kw)
    except PinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except ValueError as exc:  # registered-path guards (m9 MQC, m12 window shift)
        print(f"error: {exc}", file=sys.stderr)
        return 4
    print(
        f"analysed {res['n_runs']} runs ({res['n_excluded']} excluded, "
        f"{res['n_ignored']} unregistered ignored) -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
