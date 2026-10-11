# Amendments to the pre-registration

Rule: after the freeze of `PREREG.md` (hash recorded in the README and in `FROZEN`), every change to hypotheses, thresholds, windows, exclusions, lever parameters or analysis code that affects a registered analysis is recorded here **before** the affected data are analysed whenever possible. Entries are append-only.

Required fields per entry:

| Field | Content |
| --- | --- |
| ID | A-NNN (sequential) |
| UTC timestamp | ISO 8601, e.g. 2026-10-10T14:03:00Z |
| Commit | Git SHA of the commit containing the change |
| File and section changed | e.g. `PREREG.md` section 2 |
| What changed | Old value -> new value |
| Why | Reason (data-independent if possible; state whether data had been seen) |
| Data seen before the change? | yes / no (which) |
| Affected analysis becomes exploratory? | yes / no (if yes, label in paper) |

## Entries

Freeze stage 1 occurred at 2026-10-10T08:40:57Z (commit 2bb977e hashed, freeze commit 1b02071, tag `prereg-freeze`, PREREG.sha256 7f90d06b9ec9e24b4af0e6a103f6e8720fd4f7bb73c5bd3bf821c87737e4955b, 90 files).

### Gate reads (A-000 lines, appended before each gate output is read)
- A-000-1 — 2026-10-10T09:46Z — Gate 1 (H lane; PC-M incomplete, excluded), Gate 1b, Gate 2 and AC-H read on main aadabe9c4ff812c9f537f8551ba19bfaad079a8e with `uv run python scripts/analyze.py --gates` (defaults). Inputs: PC-H 24/24 ok (300 periods), PC-H-dealer 8/8 ok, TR-H 12/12 ok (100 periods), AC-H 4/4 ok; PC-S 5/5 ok but not read by this gate; PC-M 0/24 (relaunched 11:37 Paris). Analysis code at that sha is not yet hashed (stage 2 pending); recorded by ops before the read.
- A-000-2 — 2026-10-10T09:52Z — Gate 1 (S lane, PC-S, 5/5 runs ok, 300 periods) read on main 95205ad07e89d32fa514e12bff8f91d831957189 with `uv run python scripts/analyze.py --gates --gate1 PC-S --gate1b PC-H-dealer --gate2 TR-H --anchoring AC-H`; recorded by ops before the read.
- A-000-3 — 2026-10-10T10:15Z — Gate 2 (S lane, TR-S, 8/8 trunks ok, 100 periods) read on main 1234c57f7977accf05fbcc8dca1d59f6c085fdc0 with `uv run python scripts/analyze.py --gates --gate1 PC-S --gate1b PC-H-dealer --gate2 TR-S --anchoring AC-H --gates-out results/analysis/gates_S/gate2_TR-S` (PC-S re-read identically because the CLI cannot skip Gate 1); recorded by ops before the read. SH-S runs on all 8 TR-S trunks as configured; unqualified trunks are excluded by the registered analysis.
- A-000-4 — 2026-10-10T12:48Z — Gate 2 (S lane, TR-S) re-read on main ca05d8b after the lane-inference fix: NOT RUN (every trunk "incomplete, 29 of 30 window periods": the window convention mismatch, see A-008); recorded by ops before the read.
- A-000-5 — 2026-10-10T13:14Z — Gate 2 re-read on BOTH lanes on the registered window 71-100 (engine 70..99) on main 9cca349234a9 after A-008: `uv run python scripts/analyze.py --gates --gate1 none --gate2 TR-H,TR-S --gates-out results/analysis/gates/rereread_A-008`; recorded by ops before the read. The 09:46Z H-lane read (engine 71..99) is superseded by this one.
- Gate 1 (M lane): NOT RUN (PC-M dropped, A-005).
- A-000-6 — 2026-10-10T14:35Z — blind Tier 0 pass and H1 / H1-R reads on main 14b6bf4 (science; blind key ../finswarm-sealed sha dd44c47b).
- A-000-7 — 2026-10-10T15:50Z — family S final rerun, H-S and H1-M on main 2fb0de3 (after A-016).
- A-000-8 — 2026-10-10T16:15Z — H8 rerun on main cfc0159 (after A-018).

### A-001
- UTC timestamp: 2026-10-10T13:20:00Z
- Commit: the stage-2 amendment commit (`freeze.py --amend --amendment A-001 --stage 2`)
- File and section changed: the hashed set gains `src/finswarm/analysis/**` and `src/finswarm/metrics/**` (stage 2, PREREG section 5); no text changes
- What changed: the registered analysis code is frozen after the red-team fixes (B1-B3, M4, M6, M7, m9, m11, m12), the A-003 L2 rule, the A-008 period convention and the MQC document (results/analysis/SC-1/mqc.json, A-007). Gate 2 was re-read on both lanes on the registered window (A-000-5): H lane PASS 11/12 (seeds 0-7, 9-11), S lane PASS 8/8, trend rule not fired. No lever, sham or ghost contrast has been computed before this hash.
- Why: PREREG section 5 (two-stage freeze; stage 2 before any lever contrast is read).
- Data seen before the change?: instrument campaigns (gates), scripted CPU controls (MQC); no LLM lever, sham or ghost outcome.
- Affected analysis becomes exploratory?: no.

### A-002
- UTC timestamp: 2026-10-10T09:22:59Z (commit af5ec0e)
- Commit: the commit adding this entry (main, after 910520c)
- File and section changed: `campaign_table.md` budget proof and Sonnet truncation order (D-13); no hashed file edited: LT-S is rule-resolved and SH2-S is simply not launched
- What changed: Sonnet arm truncation steps 2 and 3 applied before TR-S launches: LT-S runs L2 + one rule lever (not three); SH2-S is not run (the Sonnet arm has no second sham; sigma_d is taken from the Haiku arm and stated as such). Step 1 (GH-S to Tier 2) was applied before the freeze.
- Why: measured costs at 11:20 Paris: PC-S (Fish template, 300 periods) at 0.01425 USD per call (1,095 output tokens per call on average) against a sizing of 0.00812; Haiku dealer_v1 within +13 % of sizing; the projected total without truncation exceeded the 200 USD cap (about 215 USD); with steps 2-3 the projection is about 180 USD.
- Data seen before the change?: yes, cost ledger only (USD per call, token counts, completion status); no outcome metric (Delta, prices, profits) of any registered arm was read.
- Affected analysis becomes exploratory?: no (the truncation order was registered in D-13); H-S keeps its primary L2 contrast; the Sonnet frontier loses one lever and its sham2 noise band (reported as a limitation).


## Template

```
### A-001
- UTC timestamp:
- Commit:
- File and section changed:
- What changed:
- Why:
- Data seen before the change?:
- Affected analysis becomes exploratory?:
```

### A-003
- UTC timestamp: 2026-10-10T10:27:10Z (commit 0555d6e)
- Commit: the commit adding this entry (main, after 52a1a3c)
- File and section changed: `PREREG.md` D-17 (dual-benchmark criterion) as applied to lever L2; `SAP.md` estimators; implementation in `src/finswarm/analysis/campaign.py` (stage 2)
- What changed: the pinned own benchmarks of the L2 market (sweep, R = 10: p_N' = 1.3375, p_M' = 2.05) coincide with the trunk benchmarks (1.3315, 2.0544) within 0.005 in the price range [1.5, 2.0] (the public fringe at share 0.3 barely moves the dealers' one-shot equilibrium; the sweep grid step is 0.05 and the Nash points split between 1.30 and 1.35 across starts). The own-scoring co-condition is therefore uninformative for L2. For L2 only, the label "collusion reduction" requires (a) the trunk-scored one-sided test to be significant (Holm-adjusted in H1-R, raw in H1 / H1-M / H-S) AND (b) the LLM incumbents' mean ask fall under L2 (lever minus sham, periods 111-150) to exceed twice the SC1 best-responders' mean ask fall under the same lever (MQC-L2 versus the SC-1 sham, CPU, 20 seeds, computed before any L2 LLM data are read); otherwise the result is labelled "price reduction consistent with competitive adjustment". The own-scoring test is still computed and reported for L2, and max |d_own - d_trunk| is reported for every lever. Other levers keep the D-17 rule unchanged.
- Why: found by the read-only red-team of the analysis code before the stage-2 hash (finding 5); the factor 2 is set now, before data; a finer re-pin of the L2 sweep (grid <= 0.01, more starts) would cost about an hour of CPU during the campaigns and would not change the conclusion that the two benchmark pairs are numerically close.
- Data seen before the change?: no outcome data of any lever or sham branch (L2-H was running; nothing analysed); only the pinned benchmark file and the gate outputs of the instrument campaigns (A-000-1..3).
- Affected analysis becomes exploratory?: no; H1 keeps its registered test; only the labelling rule for L2 changes, pre-data.

### A-004
- UTC timestamp: 2026-10-10T10:31:56Z (config commit 697a340; hash commit 35b274c at 10:32:27Z)
- Commit: 697a340 (config change) and 35b274c (amended hash, `freeze.py --amend --amendment A-004`). PC-M relaunched from scratch at 12:35 Paris; its manifests record sha 35b274c (correction 2026-10-10T12:05Z, from the manifests themselves). PC-M was then stopped and dropped (see A-005).
- File and section changed: `llm.params.max_tokens` in every M-lane config under configs/sprint (PC-M, TR-M, TR-M-ext, SH-M, LT-M, CA-M, GH-M, S2-M, R3-H newcomer, SENS-TEMP05-M); PREREG section 0 Mistral protocol line
- What changed: max_tokens 600 -> 2048 for the Mistral lane (ministral-8b-latest). The Anthropic lanes already use 2048.
- Why: instrument failure found by ops in the ledger during PC-M (11:37-12:40 Paris): ministral-8b hits the 600-token output cap on almost every decision (the registered JSON output with PLANS and INSIGHTS does not fit), each of the three repairs is truncated again, so about 4 provider calls per decision and parse failures on most decisions; Gate 1 (M lane) would fail on the parse-failure criterion for a mechanical reason. The truncated PC-M runs (24 runs, about 38 periods each; two resumed runs about 108) are discarded (the parameter change gives new run ids) and the episode is reported in the paper's "what did not work".
- Data seen before the change?: ledger rows only (status, attempts, output_tokens = 600, latency, call counts per decision); no price or profit outcome of any M-lane run was read.
- Affected analysis becomes exploratory?: no (the M lane keeps its registered role; the cost is time on a free lane: PC-M restarts from scratch).

### A-006
- UTC timestamp: 2026-10-10T11:45:00Z
- Commit: the commit adding this entry, applied with `freeze.py --amend --amendment A-006`
- File and section changed: `PREREG.md` section 2, "Differentiation from the closest work" (Garra 2026 sentence)
- What changed: wording only. Old: "Garra applies an entrant or a regulator for the whole run, in a duopoly, with a single provider". New: "Garra runs the interventions (a warning, a Harrington-style regulator, a random-price entrant) in independent runs with no checkpoint pairing, with a single provider (DeepSeek-V3.1) and no efficiency-cost measure (three firms once the entrant is in; entry timing unverified)". The library entry `garra2026`, the gap table and the outline carry the same wording.
- Why: the project-manager session re-read the Garra 2026 page: "for the whole run" and "duopoly" are not supported by the paper; the registered document and the paper must agree. No hypothesis, threshold, window, exclusion, lever parameter or analysis code changes.
- Data seen before the change?: irrelevant to any analysis (no outcome data involved).
- Affected analysis becomes exploratory?: no.

### A-005
- UTC timestamp: 2026-10-10T11:45:00Z (decision); applied with the A-005 config commit and `freeze.py --amend --amendment A-005`
- File and section changed: `PREREG.md` section 2 "Families" and D-11; `campaign_table.md` M-lane rows; configs TR-M, SH-M, LT-M (lane, model, family, seeds), CA-M, GH-M, S2-M, TR-M-ext (moved to Tier 2, not run), PC-M (header: dropped); `configs/analysis/registered_arms.yaml` lanes.M.model label
- What changed: the second family is no longer Mistral. PC-M (Fish exact on ministral-8b) is dropped and Gate 1 (M lane) is NOT RUN. The confirmatory second-family set (H1-M) runs on Llama-3.3-70B via OpenRouter (`openrouter_llama`, paid, inside the 10 USD credit) with a reduced scope: TR-M on 6 trunks (seeds 0-5), SH-M, LT-M with L2 only; CA-M, GH-M, S2-M and TR-M-ext are not run. Llama-family runs are judged by Sonnet 5.5 (the judge is never the family being judged); Llama keeps its role as judge of the Sonnet arm. Model size across families: 70B versus Haiku 5.5 (unknown size); H1-M stays a robustness arm for the direction of the L2 effect.
- Why: instrument failure. ministral-8b-latest produced the registered JSON output in about one decision out of four (PC-M: about 4 provider calls per decision at max_tokens 600, 3 at 2,048 with about 1,700 output tokens on average and 36 % at the cap); ministral-14b-latest failed the unregistered 20-call pilot the same way (40 % parse failures, 3.3 calls per decision); Llama-3.3-70B passed the same pilot (1.00 call per decision, 0 parse failures, 250 output tokens, 0.0003 USD per call). Decision taken on ledger and parse statistics only.
- Data seen before the change?: no outcome metric (price, profit, Delta) of any M-lane run was read; only call counts, token counts, parse outcomes and latencies.
- Affected analysis becomes exploratory?: H1-M stays confirmatory in its reduced scope (6 trunks, L2 only) with the registered test; the Mistral transfer test of H9 (detection on the M lane) is reported on the Llama arm instead and labelled as such; CA-M, GH-M and S2-M claims are dropped.

### A-007
- UTC timestamp: 2026-10-10T12:20:00Z
- Commit: the commit adding this entry
- File and section changed: `PREREG.md` D-19 sentence "MQC for every lever is computed on CPU and committed before Gate 2 is read" (text unchanged; this entry records the deviation and the rule as applied)
- What changed: Gate 2 (H lane) was read at 09:46Z (A-000-1) before any MQC document existed (the scripted CPU controls SC-1 and MQC-L* had not been run; they start at 13:30 Paris). The rule as applied: the MQC document (`mqc.json` from `analyze.py --mqc-only` on SC-1 and MQC-L1..L6) is committed, with its sha and UTC, before any lever contrast is read, and in particular before the H1 primary read; A-003(b) (SC1 ask fall under L2) comes from the same document.
- Why: scheduling oversight by the lead; no effect on qualification (Gate 2 reads trunk data only) and none on the MQC values (scripted controls, deterministic given the configs, independent of every LLM outcome).
- Data seen before the change?: gate outputs of the instrument campaigns only (A-000-1..3); no lever or sham outcome.
- Affected analysis becomes exploratory?: no.

### A-008
- UTC timestamp: 2026-10-10T13:10:00Z
- Commit: the commit adding this entry (documentation; the mapping is implemented in the analysis code before stage 2)
- File and section changed: convention note for every window in `PREREG.md`, `SAP.md`, `DETECTION_PROTOCOL.md` and `configs/analysis/registered_arms.yaml` (texts unchanged)
- What changed: the registered windows are 1-based period labels (trunk 71-100, branch 111-150, Fish 251-300, trend slope 81-100, branch at period 100 = after 100 periods); the engine logs periods from 0 (trunk 0..99, branch 0..149). The analysis maps registered period p to engine period p-1 (trunk 70..99, branch 110..149, Fish 250..299, trend 80..99) and keeps the 1-based labels in reports. Found when the completeness filter (stage-2 fix B2) rejected every trunk ("29 of 30 periods"). The Gate 2 reads of 09:46Z (H lane) and the S-lane re-read used engine periods 71..99 (29 periods, shifted by one); both lanes are re-read on the registered window with new A-000 lines before stage 2.
- Why: convention mismatch between the documents and the store; no threshold, hypothesis or estimator changes.
- Data seen before the change?: gate outputs on instrument campaigns only (A-000-1..4); no lever or sham outcome.
- Affected analysis becomes exploratory?: no.

### A-009
- UTC timestamp: 2026-10-10T13:15:00Z
- Commit: the commit adding this entry
- File and section changed: `PREREG.md` H1 (MQC criterion) and A-003(b); reading of the MQC document
- What changed: (1) The registered MQC definition and the H1 criterion "MQC(L2) point <= 5 pp" are KEPT as the confirmatory rule. The scripted competitive market (SC-1, 20 seeds, CPU) gives MQC(L2) = 26.5 pp [26.2, 26.8]: under the registered fringe design 30 % of arriving buyers become fringe sellers, so total surplus falls mostly because buyers leave the market (trades 3,835 -> 3,011 on seed 0), not because prices form worse; H1 therefore cannot be "supported" whatever the L2 contrast shows, and L2 is ineligible for the best-lever estimate; this is reported plainly with the registered MQC components (volume, allocative efficiency, buyer surplus). (2) An EXPLORATORY per-buyer efficiency measure is added, decided now before any L2 LLM outcome is read: total surplus per arriving buyer (and allocative efficiency per executed trade) under the lever versus the sham, on the same SC-1 runs and, descriptively, on the LLM arms; it is labelled exploratory everywhere and changes no confirmatory verdict. (3) A-003(b): the SC1 best-responders' ask response to L2 is exactly 0.00, so the "twice the SC1 ask fall" bound degenerates to 0; the bound becomes max(2 x SC1 ask fall, 2 x SD across trunks of the sham-minus-sham2 mean-ask difference on periods 111-150), computed after stage 2 from the sham and sham2 branches only, before any L2 contrast is read. Reference values: MQC(L1) = -1.79 pp (gain; SC1 asks 1.35 -> 1.10), MQC(L3) = -0.11 pp; L4a and L6 pending.
- Why: the registered lever design confounds liquidity supply with demand removal; the confirmatory rule is kept to avoid a forking path after seeing the MQC value; the per-buyer measure is the data-independent sensitivity analysis a reader would ask for; the A-003(b) bound must not label sampling noise as a reduction.
- Data seen before the change?: scripted CPU controls only (SC-1, MQC-L1/L2/L3); no LLM lever, sham or sham2 outcome.
- Affected analysis becomes exploratory?: the per-buyer measure is exploratory by construction; H1 keeps its registered rule.

### A-012
- UTC timestamp: 2026-10-10T13:50:00Z
- Commit: the commit adding this entry
- File and section changed: LT-M and LT-S template rule resolution (D-26; A-002 step 2; A-005)
- What changed: on the blind Tier 0 pass (main 14b6bf4, arms as letters, sealed key ../finswarm-sealed, sha dd44c47b), the rule "largest capped behavioural Delta-C_p under the trunk scoring among levers with MQC <= 5 pp, L2 excluded" selects letter B (0.620, p 1e-5, MQC -0.58 pp); next G/A tie at 0.001. Letters are linked to levers by their MQC values (the --mqc input leaks the mapping), so B = L6 (entrant). Resolved: LT-M = L2 + L6; LT-S = L2 + L6 (one rule lever, A-002). Disclosure: the blind output prints the H1 section for D (= L2): Delta-C_p 0.003, trunk p 0.21, own p 0.26; science saw it before computing the A-003(b) noise bound, which is moot since the trunk test is not significant.
- Why: registered rule resolution before the LT campaigns launch.
- Data seen before the change?: Tier 0 branch contrasts under letters (and the H1 line for D as disclosed).
- Affected analysis becomes exploratory?: no.

### A-013
- UTC timestamp: 2026-10-10T14:25:00Z
- Commit: the commit adding this entry, applied with `freeze.py --amend --amendment A-013`
- File and section changed: `configs/analysis/registered_arms.yaml` (ghost arm match: {arm: sham, label: SH-H, ghost: true}, as ghost_S on SH-S); `src/finswarm/analysis/corpus_v1.py` (descends into ghost sub-runs; lane filtered by campaign name)
- What changed: the H ghost arm matched a GH-H parent campaign that was never run as such (the 48 ghosts, 4 per sham, sit under each SH-H sham run); the detection corpus builder therefore found no ghost windows. Discovery only; no inference change.
- Why: data layout mismatch found when building the detection corpus for the owner's blind labels.
- Data seen before the change?: H1 and H1-R were read before (A-000-6 and the frontier read); no detection output existed; no ghost metric read.
- Affected analysis becomes exploratory?: no.

### A-014
- UTC timestamp: 2026-10-10T14:40:00Z
- Commit: the commit adding this entry, applied with `freeze.py --amend --amendment A-014` once the code lands
- File and section changed: `src/finswarm/analysis/{detect,corpus_v1}.py`, `scripts/analyze.py`, `configs/detection/corpus.yaml`
- What changed: (1) `--detect split` writes the development / evaluation seed groups only (status "split", no thresholds), so the owner's blind labels can be sampled from the development half before any rule is calibrated; (2) corpus v1 "sampler-first" mode draws the 60 development-half items before they are judge-scored and writes a score-request list, keyed by the split's lane:seed groups, with a family cap of 10 items per seed group and a round-robin draw over the strata; items need not be pre-scored, and kappa uses their cached D9 score once scored; (3) the M entries of corpus.yaml are disabled (A-005: no M-lane detection corpus). No inference change; the detection freeze (thresholds on the development half) and the single evaluation stay as registered.
- Why: ordering of tonight's steps (labels 20:30-21:30, rules frozen 22:00, evaluation once).
- Data seen before the change?: no detection output; H1 and H1-R read earlier (A-000 lines).
- Affected analysis becomes exploratory?: no.

### A-015
- UTC timestamp: 2026-10-10T14:55:00Z
- Commit: the commit adding this entry, applied with `freeze.py --amend --amendment A-015`
- File and section changed: `configs/analysis/registered_arms.yaml` arm GR-s1 (match condition)
- What changed: the GR-s1 match `config: {market.routing.tau: 0.25}` is replaced by a match on `tags.level: s1` (the GR-H s1 row sets `tau: null`, which the engine resolves to alpha * mu = 0.25; the literal 0.25 never matches), so H2 reads the 16 GR-H runs (4 levels x 4 CRN seeds) as registered instead of 12. The other GR levels may match on `tags.level` likewise. No inference change.
- Why: registry / config literal mismatch found by the preliminary family S read (H2 computed on 12 runs).
- Data seen before the change?: the preliminary H2 (12 runs: rho -0.59, permutation p 0.028; pooled s1 - s4 gap 0.17) was read before the fix; the final H2 on 16 runs supersedes it and both are disclosed.
- Affected analysis becomes exploratory?: no (the fix restores the registered sample; the preliminary read is reported as such).

### A-010
- UTC timestamp: 2026-10-10T15:05:00Z
- Commit: the commit adding this entry, applied with `freeze.py --amend --amendment A-010`
- File and section changed: `scripts/verify.py`, `src/finswarm/analysis/family_s.py`
- What changed (analysis, no threshold change): verify.py recomputes the SAP exclusions and the trunk qualification from the raw tables (and raises on a mismatch with the analysis list) instead of reading them from stats.json; p-values are compared at 1e-6 relative with an alpha-decision check; the H8 follow-difference test uses t with K-1 degrees of freedom (K = collusive clusters) instead of a normal approximation, in family_s.py and verify.py alike.
- Why: red-team findings 8 and 10 of the analysis code (independence of the verifier; anti-conservative H8 p with about 6 clusters).
- Data seen before the change?: H1 and H1-R read; family S read only in its preliminary form (H8 "not testable yet" at that time).
- Affected analysis becomes exploratory?: no.

### A-016
- UTC timestamp: 2026-10-10T15:30:00Z
- Commit: the commit adding this entry, applied with `freeze.py --amend --amendment A-016`
- File and section changed: `configs/analysis/registered_arms.yaml` (S1c_LLM match and config paths; family S lane of H7-other if needed in `src/finswarm/analysis/family_s.py`)
- What changed: the S1c probe runs carry labels S1c-LLM-H (CA-H directory, S1 and S2 on the 4 CA-H runs) and S1c-LLM-TR-H (S2 on 3 non-qualified trunks), while the registry matched label S1c-LLM and config S1c-LLM.yaml; the match is widened to both labels. H7-other reads arm R3 (lane H, newcomer from the second family, Llama per A-005). Discovery only; no inference change.
- Why: family S final read reported H7-other and H8 "not testable" for registry reasons.
- Data seen before the change?: family S read (H7-same supported; H2/H4/H5 not; H6 not testable; H10 not supported); no H7-other or H8 value.
- Affected analysis becomes exploratory?: no.

### A-018
- UTC timestamp: 2026-10-10T16:05:00Z
- Commit: the commit adding this entry, applied with `freeze.py --amend --amendment A-018`
- File and section changed: `src/finswarm/metrics/collusion.py` (sting_starts_from_events), `src/finswarm/analysis/family_s.py`, `scripts/verify.py`
- What changed: the sting follow counts read the deviation size from the nested event payload (`payload.event`, as the engine writes it; the top-level read gave NaN and 0 trials) and start from the three "raise" sting phases per probe instead of the single sting row: 12 trials per run (4 rivals x 3 probes) on S1 and S1c-LLM-H. No inference change; H8 had reported "not testable" for this reason.
- Why: event-layout mismatch between the engine and the metric, found in the family S final read.
- Data seen before the change?: no H8 value (0 trials); other family S results read.
- Affected analysis becomes exploratory?: no.

### A-011
- UTC timestamp: 2026-10-10T16:40:00Z
- Commit: the commit adding this entry, applied with `freeze.py --amend --amendment A-011`
- File and section changed: `src/finswarm/analysis/figures.py`, `scripts/make_figures.py`, `scripts/verify.py`
- What changed (figure and verifier code only, no inference change): make_figures merges several --stats inputs, pools the Fig. 2 levels, turns the unverified Fish band / reference off by default and labels it unverified when shown, draws AC-H as a distinct point, fixes legend and clipping, and writes gate CIs plus role, input sha and code sha in the sidecars; verify.py no longer crashes on the GR-H s4 rows (logit benchmark underflow at tau 0.02).
- Why: science's Fig. 1 findings; verifier robustness at the registered s4 level.
- Data seen before the change?: all Tier 0 / family S reads to date; no detection output.
- Affected analysis becomes exploratory?: no.

### A-019
- UTC timestamp: 2026-10-10T17:05:00Z
- Commit: the commit adding this entry, applied with `freeze.py --amend --amendment A-019`
- File and section changed: new configs `configs/sprint/tier2/L6b-H.yaml`, `L6c-H.yaml` (entrant type: fixed undercutter; static Nash-price seller); existing Tier 2 configs `R2-H.yaml` (replacement with memory kept) and `R4-H.yaml` (two dealers replaced) promoted to run; CPU check of the ghost null against a plain cyclic-shift null on the existing ghost windows
- What changed: three EXPLORATORY follow-ups, decided after the registered reads and labelled exploratory everywhere (no hypothesis, no decision rule, descriptive contrasts with CIs on the 11 qualified trunks): (1) entrant type (which contestability breaks the established regime: no-regret learner in L6 vs fixed undercutter vs static Nash seller); (2) replacement hardening (memory kept vs erased; one vs two dealers replaced); (3) false-positive rate of the ghost null vs a cyclic-shift null on ghost windows (CPU, no LLM calls). Haiku cost about 5-6 USD inside the 200 USD cap; launched before the Sunday 06:00 bound.
- Why: the owner's adversarial novelty search (project-manager session): entry-breaks-collusion is known (Garra 2026, Deshpande-Jacobson 2026, Keppo 2026); the open and regulator-relevant questions are the entrant type and what the replacement erases; the ghost-null check answers Papadopoulos 2026.
- Data seen before the change?: every registered Tier 0 / family S result; no data of these four campaigns (not yet run).
- Affected analysis becomes exploratory?: these follow-ups are exploratory by construction; nothing registered changes.
- Code note 2026-10-10T19:40Z: `analyze.py --detect fpr-null` (detect.py) implements item (3), the FPR check of D5-net on ghost windows under the plain cyclic-shift null and under the ghost null; exploratory, writes fpr_null.json, nothing frozen depends on it; the exploratory A-019 arms (L6b, L6c, R2, R4) are registered as exploratory in registered_arms.yaml.
- Correction 2026-10-10T18:20Z (before any L6b/L6c data: the first L6b-H launch failed at construction, 0 calls): the entrant params copied from L6 (`n_prices`, `horizon`) are replaced by the policies' own: undercutter `eps 0.05, floor 1.0`; static Nash seller `price 1.3315` (the trunk p_N).

### A-020
- UTC timestamp: 2026-10-10T17:40:00Z
- Commit: the commit adding this entry, applied with `freeze.py --amend --amendment A-020`
- File and section changed: `configs/sprint/tier2/PREV-H.yaml` (existing, promoted to run), new `PREV-L5-H.yaml` (poisson revision clock from period 0), new `PREV-L6-H.yaml` (a no-regret EXP3 seller present from period 0 next to 4 Haiku dealers, Garra-style design in our swarm market)
- What changed: EXPLORATORY prevention block, decided after the registered cure results and before any prevention data: do the levers that fail as cures (fringe L2, asynchrony L5) also fail when present from the start, and does entry from the start prevent the regime as it breaks it mid-run? 4 seeds x 100 periods each, about 1,600 Haiku calls per config (about 3 USD in total); descriptive Delta_p on 71-100 against the TR-H trunks (capped index, trunk benchmarks, BCa CIs), labelled exploratory, no decision rule. GR-H (routing from scratch) already serves as the prevention arm for L1.
- Why: the owner's question on prevention; it answers the reviewer objection "the levers came too late" and tests for hysteresis.
- Data seen before the change?: every registered result; none of these three campaigns.
- Affected analysis becomes exploratory?: exploratory by construction; nothing registered changes.

### A-021
- UTC timestamp: 2026-10-10T17:50:00Z
- Commit: the commit adding this entry
- File and section changed: A-020 withdrawn (configs stay on disk, hashed, unused)
- What changed: the owner chose a single storyline for the paper ("what breaks an established LLM cartel, and at what cost?"); the exploratory prevention block A-020 is cancelled before any of its campaigns ran; GR-H remains the only from-scratch evidence and prevention is a one-sentence limitation.
- Why: breadth control.
- Data seen before the change?: none from A-020 (no run launched).
- Affected analysis becomes exploratory?: not applicable.

### A-017
- Never assigned (numbering gap; recorded here so the appendix table is complete).
- Correction to A-016 (2026-10-10T20:40Z): the R3 newcomer is ministral-8b-latest as configured (R3-H.yaml never changed under A-005); its runs are excluded by the registered 10 % failed-decision rule for the same output-format reason as PC-M.

### A-022
- UTC timestamp: 2026-10-10T21:30:00Z
- Commit: the commit adding this entry, applied with `freeze.py --amend --amendment A-022`
- File and section changed: `src/finswarm/analysis/{campaign,stats,figures}.py`, `scripts/verify.py`, `scripts/make_figures.py`
- What changed (reporting fields only, no inference change): per lever, the raw (uncapped) contrast with BCa interval and SD and a raw-index TOST, sign counts (trunks with zero, negative, positive differences), the raw sham-minus-sham2 SD; sigma_d relabelled "pooled lever-minus-sham SD including L6" (not a noise scale); Fig. 3 gains a ceiling panel (raw contrast and share of quotes above p_M per lever).
- Why: results red-team findings 2, 3 and 12 (the capped index cannot register cuts that stay above p_M; capped sham-sham2 differences are 0 in 11/11 pairs).
- Data seen before the change?: every registered and exploratory result.
- Affected analysis becomes exploratory?: no (reporting only; registered tests unchanged).

### A-024
- UTC timestamp: 2026-10-10T22:10:00Z
- Commit: the commit adding this entry; code applied with `freeze.py --amend --amendment A-024`
- File and section changed: `src/finswarm/analysis/**` (new diagnostic functions, exploratory), reporting in `paper/results_notes.md`
- What changed: three exploratory diagnostics on existing data, no inference change, decided after the debate of 2026-10-10 evening: (a) replacement: regression of the fresh copy's first post-replacement ask and window mean on the incumbents' T0 ask across the qualified seeds (default opening versus joining the convention); (b) entry: post-entry incumbent asks versus the static logit best response to the entrant's ask; (c) forced-undercut probes: rivals' response versus the static best response, profitability versus the sham, reversion to the pre-probe ask. Also reporting rules: per-trunk difference tables and exact sign tests as the primary display, uncapped index with 90 % CI beside the capped one, TOST confined to appendix A6 ("degenerate variance"), screened versus qualified trunks reported.
- Why: adversarial debate (PM, science, metrics): the capped index saturates at the ceiling; "re-collusion" may be the default opening plus public history; matching an undercut may be the static best response.
- Data seen before the change?: every result read; the diagnostics are descriptive.
- Affected analysis becomes exploratory?: the diagnostics are exploratory; registered tests unchanged.

### A-023
- Never assigned (reserved on 2026-10-10 evening for a Sonnet L6 robustness branch, withdrawn after the debate: same provider, not a second family; no run). Numbering summary for the appendix: A-001..A-024 with A-017 and A-023 never assigned, A-020 cancelled before data (A-021).

## Timestamp reconciliation (added 2026-10-11; authoritative times are git commit times, UTC)

The `UTC timestamp` lines inside the entries were typed by hand during the sprint and several are wrong by one or two hours (they were estimated from Paris time). The authoritative record is this table: the commit that added each entry (git author date, UTC) and the hash-amendment lines in `FROZEN`. Entries A-003, A-007, A-008, A-009, A-021, A-024 changed no hashed file and therefore have no `FROZEN` line. A-019 ordering: the entry (configs) was hashed at 16:14:46Z; the parameter correction was hashed at 16:39:18Z, BEFORE the L6b/L6c campaigns ran (relaunch after that hash; results logged after 17:00Z); the hand-typed '18:20Z' in the correction line is wrong.

| Amendment | Entry commit | Entry commit UTC | FROZEN hash lines (UTC, commit) |
| --- | --- | --- | --- |
| A-001 | 873562d | 2026-10-09T20:40:48+02:00 | 2026-10-10T13:18:10Z (230507a) |
| A-002 | af5ec0e | 2026-10-10T11:22:59+02:00 | none (no hashed file changed) |
| A-003 | 0555d6e | 2026-10-10T12:27:10+02:00 | none (no hashed file changed) |
| A-004 | 697a340 | 2026-10-10T12:31:56+02:00 | 2026-10-10T10:32:25Z (697a340) |
| A-006 | 6204855 | 2026-10-10T13:16:23+02:00 | 2026-10-10T11:16:52Z (6204855) |
| A-005 | 9eca8c9 | 2026-10-10T13:39:27+02:00 | 2026-10-10T12:56:38Z (d0eb2c2) |
| A-007 | 9694cbf | 2026-10-10T13:51:31+02:00 | none (no hashed file changed) |
| A-008 | 7cdb81a | 2026-10-10T14:54:37+02:00 | none (no hashed file changed) |
| A-009 | 7cdb81a | 2026-10-10T14:54:37+02:00 | none (no hashed file changed) |
| A-012 | a9f1699 | 2026-10-10T15:31:46+02:00 | 2026-10-10T13:35:49Z (52619e6) |
| A-013 | e5ff28d | 2026-10-10T15:58:54+02:00 | 2026-10-10T13:59:08Z (e5ff28d) |
| A-014 | c21be2a | 2026-10-10T16:10:31+02:00 | 2026-10-10T14:21:52Z (a181a39); 2026-10-10T14:26:26Z (3e73cde); 2026-10-10T14:43:00Z (6de47b0) |
| A-015 | ff80fe4 | 2026-10-10T16:18:12+02:00 | 2026-10-10T14:24:38Z (91a1d94) |
| A-010 | bc94dfa | 2026-10-10T16:37:18+02:00 | 2026-10-10T14:37:38Z (bc94dfa) |
| A-016 | 3a1d4dd | 2026-10-10T17:16:26+02:00 | 2026-10-10T15:20:33Z (e9af2de) |
| A-018 | b10f7fa | 2026-10-10T17:36:19+02:00 | 2026-10-10T15:36:31Z (b10f7fa) |
| A-011 | 1a51b39 | 2026-10-10T17:57:14+02:00 | 2026-10-10T15:57:30Z (1a51b39) |
| A-019 | d6836f4 | 2026-10-10T18:14:32+02:00 | 2026-10-10T16:14:46Z (d6836f4); 2026-10-10T16:39:18Z (9132d0a); 2026-10-10T17:37:07Z (667a703) |
| A-020 | 657924f | 2026-10-10T18:24:25+02:00 | 2026-10-10T16:24:41Z (657924f) |
| A-021 | a522a5b | 2026-10-10T18:30:38+02:00 | none (no hashed file changed) |
| A-017 | 89e6f2e | 2026-10-10T19:57:21+02:00 | none (no hashed file changed) |
| A-022 | 61d579d | 2026-10-10T20:28:38+02:00 | 2026-10-10T18:28:54Z (61d579d) |
| A-024 | 70400b7 | 2026-10-10T20:53:55+02:00 | none (no hashed file changed) |
| A-023 | 2f6646f | 2026-10-11T03:53:19+02:00 | none (no hashed file changed) |

### A-025
- Environmental, no inference change (2026-10-11 morning): `scripts/verify.py` reads Parquet through polars when `pyarrow.parquet` cannot load (Windows application control blocks the DLL on this machine); same API, same values.

### A-026
- Exclusion after data (2026-10-11 ~07:45 Paris, before any H8 number was read on this arm for the paper): the arm `S1c-LLM-TR-H` (competitive side of the sting on TR-H trunks, 3 runs, A-016) is excluded from H8 and from claim C7. Reason: an exact `request_id` join of its post-branch decisions with the cost ledger attributes all 597 of 597 incumbent decisions to cache entries written by the same-trunk `S2` runs (forced undercut by slot 0); the arm made 0 live calls after the checkpoint, so its incumbents never saw the slot-4 sting. It also ran on qualified trunks (seeds 0, 1, 2), against the registry's "non-qualified TR-H trunks". The H8 competitive side is the CA-H `S1c-LLM-H` arm alone, as registered in `registered_arms.yaml` (S1c_LLM). H8's reported value does not change (follow rates are 0 on both sides); the number of competitive clusters does. The replay count is published as a numbers row (`C7_s1c_tr_cache_replay`). The same join on every lever and replacement branch finds 0 post-branch decisions served from the sham's cache (rows `C13_cache_reuse_<arm>`), which is reported in Methods; the mechanism (branches keep the trunk request seeds) is disclosed.
- Also re-hashed under this entry: `configs/analysis/registered_arms.yaml` as changed at commit 944bfe3 (2026-10-11 04:54 Paris): the A-019 exploratory replacement arms R2 and R4 added to the registry so that their descriptive contrasts are verified rows; no registered arm, window or rule changed. The hash was not re-run at that commit; this entry closes the gap.
