# Statistical analysis plan (SAP)

Version v1.1-draft (amended by D-3' and by the second review, D-16 to D-27), 2026-10-10; status DRAFT (frozen with `PREREG.md`, in the hashed file set). Source: plan Appendix D.3, revised after two red-team reviews (`REDTEAM_RESPONSE.md`). Companion: `PREREG.md` (hypotheses, thresholds), `DETECTION_PROTOCOL.md`.

## 1. Units and estimands

| Item | Rule |
| --- | --- |
| Unit of analysis | The run / trunk (never the period or the decision); every bootstrap resamples trunks (or runs), never individual differences |
| Raw price index | Delta_p = (p_bar - p_N) / (p_M - p_N); p_bar over the **incumbent LLM dealers** only (trunk slots 0-3; entrant seats, sting dealers, replayed rivals and the fringe excluded) |
| Capped index | min(Delta_p, 1); used by H1, H1-R, H1-M, H-S and the family-S lever contrasts. Raw index and share of quotes above p_M reported for every window |
| Scorings | (i) trunk scoring: both arms on the pinned pre-lever trunk benchmarks (primary, registered); (ii) own scoring: each arm on the benchmarks of its own market, the sham on the trunk's (p_N, p_M) and the lever on the lever market's (p_N', p_M') from the pinned file (section 5); d_own = Delta_p,sham(trunk) - Delta_p,lever(own) (capped). Scoring both arms on one benchmark pair would cancel p_N' and reduce to a rescaled trunk contrast |
| Profit index | Delta_pi with per-window benchmarks; co-condition of "collusion established" (>= 0.30 on 71-100) and of Gate 1 / Gate 1b (>= 0.10 on 251-300) |
| Paired contrast (branch vs sham of the same trunk) | d_lk = capped Delta_p(sham_k) - capped Delta_p(lever l on k), window 111-150, under each scoring; Delta-C_p,l = mean_k d_lk |
| Mechanical part | Frozen-quote replay (CPU): the incumbents' sham quotes held fixed period by period, re-scored under lever l's market with the same background seed; Delta-C^mech_l = mean_k [Delta(sham_k) - Delta(replay_lk)], for both scorings and for Delta_pi |
| Behavioural part | Delta-C^beh_l = Delta-C_l - Delta-C^mech_l, per trunk then averaged; CI by resampling trunks |
| Reported next to every Delta-C_p | raw-index contrast; share of quotes above p_M; benchmark shift (p_N, p_M before and after the lever); SC1 best-responder price response to the lever (MQC runs); trade-weighted market price |
| From-scratch comparisons (gradient s1-s4) | Delta_p with each level's own benchmarks; run-level Spearman over the 16 GR-H runs (s1-s4 x CRN seeds 3001-3004), permutation p within seed groups (section 4) |
| Rates (sting follow and punishment) | Difference of proportions, cluster = run |
| Dynamics | Time to collude (first period with a 10-period moving mean Delta_p >= 0.30); break (Delta_p <= 0.5 x trunk level); recovery time; hysteresis; right-censoring handled explicitly |
| MQC | SC1 `best_responder` market with and without lever l, 20 seeds (CPU); trades-based total surplus at true valuations (no taste shock), one unit for every lever; MQC_l = loss in pp of the no-lever surplus; point estimate (used in the rules) with a 95 % CI (bootstrap over seeds); computed for every lever and committed before Gate 2 is read |

## 2. Sidedness (stated per test)

| Hypothesis | Test | Sided |
| --- | --- | --- |
| Gate 1, Gate 1b | Run-level bootstrap CI of the pooled mean Delta_p (runs resampled, stratified by alpha for Gate 1); pooled Delta_pi >= 0.10 | two-sided 95 % CI (criterion: excludes 0) |
| H1 (L2, pre-specified primary) | Paired t on d_{L2,k}, H0: mean <= 0, all qualified trunks, under each scoring; Wilcoxon signed-rank as sensitivity | one-sided (reduction) |
| H1 practical significance | Paired-t lower bound of Delta-C_p(L2), trunk scoring; BCa 90 % interval reported alongside | one-sided 95 % lower bound |
| H1 null | TOST on Delta-C_p(L2), trunk scoring, margin +/-0.20 | equivalence |
| H1-R (8 levers) | Paired t on d_lk, H0: mean <= 0, all qualified trunks, under each scoring | one-sided (reduction) |
| H1-M, H-S | Paired t on d_{L2,k} on the qualified M / Sonnet trunks, under each scoring | one-sided (reduction) |
| H2 | Spearman rho(level, Delta_p) over the 16 GR-H runs, permutation p within seed groups; Welch t of pooled s1 (16 runs) minus s4 (4 runs) against 0.30 | one-sided (decrease); one-sided (H0 mean(s1) - mean(s4) <= 0.30) |
| H3 (prevention, outside Holm) | t of Delta_bar_p(71-100) against 0.15 | one-sided (H0 mu >= 0.15) |
| H4, H6-carrier, H7-other | Paired t against the MEI 0.15 | one-sided (H0 mu <= 0.15) |
| H5 | Two one-sided paired t against +0.15 and -0.15, alpha / 2 each; Holm input p = min(1, 2 x min(p_low, p_high)) | two-sided (H0 abs(mu) <= 0.15) |
| H7-same, H10 | TOST, margin 0.15 (the MEI) | equivalence |
| H8 | Difference of follow proportions against 0.30 (cluster = run) | one-sided (H0 diff <= 0.30) |
| H9 (outside Holm) | Frozen thresholds; LLM-only AUC with CI resampling seed groups | n/a (rule-based) |

## 3. Multiplicity and equivalence

- **H1 (primary):** single pre-specified contrast (L2), alpha = 0.05, **no multiplicity adjustment**; "collusion reduction" requires significance under both scorings (dual-benchmark rule); trunk-only significance is reported as "price reduction consistent with competitive adjustment". If L2 cannot run, H1 is "not tested".
- **H1-R (secondary confirmatory):** Holm over the **fixed list of m = 8** levers {L1, L2, L3, L4a, L5, L6, L7, L8} on the full sample, family-wise alpha = 0.05, within each scoring; dual-benchmark rule for the label; a lever not run enters with p = 1.
- **Family S:** {H2, H4, H5, H6-carrier, H7-same, H7-other, H8, H10}, **m = 8 fixed**, Holm at alpha = 0.05; a hypothesis that cannot be tested enters with p = 1. For a joint rule the Holm p-value is the larger component p-value (intersection-union); point co-conditions are checked on the estimate. H3 and H9 are outside Holm.
- **H1-M, H-S:** single pre-specified contrast each (L2), alpha = 0.05, dual-benchmark rule; outside family S; H1-M reported as robustness only.
- Exploratory results labelled as such; no multiplicity claim.
- TOST margins: H1 and H1-R nulls +/-0.20; family S = the hypothesis's MEI.

## 4. Procedures

1. **Primary set** = all trunks that qualify at Gate 2 (Delta_bar_p(71-100) >= 0.30 and Delta_bar_pi(71-100) >= 0.30), in seed order (5-12). No seed is ever added. H1, H1-R, H7, H8, H10 and the frontier use the primary set.
2. **H1 (L2).** One-sided paired t (H0: mean <= 0) under the trunk scoring and under the own scoring, alpha = 0.05, no adjustment. **Collusion reduction (supported)** iff p < 0.05 under both and the MQC(L2) point estimate <= 5 pp. **Practically significant** iff the one-sided 95 % paired-t lower bound (trunk scoring) > 0.10; **inconclusive** iff the 90 % two-sided paired-t interval straddles 0.10-0.20 (lower limit <= 0.10, upper limit >= 0.20). The BCa 90 % interval (10,000 resamples of trunks, fixed seed 20261010) is reported alongside. Null: TOST +/-0.20. Sensitivity: Wilcoxon signed-rank (smallest attainable one-sided exact p with n pairs is 2^-n: 0.031 at n = 5, 0.0156 at n = 6, 0.0078 at n = 7, 0.0039 at n = 8).
3. **H1-R.** One-sided paired t per lever, both scorings, p = 1 for a lever not run, Holm (m = 8); paired 95 % CIs by resampling trunks. Wilcoxon reported (it cannot reach the first Holm threshold 0.00625 at n <= 7).
4. **Best lever (descriptive).** Only with >= 8 qualified trunks (off at 5-7). Eligibility from CPU data (committed before Gate 2): MQC_l point <= 5 pp and 95 % upper bound <= 7 pp. The eligible lever with the largest capped Delta-C_p (trunk scoring) on the odd-seed qualified trunks is selected (tie: lower MQC); its estimate is Delta-C_p on the even-seed qualified trunks with a CI. Runs blind (letters).
5. **LT-M / LT-S rule.** At the Saturday-evening Tier 0 analysis, blind: L2 plus the eligible Tier 0 lever(s) other than L2 with the largest capped behavioural Delta-C_p (trunk scoring) and MQC <= 5 pp on the Haiku arm (one for LT-M, two for LT-S). The resolved configs are logged in `AMENDMENTS.md`; the rule is hashed.
6. **H2.** Run-level Spearman rho(level, Delta_p) over the 16 GR-H runs (levels s1, s2, s3, s4 at each CRN seed 3001-3004; the s1 level is the CRN-paired addition, +1,600 H calls), p by permuting the four level labels within each seed group (10,000 random permutations out of 24^4 = 331,776). Second component: H0 mean(s1) - mean(s4) <= 0.30, rejected at alpha 0.05 by a one-sided Welch t with the **pooled s1 sample** (12 TR-H trunks + 4 GR-H s1 runs = 16; the s1 market is identical across the two campaigns, only the seeds differ) against the 4 GR-H s4 runs; the CRN-paired version (paired t, n = 4) is reported as a robustness check. Holm p = the larger of the two components. The TR-H trunks enter H2 only through this pooled comparison, never the Spearman.
7. **H6-carrier dependence.** D5-net = transfer entropy minus its value on the sham's ghost markets (the same baseline for the sham and the L4a branch); fall in D5-net >= 50 % as point co-condition, evaluated only when the sham's D5-net exceeds the 95th percentile of its ghost permutation null; otherwise H6 is "not testable: no dependence to remove in the sham" (no decision) and the carrier claim rests on Delta-C_p only.
8. **Frontier.** Both scorings, Delta-C_pi, mechanical and behavioural parts, MQC, CE and the reported quantities of section 1 for every lever on all qualified trunks.
9. Intervals with n <= 6 are reported with the percentile interval next to them.

## 5. Benchmarks and pinning

| Case | Method |
| --- | --- |
| Closed-form demand | Nash and monopoly by bisection on the first-order conditions (`calvano2020`, `finswarm.market.demand`): duopoly p_N = 1.4729 alpha, p_M = 1.9250 alpha, pi_N = 0.2229 alpha beta, pi_M = 0.3375 alpha beta; n = 4 at s1: p_N = 1.3315 alpha, p_M = 2.0544 alpha, pi_N = 0.0815 alpha beta, pi_M = 0.2011 alpha beta |
| Swarm demand (no closed form) | Expected profits by Monte Carlo with common random numbers (10 replications); Nash by best response on a price grid (10 starts); monopoly by joint maximum; validated to within 3 % of the closed form at s1 |
| Fringe on (L2) | Calvano's closed form no longer applies: an arriving ZI buyer becomes a one-lot fringe seller with probability = share (0.30), ask uniform between the dealers' marginal cost and its own valuation (about U[1, 2] x alpha, mean about 1.5 alpha), finite lifetime. The lever market's own p_N, p_M and profit benchmarks are the static Nash and joint maximum of the dealers' one-shot game with the fringe present, computed by sweeps |
| Empirical Delta | SC1 `best_responder` as the Nash analogue; robustness variant |
| CDA (Agrawal setting) | Delta_p = (p_bar - 90) / 10 (competitive equilibrium 90; `agrawal2025`) |

**Pinning per campaign.** Benchmarks are pinned **before freeze** as one JSON per campaign under `configs/analysis/benchmarks/<campaign>.json` (in the hashed file set). Each file holds: the trunk benchmarks (p_N, p_M, pi_N, pi_M) used by the trunk scoring of every branch; the lever market's own benchmarks (swept; closed form where exact) used by the own scoring and by Delta_pi; per-level benchmarks for GR-H and TR-H; the duopoly benchmarks of Gate 1 and Gate 1b; the method and its replication count. `scripts/verify.py` re-solves every value independently. If v1.2 changes trunk parameters (F1, F2, trend rule), the files are re-pinned and their hashes appended before any branch.

## 6. Registered arms and slots (whitelist)

`configs/analysis/registered_arms.yaml` (hashed at freeze) lists, for every confirmatory arm: campaign ID, `tags.arm`, lane, seeds (or the rule that resolves them), hypothesis, window, benchmark file and key, the **incumbent slots** that enter Delta_p (0-3 for the trunk market) and the **excluded slots** (entrant seats, sting seat, replayed rivals, fringe). Confirmatory analyses read only whitelisted arms and slots; a run whose resolved config differs from its arm entry is excluded. Arms not in the file are exploratory by construction.

## 7. Windows

Trunks: 71-100 (30 periods). Branches: 111-150 (40 periods; 101-110 is a transition). Fish replication and Gate 1b: 251-300 (50 periods). AC-H: 71-100. If the trend rule fires (evaluated on all 12 TR-H trunks before qualification), every 100-period LLM campaign is extended to 120 periods by an `ext` branch and every window here, in PREREG and in DETECTION_PROTOCOL shifts by +20 (levers and shams as `.shifted.yaml` configs, PREREG section 1). Branch sensitivity windows: 121-150 (30), 111-150 (40), 101-150 (50); reported, not selected.

## 8. Noise, sigma_d and power

- **Sham-vs-sham band.** Sham2 runs on every qualified trunk; the SD of Delta_p(sham) - Delta_p(sham2) over trunks gives the band on every figure. The band is **conservative** (cache-salted: sham2 never reuses the sham's responses).
- **sigma_d** = pooled within-lever SD of d_lk over all levers run and all qualified trunks (each lever centred on its own mean), with a CI by resampling trunks. Never used to add seeds or change the design.
- **Power.** `scripts/power.py` computes (i) for the primary L2 test, N in {5, 6, 7, 8, 10, 12}, sigma_d in {0.15, 0.20, 0.30}, true effects {0.15, 0.25, 0.35}, the power of the p < 0.05 rule and of the lower-bound > 0.10 rule; (ii) the H1-R Holm-family power for one true effect among 8. Table in `power_table.md` (seed 20261010, 20,000 replications); planning values in `PREREG.md` section 2 (N = 12, sigma_d 0.20, effect 0.25: P(p < 0.05) = 0.992, P(t lower bound > 0.10) = 0.785, H1-R Holm power 0.882 with FWER 0.042); N in {5, 6, 7} appended before Gate 2; the practical-significance bound is the paired-t bound. After data: the **minimum detectable effect at the realised N and sigma_d_hat** for each contrast.

## 9. Sensitivities (reported, not selected)

P2 prompt; temperature 0.5 (Mistral only: Anthropic refuses non-default temperature); memory 30; alpha 3.2; branch windows (section 7); raw vs capped index; trunk vs own scoring; Delta_p vs Delta_pi; Delta_emp; grid-constrained benchmarks (tick 0.05) at s3 and s4; trimmed means; exclusion of flagged decisions; CRN on/off. Source of the temperature conflict: `lin2024` vs `xu2026`.

## 10. Failure handling

- JSON invalid: at most 10 repairs, then hold the previous quote and flag.
- Price outside [c, 2 p_M]: repair, then clamp and flag.
- Runs with > 10 % flagged decisions are excluded.
- API outage: always relaunch from the last checkpoint with the same seeds; exclude only if the relaunch fails; listed in appendix A1.
- An excluded lever branch drops its trunk from that contrast only; an excluded sham drops its trunk from every contrast.
- Ghost or sham2 run with any post-branch cache hit: excluded and relaunched.
- An arm stopped by a budget cap is reported as incomplete with the trunks lost; no selective relaunch.
- Refusals (Anthropic): default action, flagged.

## 11. Blind analysis

Analysis code frozen on scripted data and sham-vs-sham. The frontier is generated with `--blind` (levers as letters) and committed before unblinding. Scratchpad labelling blind. Thresholds frozen before the evaluation half is read.

## 12. Independent verification

`scripts/verify.py`: pure numpy, shares no code with `analysis/`, fixed-seed bootstrap, recomputes every number in `numbers.json` from raw logs and re-solves the pinned benchmarks; a mismatch blocks delivery. A claim without a passing `verify` row is removed.

## 13. Reporting

Effect sizes with 95 % intervals; sham-vs-sham band on every figure; both scorings, raw and capped indices, share above p_M, benchmark shift, SC1 response and trade-weighted price next to every Delta-C; mechanical and behavioural parts; levers not run listed with their reason; failures, exclusions and incomplete arms in appendix A1; nothing described as "in progress" in the paper.
