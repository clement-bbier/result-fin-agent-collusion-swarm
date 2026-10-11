# Campaign table (revised budget, post red-team)

Version v1.1-draft (amended by D-3' and by the second review, D-16 to D-27), 2026-10-10 (frozen with `PREREG.md`, in the hashed file set). Source: plan Appendix D.2, revised by plan sections 1 and 12 (about 200 USD of Anthropic credits) and by two red-team reviews (`REDTEAM_RESPONSE.md`). Call counts are taken from the authored configs (`docs/04_runbooks/campaigns.md`, new calls only) plus the red-team additions. Calls = n x (T - T0) x seeds for branches, n x T x seeds for from-scratch runs. One agent decision = one call.

Lanes: **H** = Haiku 5.5 (primary), **S** = Sonnet 5.5 (capability arm and judge on Haiku runs), **M** = Mistral Experiment tier, `ministral-8b-latest`, 150 req/min (free; second family; H1-M, reported as robustness), **L** = Llama-3.3-70B via OpenRouter (paid, 10 USD lane budget; third family: R3 newcomer and judge on Sonnet-arm runs), **G** = Gemini free tier (backup judge), **CPU** = scripted markets (0 LLM calls). Groq is dropped. H1 = the pre-specified L2 contrast. Routing levels (alpha = 1): s1 = tau 0.25, s2 = 0.125, s3 = 0.0625, s4 = 0.02; L1 = s1 -> s3.

**Registered launch order:** PC-H -> PC-H-dealer + AC-H -> TR-H -> SH-H + SH2-H -> levers -> GH-H.

Branch rows assume 12 qualified trunks; with fewer, every branch row scales by qualified / 12. **Rule-parametrised configs** (the rule is hashed, the resolved config is logged in `AMENDMENTS.md`): S1 and S2 (first 6 qualified trunks in seed order, resolved at Gate 2), LT-M and LT-S (L2 plus the best eligible Tier 0 lever(s) other than L2 by largest capped behavioural Delta-C_p with MQC <= 5 pp on the Haiku arm, resolved at the Tier 0 analysis), S1c-LLM (non-qualified trunks, resolved at Gate 2).

## Pre-freeze pilot (done, unrecorded measurement)

| ID | Condition | Lane | Calls |
| --- | --- | --- | --- |
| PILOT-H | Trunk configuration (prompt revision 2), 1 run x 4 dealers x 5 periods: USD/call, tokens, cache hit rate, latency | H | 20 |
| PILOT-S | Same on Sonnet 5.5 | S | 20 |
| Round-100 re-measure | Rendered round-100 prompt, revision 3, two consecutive calls per lane (cold, then warm) | H, S | 4 |

## Tier 0 (Saturday afternoon and evening) - lane H unless stated

| ID | Condition | Lane | n | T | T0 | Seeds / runs | Calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PC-H | Gate 1: Fish exact (`fish_v1_json`), alpha in {1, 3.2, 10} x 8 seeds | H | 2 | 300 | 0 | 24 runs | 14,400 |
| PC-H-dealer | Gate 1b: same duopoly with `dealer_v1`, alpha = 1 (n = 2, closed form, tick 0.01) | H | 2 | 300 | 0 | 8 | 4,800 |
| AC-H | Anchoring control: 1 LLM dealer + 3 `best_responder` rivals, trunk market, from scratch (runs with the Gate wave) | H | 1 LLM | 100 | 0 | 4 | 400 |
| PC-M | Gate 1 on M | M | 2 | 300 | 0 | 24 runs | 14,400 (free) |
| TR-H | A2 trunk (s1, V0, tick 0.05, sync, `dealer_v1`), seeds 0-11 | H | 4 | 100 | 0 | 12 | 4,800 |
| SH-H | Sham | H | 4 | 150 | 100 | 12 | 2,400 |
| SH2-H | Sham2 on every trunk | H | 4 | 150 | 100 | 12 | 2,400 |
| L1, L2, L3, L4a, L6 | Levers (L1 = tau 0.25 -> 0.0625) | H | 4 | 150 | 100 | 12 each | 5 x 2,400 = 12,000 |
| GH-H | Fresh-draw ghost markets (12 trunks x 4 live dealers) | H | 1 live | 150 | 100 | 12 | 2,400 |
| SC-1...5; MQC-l; frozen-quote replay | Scripted markets (SC1 = `best_responder`); MQC for every lever, committed before Gate 2 is read; mechanical part of Delta-C | CPU | - | 150 | - | 20 | 0 |
| **Total Tier 0** | | **H 43,620** (incl. PILOT-H 20) + M 14,400 | | | | | **58,020** |
| Contingent: Gate 1b bounded repair | PC-H-dealer re-run on fresh seeds 9101-9108, launched by 18:00 | H | 2 | 300 | 0 | 8 | 4,800 (not in the total) |

**Contingent on the trend rule** (not in the totals; PREREG section 1): every 100-period LLM campaign is extended to 120 periods by an `ext` branch from its period-100 checkpoint (+20 % calls): TR-H-ext (`configs/sprint/tier0/TR-H-ext.yaml`, 12 x 4 x 20 = 960 H), AC-H ext (4 x 1 x 20 = 80 H), GR-H ext (16 x 4 x 20 = 1,280 H), CA-H ext (4 x 4 x 20 = 320 H), PREV-H ext if run (320 H); TR-S ext (8 x 4 x 20 = 640 S) and TR-M ext (6 x 4 x 20 = 480 M, free) only if their own slope rule fires. Levers and shams then run as `.shifted.yaml` configs from `scripts/resolve_registered.py --shift 20` (parent TR-H--ext, branch at period 120), logged in `AMENDMENTS.md`.

Reconciliation: runbook Tier 0 H 37,200 + 1,200 (SH2-H on 12 trunks) + 4,800 (PC-H-dealer) + 400 (AC-H) + 20 (PILOT-H) = 43,620.

## Tier 1 (Saturday evening to Sunday morning) - lane H unless stated

| ID | Condition | Lane | Calls |
| --- | --- | --- | --- |
| L5 | Staggered updates: per-slot geometric revision times, mean = `epoch_ticks` | H | 2,400 |
| L7 | Speed bump: FIFO constant delay 300 ticks (3 periods), no random draw | H | 2,400 |
| R1 | Replacement, same model, memory erased | H | 2,400 |
| R3 | Replacement by a Llama-3.3-70B newcomer (cross-family) | H (+ L for the newcomer: 12 x 1 x 50 = 600 of the 2,400) | 2,400 |
| CA-H | "Compete aggressively" (4 seeds x 4 x 100) | H | 1,600 |
| GR-H | Gradient from scratch at s1, s2, s3, s4 (tau 0.25, 0.125, 0.0625, 0.02), CRN seeds 3001-3004: 4 levels x 4 seeds x 4 x 100 (the CRN-paired s1 level adds 1,600 to the authored 4,800) | H | 6,400 |
| L9 | Monitoring notice (H10) | H | 2,400 |
| S1, S2 | Sting: forced rise; forced undercut, on the **first 6 qualified trunks** in seed order | H | 1,200 + 1,200 |
| S1c-LLM | Competitive side of H8: S1 probes on the 4 CA-H runs (4 x 4 x 50 = 800) and on non-qualified TR-H trunks (4 x 50 = 200 each, at most 7 trunks = 1,400) | H | at most 2,200 (to author; editor's reading of D-21) |
| NI | Bonjour pair test (2 LLM + 2 scripted, 4 runs) | H | 800 |
| S1c-SC | Competitive sting side, scripted | CPU | 0 |
| JD | Judge on the corpus, Sonnet 5.5 | S (judge) | about 1,100 |
| CDA | Agrawal double auction, optional (engine C) | H | +1,440 |
| **Total Tier 1** | | **H 25,400** at most (authored 21,600 + GR-H s1 1,600 + S1c-LLM 2,200; +1,440 optional) + judge about 1,100 | |

## Mistral set (free lane M; H1-M cannot be cut, reported as robustness)

| ID | Condition | Calls | Status |
| --- | --- | --- | --- |
| PC-M | Gate 1 on M (Tier 0 above) | 14,400 | authored |
| TR-M | 6 trunks (seeds 0-5) x 4 x 100, same parameters as TR-H | 2,400 | to author |
| SH-M | Sham, 6 x 4 x 50 | 1,200 | to author |
| LT-M | **H1-M**: L2 + the rule-selected second lever, 2 x 6 x 4 x 50 | 2,400 | to author (rule-parametrised) |
| CA-M | "Compete aggressively", 3 seeds x 4 x 100 | 1,200 | to author |
| GH-M | Fresh-draw ghosts on the 6 M trunks, 6 x 4 x 50 | 1,200 | to author |
| S2-M | Forced-undercut probe on the qualified M trunks, 6 x 4 x 50 | 1,200 | to author |
| **Total** | | **24,000** (about 2.7 h at the 150 req/min lane limit, measured 188 req/min) | |

Drop order if throughput falls below 3k calls/h (PREREG A-1): SENS-TEMP05-M (Tier 2) -> GH-M and S2-M -> CA-M -> PC-M alpha 3.2 and 10 (keep alpha 1, 4,800 calls). Never TR-M, SH-M, LT-M.

## Sonnet 5.5 capability arm (lane S)

Question: does the L2 effect hold when the model is more capable (`keppo2026` on model scale and capability effects; Ross-So-Lo 2026, `ross_so_lo2026`, UNVERIFIED in library)? H-S tests L2 (PREREG section 2); everything else on the arm is descriptive.

| ID | Condition | Calls |
| --- | --- | --- |
| PC-S | Fish exact, 5 seeds x 2 x 300, alpha = 1 | 3,000 |
| TR-S | 8 trunks x 4 x 100 | 3,200 |
| SH-S | Sham, 8 x 4 x 50 | 1,600 |
| SH2-S | Sham2, 8 x 4 x 50 (noise check on S; as in the config) | 1,600 |
| LT-S | 3 levers: L2 + the two rule-selected levers, 3 x 8 x 4 x 50 | 4,800 |
| GH-S | **Moved to Tier 2** (truncation step 1, decided before the freeze on the measured round-100 prices); ghosts remain on Haiku (GH-H) | 0 (1,600 in Tier 2) |
| **Total (step 1)** | | **14,200** (+ PILOT-S 20) |

Judge for S-lane runs: Llama-3.3-70B via OpenRouter (lane L, paid; backup Gemini Flash-Lite). Step 2 of the truncation order drops the third LT-S lever, never L2.

### Registered truncation order (sized on the round-100, revision-3 prices)

Projected registered total = Haiku arm + Sonnet arm + lane L; the stopping point is **<= 180 USD with +10 % retries**. Steps 1-3 are the integrator's registered order; steps 4-5 make the rule terminate. The figures are derived by the editor at the pricing agent's average Sonnet cost (112 USD / 13,800 calls = 0.00812 USD/call) and are approximate; they include the SH2-S correction to 1,600 calls (8 trunks, as in the config; the pricing agent's 112 USD counted 1,200).

| Step | Cut | Sonnet arm calls | Sonnet USD | Projected total (Haiku 35 + Sonnet + lane L 10) | With +10 % retries |
| --- | --- | --- | --- | --- | --- |
| 0 | none | 15,800 | about 128 | about 173 | about 191 (> 180) |
| 1 | GH-S to Tier 2 | 14,200 | about 115 | about 160 | about 176 (**stop: step reached before freeze**) |
| 2 | LT-S 3 -> 2 levers (L2 kept) | 12,600 | about 102 | about 147 | about 162 |
| 3 | drop SH2-S | 11,000 | about 89 | about 134 | about 148 |
| 4 | Sonnet trunks 8 -> 6 (TR-S, SH-S, LT-S scale by 6/8) | 9,000 | about 73 | about 118 | about 130 |
| 5 | arm reduced to PC-S (capability check only) | 3,000 | about 24 | about 69 | about 76 |

**Sonnet cost gate (Saturday 17:00):** if the measured Sonnet USD/call exceeds the projection by more than 20 %, the order continues from step 2 for the Sonnet items not yet launched.

## Tier 2 (only if ahead; caps H-only)

Caps: H <= 16,000 calls. Menu (runbook, 33,040 H calls + 1,600 M; plus GH-S, 1,600 Sonnet calls, moved here at truncation step 1; the cap applies to what is launched): EXT-H (extra trunks, seeds 12-15, analysed separately, never pooled into H1); hysteresis (2 levers, 30 + 30 periods); prevention (fringe from the start, H3); S0 captive-flow LLM; R2 (memory kept), R4 (2 replaced); L4b (aggregated + delayed; exploratory); L3b (tick 0.125; exploratory); GR+ (more gradient seeds; the CRN-paired s1 level is now part of GR-H in Tier 1); sensitivities (P2, memory 30, alpha 3.2, temperature on M); FBA; CDA with text +/- notice. Go/no-go Sunday 08:00-11:00: cumulative Anthropic spend <= 120 USD **and** projected Tier 2 cost <= remaining reserve (195 - spent - 15 USD), analyses of Tier 0/1 finished, >= 12 h remaining.

## Budget proof

Prices (`configs/models.yaml`): Haiku 5.5 0.10 / 0.50 USD per million input / output tokens; Sonnet 5.5 2.00 / 10.00; Llama-3.3-70B via OpenRouter 0.22 / 0.50 (owner's figure, 10 Oct).

**Prompt revision 3 and memory window.** Revision 3 (`dealer_v1@e65f08608d8a`, `fish_v1@556ad0e15faa`, `fish_v1_json@600875490c6e`; final hashes at freeze) changes only the block order (MARKET_DATA first, rounds oldest first) and adds one linking sentence; two Anthropic cache breakpoints (system part; last history round), 5-minute lifetime. Memory window for every lane and arm: `history_periods` 100, `memory_step` 25 (the last 76-100 rounds, the oldest 25 dropped at once; saw-tooth), expected cached share about 88 % of input tokens. The first live period of every branch and any call delayed by more than 5 minutes is a cold write.

**Round-100 measurement (revision 3, two consecutive calls per lane; used for sizing):** Haiku 8,007 input / 990 output tokens: cold **0.001482 USD**, warm (7,465 cached) **0.000438 USD**; Sonnet cold **0.025677 USD**, warm **0.007776 USD**. Input tokens per call: 1,965 / 4,995 / 8,069 / 8,117 at rounds 1 / 50 / 100 / 150.

**Pilot (revision 2, rounds 1-5; superseded for sizing):** PILOT-H, PILOT-S on main 33ff307, 2026-10-10 about 04:45Z, `dealer_v1@2b4c0c8f546d`, seed 9001, 5 periods x 4 dealers, TR-H market, 20 calls per lane: Haiku 0.000492 USD/call (0.009841 USD in total; per call on average 2,028 input tokens, of which 1,097 cached (40,551 input tokens in total, of which 21,936 cached), plus 762 output tokens); Sonnet 0.00637 USD/call (0.1275 USD in total; per call 1,969 input tokens, of which 1,097 cached, plus 438 output tokens); 20/20 parsed, 0 repairs, 0 refusals, 0 cache hits on both lanes.

| Item | Calls | Projected USD (pricing agent, memory_step 25) |
| --- | --- | --- |
| Haiku arm (Tier 0 + Tier 1 incl. AC-H and the GR-H s1 level) | 66,820 | about 35 |
| Haiku S1c-LLM (at most) | 2,200 | about 1.2 (editor, at 35 / 66,820 = 0.000524 USD/call) |
| Haiku Tier 2 cap (if launched) | 16,000 | about 8.4 (editor, same average) |
| Haiku contingent Gate 1b repair | 4,800 | about 2.5 (editor, same average) |
| Sonnet arm at truncation step 1 | 14,200 | about 115 (pricing agent 112 for 13,800 calls, + 400 SH2-S calls) |
| Lane L: Llama-3.3-70B (Sonnet-arm judge, R3 newcomer 600 calls) | about 10,000 | <= 10 (lane budget) |
| Pilots and round-100 measure (spent) | 44 | 0.17 |
| Mistral (`ministral-8b-latest`), Gemini (free) | 24,000 + Tier 2 | 0 |

**Registered total:** Haiku 35 + Sonnet 115 + lane L 10 = **about 160 USD; about 176 USD with +10 % retries**, under the 180 USD stopping point (S1c-LLM adds about 1.3 USD with retries). Tier 2 and the contingent repair are not in this total and are bounded by the caps and the Tier 2 go/no-go.

**Caps (registered).** Lane caps in the sprint configs: **Haiku 77 USD in total, Sonnet 105.5 USD in total**; lane caps in `configs/models.yaml`: **Haiku 80 USD, Sonnet 120 USD**; lane L 10 USD; **global stop 195 USD** inside the 200 USD hard cap; alerts at 100 and 150 USD; target <= 160 USD. The Sonnet arm projection with retries (about 127 USD) is above both Sonnet caps: if a cap binds, the arm is reported as incomplete with the trunks lost (no selective relaunch), and the 17:00 cost gate continues the truncation order before that point when it fires. Concurrency: 48 Haiku and 16 Sonnet calls in flight.

## Wall-clock check (pilot latency: Haiku 5.5 s mean, Sonnet 5.1 s mean; round-100 latency may be higher)

Haiku: 48 calls in flight at about 5.5 s/call = about 8.7 calls/s = about 31k calls/h; Tier 0 + Tier 1 H (69,020 calls incl. S1c-LLM) = about 2.2 h of saturated lane. Sonnet: 16 in flight at about 5.1 s/call = about 3.1 calls/s = about 11k calls/h; the arm at step 1 (14,200 calls) = about 1.3 h. Gate 1 / Gate 1b need at least 28 min for their 300 sequential periods, consistent with Gate 1 at about 16:30. The paired Tier 0 frontier is 12 trunks x (5 levers + sham + sham2) = 84 branches of 50 periods plus 48 ghost markets (19,200 calls, about 0.6 h of lane). Mistral set (24,000 calls) about 2.7 h at 150 req/min, in parallel.

Strict-zero-dollar variant (kept as contingency): frontier on M at the measured Mistral rate (about 9,000 calls/h) and a reduced Tier 1.
