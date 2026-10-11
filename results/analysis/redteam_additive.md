# Red-team additive descriptives (exploratory, no registered value changes)

## (a) Sham cache reuse after the branch

post-branch (periods 101-150) incumbent LLM decisions (slots 0-3) of the qualified trunks; from_sham = cache_hit true and the same-trunk sham logged a live call (cache_hit false) at the same period and slot with the same prompt hash and the same raw output; share = n_from_sham / n_decisions (pooled); all_trunks pools every trunk the arm ran on

| arm | decisions | from sham (share) | any cache hit (share) | prompt = sham's | and output = sham's |
|---|---|---|---|---|---|
| L1 (qualified, 11) | 2200 | 0 (0.0000) | 0 (0.0000) | 44 | 0 |
| L1 (all trunks, 12) | 2400 | 0 (0.0000) | 0 (0.0000) | 48 | 0 |
| L2 (qualified, 11) | 2200 | 0 (0.0000) | 0 (0.0000) | 44 | 0 |
| L2 (all trunks, 12) | 2400 | 0 (0.0000) | 0 (0.0000) | 48 | 0 |
| L3 (qualified, 11) | 2200 | 0 (0.0000) | 0 (0.0000) | 93 | 0 |
| L3 (all trunks, 12) | 2400 | 0 (0.0000) | 0 (0.0000) | 98 | 0 |
| L4a (qualified, 11) | 2200 | 0 (0.0000) | 0 (0.0000) | 0 | 0 |
| L4a (all trunks, 12) | 2400 | 0 (0.0000) | 0 (0.0000) | 0 | 0 |
| L5 (qualified, 11) | 2200 | 0 (0.0000) | 153 (0.0695) | 0 | 0 |
| L5 (all trunks, 12) | 2392 | 0 (0.0000) | 153 (0.0640) | 1 | 0 |
| L6 (qualified, 11) | 2200 | 0 (0.0000) | 0 (0.0000) | 44 | 0 |
| L6 (all trunks, 12) | 2400 | 0 (0.0000) | 0 (0.0000) | 48 | 0 |
| L7 (qualified, 11) | 2200 | 0 (0.0000) | 0 (0.0000) | 82 | 0 |
| L7 (all trunks, 12) | 2400 | 0 (0.0000) | 0 (0.0000) | 88 | 0 |
| L6b (qualified, 11) | 2200 | 0 (0.0000) | 0 (0.0000) | 44 | 0 |
| L6b (all trunks, 12) | 2400 | 0 (0.0000) | 0 (0.0000) | 48 | 0 |
| L6c (qualified, 11) | 2200 | 0 (0.0000) | 0 (0.0000) | 44 | 0 |
| L6c (all trunks, 12) | 2400 | 0 (0.0000) | 0 (0.0000) | 48 | 0 |
| R1 (qualified, 11) | 2200 | 0 (0.0000) | 0 (0.0000) | 47 | 0 |
| R1 (all trunks, 12) | 2400 | 0 (0.0000) | 0 (0.0000) | 50 | 0 |
| L9 (qualified, 11) | 2200 | 0 (0.0000) | 0 (0.0000) | 0 | 0 |
| L9 (all trunks, 12) | 2400 | 0 (0.0000) | 0 (0.0000) | 0 | 0 |
| R2 (qualified, 11) | 2200 | 0 (0.0000) | 0 (0.0000) | 92 | 0 |
| R2 (all trunks, 12) | 2400 | 0 (0.0000) | 0 (0.0000) | 98 | 0 |
| R4 (qualified, 11) | 2200 | 0 (0.0000) | 0 (0.0000) | 28 | 0 |
| R4 (all trunks, 12) | 2400 | 0 (0.0000) | 0 (0.0000) | 30 | 0 |
| S1 (qualified, 6) | 1200 | 0 (0.0000) | 0 (0.0000) | 24 | 0 |
| S1 (all trunks, 6) | 1200 | 0 (0.0000) | 0 (0.0000) | 24 | 0 |
| S2 (qualified, 6) | 1194 | 0 (0.0000) | 144 (0.1206) | 18 | 0 |
| S2 (all trunks, 6) | 1194 | 0 (0.0000) | 144 (0.1206) | 18 | 0 |
| S1c_LLM_TR (qualified, 3) | 597 | 0 (0.0000) | 597 (1.0000) | 9 | 0 |
| S1c_LLM_TR (all trunks, 3) | 597 | 0 (0.0000) | 597 (1.0000) | 9 | 0 |
| sham2 (qualified, 11) | 2200 | 0 (0.0000) | 0 (0.0000) | 93 | 0 |
| sham2 (all trunks, 12) | 2400 | 0 (0.0000) | 0 (0.0000) | 100 | 0 |

Positive control (the same rule with S2 as the reference: S1c-LLM-TR-H decisions served from the same-trunk S2 run's answers (a known replay)): 541 of 597 decisions (0.9062).

Ledger cross-check (post-branch calls, every trunk the arm ran on; hits attributed to their writer by request_id):

| arm | trunks | live calls | cache hits | writer of the hit entries |
|---|---|---|---|---|
| sham | 12 | 2400 | 0 | - |
| L1 | 12 | 2400 | 0 | - |
| L2 | 12 | 2400 | 0 | - |
| L3 | 12 | 2400 | 0 | - |
| L4a | 12 | 2400 | 0 | - |
| L5 | 12 | 3679 | 274 | same run: 274 |
| L6 | 12 | 2402 | 0 | - |
| L7 | 12 | 2400 | 0 | - |
| L6b | 12 | 2400 | 0 | - |
| L6c | 12 | 2400 | 0 | - |
| R1 | 12 | 2400 | 0 | - |
| L9 | 12 | 2400 | 0 | - |
| R2 | 12 | 2400 | 0 | - |
| R4 | 12 | 2400 | 0 | - |
| S1 | 6 | 1200 | 0 | - |
| S2 | 6 | 1194 | 144 | same run: 144 |
| S1c_LLM_TR | 3 | 0 | 597 | S2 (same trunk): 597 |
| sham2 | 12 | 2400 | 0 | - |

## (b) L6c minus L6 (entrant type, exploratory)

Delta C(L6c) - Delta C(L6) = capped idx(L6) - capped idx(L6c) per trunk (registered per-run delta_p, trunk benchmarks, window 111-150), BCa 95 %

n 11: 0.014 [-0.146, 0.155]
