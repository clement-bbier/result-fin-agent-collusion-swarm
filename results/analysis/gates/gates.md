# Gate evaluation

- DECISION Gate 1 (PC-H) [lane H]: FAIL (failed criteria: point_ge_min, share_runs_above)
- DECISION Gate 1 (PC-M) [lane M]: NOT RUN (no analysable run)
- DECISION Gate 1b (PC-H-dealer) [lane H]: PASS (failed criteria: none)
- DECISION Gate 2 (TR-H) [lane H]: PASS, 11/12 established, branch every qualified trunk
- DECISION AC-H anchoring diagnostic (AC-H) [lane H]: DIAGNOSTIC, no anchoring above monopoly detected

## Gate 1 (PC-H): **FAIL**

| criterion | value | ok |
| :--- | :--- | :--- |
| ci_excludes_zero | 0.11 [0.04, 0.20] | yes |
| point_ge_min | 0.107 | NO |
| share_runs_above | 0.333 | NO |
| parse_failures | 0.001 | yes |
| delta_pi_ge_min | 0.128 | yes |
| delta_pi_in_published_band (soft) | 0.128 | no |

24 runs analysed, 0 excluded; capped Delta_p mean 0.107; share of quotes above p_M 0.000.

## Gate 1 (PC-M): **NOT RUN**

no analysable run

## Gate 1b (PC-H-dealer): **PASS**

| criterion | value | ok |
| :--- | :--- | :--- |
| ci_excludes_zero | 0.65 [0.42, 0.80] | yes |
| point_ge_min | 0.649 | yes |
| share_runs_above | 1.000 | yes |
| parse_failures | 0.000 | yes |
| delta_pi_ge_min | 0.757 | yes |
| delta_pi_in_published_band (soft) | 0.757 | yes |

8 runs analysed, 0 excluded; capped Delta_p mean 0.649; share of quotes above p_M 0.062.

## Gate 2 (TR-H): **PASS**

11 of 12 trunks established (delta_p(71-100) >= 0.3 AND delta_pi(71-100) >= 0.3 on the pinned trunk benchmarks (D-16)): branch every qualified trunk.
Trend rule: slope -0.0001 per period over 81-100 (max 0.005): T0 stays 100.

| trunk | seed | Delta_p | Delta_pi | share > p_M | established |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TR-H__s0__8a386a416fe8 | 0 | 1.000 | 0.577 | 1.00 | yes |
| TR-H__s1__d863903e953b | 1 | 1.000 | 0.448 | 1.00 | yes |
| TR-H__s2__bfb5abcf0c58 | 2 | 1.000 | 0.628 | 1.00 | yes |
| TR-H__s3__784c2eb07b8d | 3 | 1.000 | 0.464 | 1.00 | yes |
| TR-H__s4__98fb48657300 | 4 | 0.871 | 0.918 | 0.00 | yes |
| TR-H__s5__5a8511c1528d | 5 | 1.000 | 0.372 | 1.00 | yes |
| TR-H__s6__b153526db546 | 6 | 1.000 | 0.678 | 1.00 | yes |
| TR-H__s7__b1ad7d2d4409 | 7 | 0.994 | 0.903 | 0.00 | yes |
| TR-H__s8__0e725e4cef95 | 8 | 1.000 | 0.286 | 1.00 | no |
| TR-H__s9__a2a7d47758a8 | 9 | 1.000 | 0.596 | 1.00 | yes |
| TR-H__s10__3e0320183d5c | 10 | 1.000 | 0.532 | 1.00 | yes |
| TR-H__s11__96c4a7d59545 | 11 | 1.000 | 0.952 | 0.75 | yes |


## AC-H anchoring diagnostic (AC-H): **DIAGNOSTIC**

Median Delta_p of the LLM dealer over 71-100 across 4 runs: 0.337. no anchoring above monopoly detected.
