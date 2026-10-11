# Analysis — TR-H (BLIND)

**Blind analysis.** Lever arms are shown as letters; the key is outside this folder (sha256 `dd44c47b0d1138f00826a1f34b9635c1ebb1774d824deafec1fdd208eae81df6`). Commit this file before unblinding (PREREG section 5).

Registry `configs\analysis\registered_arms.yaml` (sha256 ec121ec5edba); windows: branch 111–150, trunk 71–100; runs analysed: 108; excluded: 12; ignored (unregistered): 48; bootstrap B = 10000, seed 20261010.

Primary estimand: delta_p: incumbents' CAPPED price index min(Delta_p, 1) with the pre-lever (trunk) benchmarks (D-16; raw index reported as delta_p_raw). Tests: one-sided (alternative='greater'): H1 is Delta(sham) - Delta(arm) > 0, a reduction of collusion; BCa intervals are two-sided 95 %; TOST is two one-sided tests by construction.

Provisional (closed-form fringe) benchmarks: blinded, n; re-pin with scripts/analyze.py --make-benchmarks before the registered analysis.

## Arms

| arm | n | Delta [95% CI] |
| :--- | ---: | :--- |
| trunk | 12 | 0.99 [0.95, 1.00] |
| sham | 12 | 0.99 [0.95, 1.00] |
| sham2 | 12 | 0.99 [0.95, 1.00] |
| G | 12 | 0.99 [0.94, 1.00] |
| D | 12 | 0.99 [0.95, 1.00] |
| A | 12 | 0.99 [0.95, 1.00] |
| C | 12 | 0.99 [0.95, 1.00] |
| B | 12 | 0.38 [0.26, 0.55] |
| S1 | 6 | 0.98 [0.92, 1.00] |
| S2 | 6 | 0.96 [0.83, 1.00] |

## H1 — primary lever D (no multiplicity; capped index, both scorings)

Gate 2 outcome: tested (11 qualified trunks).

n = 11 qualified trunks; ΔC_p (trunk scoring) = 0.00 [-0.00, 0.02] (BCa 95 %); one-sided p = 0.208 (Wilcoxon sensitivity 0.500); own scoring ΔC_p = 0.00 [-0.00, 0.01], one-sided p = 0.255; MQC 26.53 pp. Label (D-17): no significant reduction. Supported: no. Practical significance (D-19): below practical relevance (one-sided 95 % lower bound -0.00; 90 % t interval 0.00 [-0.00, 0.01]; BCa 90 % 0.00 [-0.00, 0.01]); TOST ±0.20: equivalent.

## H1-R — paired contrasts (primary Delta_p; Holm over m = 8)

| contrast | n pairs | ΔC [BCa 95% CI] | p (t, one-sided) | p (Wilcoxon) | p Holm (m = 8) | TOST ±0.20 | dual-benchmark (D-17) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| sham − sham2 | 11 | 0.00 [0.00, 0.00] | 1.000 | 1.000 | – | equivalent | no significant reduction |
| sham − G | 11 | 0.00 [0.00, 0.00] | 0.170 | 0.500 | 1.000 | equivalent | no significant reduction |
| sham − D | 11 | 0.00 [-0.00, 0.02] | 0.208 | 0.500 | 1.000 | equivalent | no significant reduction |
| sham − A | 11 | 0.00 [0.00, 0.00] | 0.170 | 0.500 | 1.000 | equivalent | no significant reduction |
| sham − C | 11 | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | equivalent | no significant reduction |
| sham − B | 11 | 0.62 [0.42, 0.75] | <0.001 | <0.001 | <0.001 | not shown | collusion reduction |

Missing levers counted at p = 1: E, H, F.

## Mechanical vs behavioural Delta C (levers)

| lever | measure | total ΔC | mechanical | behavioural |
| :--- | :--- | :--- | :--- | :--- |
| G | delta_p | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| G | delta_pi_window | 0.33 [0.26, 0.38] | 0.50 [0.33, 0.60] | -0.17 [-0.24, -0.08] |
| D | delta_p | 0.00 [-0.00, 0.02] | 0.00 [0.00, 0.00] | 0.00 [-0.00, 0.02] |
| D | delta_pi_window | 0.01 [-0.03, 0.05] | -0.40 [-0.47, -0.34] | 0.41 [0.34, 0.47] |
| A | delta_p | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| A | delta_pi_window | 0.00 [0.00, 0.01] | -0.00 [-0.00, -0.00] | 0.00 [0.00, 0.01] |
| C | delta_p | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| C | delta_pi_window | 0.00 [-0.00, 0.01] | 0.00 [0.00, 0.00] | 0.00 [-0.00, 0.01] |
| B | delta_p | 0.62 [0.42, 0.75] | 0.00 [0.00, 0.00] | 0.62 [0.42, 0.75] |
| B | delta_pi_window | 0.41 [0.31, 0.49] | 0.66 [0.60, 0.70] | -0.25 [-0.35, -0.17] |

## Empirical best lever (descriptive split sample)

Status: ok. Selected on odd-seed trunks (TR-H__s11__96c4a7d59545, TR-H__s1__d863903e953b, TR-H__s3__784c2eb07b8d, TR-H__s5__5a8511c1528d, TR-H__s7__b1ad7d2d4409, TR-H__s9__a2a7d47758a8): B; estimated on even-seed trunks (TR-H__s0__8a386a416fe8, TR-H__s10__3e0320183d5c, TR-H__s2__bfb5abcf0c58, TR-H__s4__98fb48657300, TR-H__s6__b153526db546): 0.62 [0.45, 0.77]. Descriptive only, no decision. sigma_d (realised, pooled lever-minus-sham) = 0.124.

## Exclusions

| run_id | arm | reason |
| :--- | :--- | :--- |
| blind:419b5f5ffff9 | E | manifest status 'running' (not ok) |
| blind:f5043a4fcd7c | E | manifest status 'running' (not ok) |
| blind:7299c87fb560 | E | manifest status 'running' (not ok) |
| blind:cc6e7b8ddbfb | E | manifest status 'running' (not ok) |
| blind:d7bff5cfc328 | E | manifest status 'running' (not ok) |
| blind:05f18d99f361 | E | manifest status 'running' (not ok) |
| blind:fcb8f3fe2216 | E | manifest status 'running' (not ok) |
| blind:514e365b1b4c | E | manifest status 'running' (not ok) |
| blind:20c6b6e0ef81 | E | manifest status 'running' (not ok) |
| blind:7a971e065357 | E | manifest status 'running' (not ok) |
| blind:191a72b2b183 | E | manifest status 'running' (not ok) |
| blind:de2581646db7 | E | manifest status 'running' (not ok) |
