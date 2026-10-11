# Analysis — TR-H

Registry `configs\analysis\registered_arms.yaml` (sha256 0222331f992c); windows: branch 111–150, trunk 71–100; runs analysed: 255; excluded: 24; ignored (unregistered): 0; bootstrap B = 10000, seed 20261010.

Primary estimand: delta_p: incumbents' CAPPED price index min(Delta_p, 1) with the pre-lever (trunk) benchmarks (D-16; raw index reported as delta_p_raw). Tests: one-sided (alternative='greater'): H1 is Delta(sham) - Delta(arm) > 0, a reduction of collusion; BCa intervals are two-sided 95 %; TOST is two one-sided tests by construction.

Provisional (closed-form fringe) benchmarks: none; re-pin with scripts/analyze.py --make-benchmarks before the registered analysis.

## Arms

| arm | n | Delta [95% CI] |
| :--- | ---: | :--- |
| trunk | 12 | 0.99 [0.95, 1.00] |
| sham | 12 | 0.99 [0.95, 1.00] |
| sham2 | 12 | 0.99 [0.95, 1.00] |
| L1 | 12 | 0.99 [0.94, 1.00] |
| L2 | 12 | 0.99 [0.95, 1.00] |
| L3 | 12 | 0.99 [0.95, 1.00] |
| L4a | 12 | 0.99 [0.95, 1.00] |
| L5 | 12 | 0.99 [0.95, 1.00] |
| L6 | 12 | 0.38 [0.26, 0.55] |
| L7 | 12 | 0.99 [0.95, 1.00] |
| ghost | 48 | 0.99 [0.97, 1.00] |
| S1 | 6 | 0.98 [0.92, 1.00] |
| S2 | 6 | 0.96 [0.83, 1.00] |
| S1c_LLM_TR | 3 | 1.00 [1.00, 1.00] |
| R1 | 12 | 0.99 [0.95, 1.00] |
| L6b | 12 | 0.96 [0.88, 0.99] |
| L6c | 12 | 0.38 [0.29, 0.50] |
| R2 | 12 | 0.99 [0.95, 1.00] |
| R4 | 12 | 0.99 [0.95, 1.00] |
| L9 | 12 | 0.99 [0.95, 1.00] |

## H1 — primary lever L2 (no multiplicity; capped index, both scorings)

Gate 2 outcome: tested (11 qualified trunks).

n = 11 qualified trunks; ΔC_p (trunk scoring) = 0.00 [-0.00, 0.02] (BCa 95 %); one-sided p = 0.208 (Wilcoxon sensitivity 0.500); own scoring ΔC_p = 0.00 [-0.00, 0.01], one-sided p = 0.255; MQC 26.53 pp. Label (D-17): no significant reduction. Supported: no. Practical significance (D-19): below practical relevance (one-sided 95 % lower bound -0.00; 90 % t interval 0.00 [-0.00, 0.01]; BCa 90 % 0.00 [-0.00, 0.01]); TOST ±0.20: equivalent.

## H1-R — paired contrasts (primary Delta_p; Holm over m = 8)

| contrast | n pairs | ΔC [BCa 95% CI] | p (t, one-sided) | p (Wilcoxon) | p Holm (m = 8) | TOST ±0.20 | dual-benchmark (D-17) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| sham − sham2 | 11 | 0.00 [0.00, 0.00] | 1.000 | 1.000 | – | equivalent | no significant reduction |
| sham − L1 | 11 | 0.00 [0.00, 0.00] | 0.170 | 0.500 | 1.000 | equivalent | no significant reduction |
| sham − L2 | 11 | 0.00 [-0.00, 0.02] | 0.208 | 0.500 | 1.000 | equivalent | no significant reduction |
| sham − L3 | 11 | 0.00 [0.00, 0.00] | 0.170 | 0.500 | 1.000 | equivalent | no significant reduction |
| sham − L4a | 11 | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | equivalent | no significant reduction |
| sham − L5 | 11 | -0.00 [-0.00, 0.00] | 0.830 | 1.000 | 1.000 | equivalent | no significant reduction |
| sham − L6 | 11 | 0.62 [0.42, 0.75] | <0.001 | <0.001 | <0.001 | not shown | collusion reduction |
| sham − L7 | 11 | 0.00 [0.00, 0.00] | 1.000 | 1.000 | 1.000 | equivalent | no significant reduction |
| sham − R1 | 11 | 0.00 [0.00, 0.00] | 1.000 | 1.000 | – | equivalent | no significant reduction |
| sham − L6b | 11 | 0.03 [0.01, 0.07] | 0.043 | 0.125 | – | equivalent | collusion reduction |
| sham − L6c | 11 | 0.63 [0.53, 0.72] | <0.001 | <0.001 | – | not shown | collusion reduction |
| sham − R2 | 11 | 0.00 [0.00, 0.00] | 1.000 | 1.000 | – | equivalent | no significant reduction |
| sham − R4 | 11 | 0.00 [0.00, 0.00] | 1.000 | 1.000 | – | equivalent | no significant reduction |
| sham − L9 | 11 | 0.00 [0.00, 0.00] | 0.170 | 0.500 | – | equivalent | no significant reduction |

Missing levers counted at p = 1: L8.

## Mechanical vs behavioural Delta C (levers)

| lever | measure | total ΔC | mechanical | behavioural |
| :--- | :--- | :--- | :--- | :--- |
| L1 | delta_p | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| L1 | delta_pi_window | 0.33 [0.26, 0.38] | 0.50 [0.33, 0.60] | -0.17 [-0.24, -0.08] |
| L2 | delta_p | 0.00 [-0.00, 0.02] | 0.00 [0.00, 0.00] | 0.00 [-0.00, 0.02] |
| L2 | delta_pi_window | 0.01 [-0.03, 0.05] | -0.40 [-0.47, -0.34] | 0.41 [0.34, 0.47] |
| L3 | delta_p | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| L3 | delta_pi_window | 0.00 [0.00, 0.01] | -0.00 [-0.00, -0.00] | 0.00 [0.00, 0.01] |
| L4a | delta_p | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| L4a | delta_pi_window | 0.00 [-0.00, 0.01] | 0.00 [0.00, 0.00] | 0.00 [-0.00, 0.01] |
| L5 | delta_p | -0.00 [-0.00, 0.00] | 0.00 [0.00, 0.00] | -0.00 [-0.00, 0.00] |
| L5 | delta_pi_window | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |
| L6 | delta_p | 0.62 [0.42, 0.75] | 0.00 [0.00, 0.00] | 0.62 [0.42, 0.75] |
| L6 | delta_pi_window | 0.41 [0.31, 0.49] | 0.66 [0.60, 0.70] | -0.25 [-0.35, -0.17] |
| L7 | delta_p | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| L7 | delta_pi_window | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |

## Empirical best lever (descriptive split sample)

Status: ok. Selected on odd-seed trunks (TR-H__s11__96c4a7d59545, TR-H__s1__d863903e953b, TR-H__s3__784c2eb07b8d, TR-H__s5__5a8511c1528d, TR-H__s7__b1ad7d2d4409, TR-H__s9__a2a7d47758a8): L6; estimated on even-seed trunks (TR-H__s0__8a386a416fe8, TR-H__s10__3e0320183d5c, TR-H__s2__bfb5abcf0c58, TR-H__s4__98fb48657300, TR-H__s6__b153526db546): 0.62 [0.45, 0.77]. Descriptive only, no decision. sigma_d (realised, pooled lever-minus-sham) = 0.104.

## Exclusions

| run_id | arm | reason |
| :--- | :--- | :--- |
| TR-H--L6b__s0__91500c7bb715 | L6b | empty analysis window |
| TR-H--L6b__s10__2769dd50eba3 | L6b | empty analysis window |
| TR-H--L6b__s11__aa8fc3fc79ff | L6b | empty analysis window |
| TR-H--L6b__s1__7a195072b362 | L6b | empty analysis window |
| TR-H--L6b__s2__bd63459f8fed | L6b | empty analysis window |
| TR-H--L6b__s3__61104b86873f | L6b | empty analysis window |
| TR-H--L6b__s4__f90f8b7a55c8 | L6b | empty analysis window |
| TR-H--L6b__s5__4755f61cd12f | L6b | empty analysis window |
| TR-H--L6b__s6__b261f5f327d4 | L6b | empty analysis window |
| TR-H--L6b__s7__412e7cbcb80d | L6b | empty analysis window |
| TR-H--L6b__s8__b9b581723f3d | L6b | empty analysis window |
| TR-H--L6b__s9__26a032597ada | L6b | empty analysis window |
| TR-H--R3__s0__d1af60b3db8c | R3 | failed decisions 19.5% > 10% |
| TR-H--R3__s10__e283a3c8c6ec | R3 | failed decisions 24.5% > 10% |
| TR-H--R3__s11__953977875dcd | R3 | failed decisions 23.0% > 10% |
| TR-H--R3__s1__7340bf8c6a10 | R3 | failed decisions 20.0% > 10% |
| TR-H--R3__s2__ccfbea4705ba | R3 | failed decisions 21.0% > 10% |
| TR-H--R3__s3__c0a8d5853ebe | R3 | failed decisions 18.5% > 10% |
| TR-H--R3__s4__f9d373304d34 | R3 | failed decisions 23.5% > 10% |
| TR-H--R3__s5__eb30dc97c47b | R3 | failed decisions 19.5% > 10% |
| TR-H--R3__s6__a3af4772b82a | R3 | failed decisions 20.5% > 10% |
| TR-H--R3__s7__e0987643a830 | R3 | failed decisions 23.5% > 10% |
| TR-H--R3__s8__3ee7c109c5c7 | R3 | failed decisions 20.5% > 10% |
| TR-H--R3__s9__d316266c25da | R3 | failed decisions 23.0% > 10% |
