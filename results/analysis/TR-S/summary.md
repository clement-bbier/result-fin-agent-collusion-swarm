# Analysis — TR-S

Registry `configs\analysis\registered_arms.yaml` (sha256 0bc079093891); windows: branch 111–150, trunk 71–100; runs analysed: 22; excluded: 0; ignored (unregistered): 6; bootstrap B = 10000, seed 20261010.

Primary estimand: delta_p: incumbents' CAPPED price index min(Delta_p, 1) with the pre-lever (trunk) benchmarks (D-16; raw index reported as delta_p_raw). Tests: one-sided (alternative='greater'): H1 is Delta(sham) - Delta(arm) > 0, a reduction of collusion; BCa intervals are two-sided 95 %; TOST is two one-sided tests by construction.

Provisional (closed-form fringe) benchmarks: none; re-pin with scripts/analyze.py --make-benchmarks before the registered analysis.

## Arms

| arm | n | Delta [95% CI] |
| :--- | ---: | :--- |
| trunk_S | 8 | 0.99 [0.96, 1.00] |
| sham_S | 8 | 0.99 [0.95, 1.00] |
| L2_S | 6 | 0.97 [0.91, 1.00] |

## H1 — primary lever L2 (no multiplicity; capped index, both scorings)

Gate 2 outcome: not tested: no paired contrast.

n = 0 qualified trunks; ΔC_p (trunk scoring) = – [–, –] (BCa 95 %); one-sided p = – (Wilcoxon sensitivity 1.000); own scoring ΔC_p = – [–, –], one-sided p = –; MQC 26.53 pp. Label (D-17): no significant reduction. Supported: no. Practical significance (D-19): not applicable (one-sided 95 % lower bound –; 90 % t interval – [–, –]; BCa 90 % – [–, –]); TOST ±0.20: not shown.

## H-S — lever L2_S on lane S (claude-sonnet-5-5; tested (8 qualified trunks))

n = 6 qualified trunks; ΔC_p (trunk scoring) = 0.01 [0.00, 0.03], one-sided p = 0.094; own scoring p = 0.099; label: no significant reduction; supported: no. capability arm, secondary outside family S

## H1-R — paired contrasts (primary Delta_p; Holm over m = 8)

| contrast | n pairs | ΔC [BCa 95% CI] | p (t, one-sided) | p (Wilcoxon) | p Holm (m = 8) | TOST ±0.20 | dual-benchmark (D-17) |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- |
| sham_S − L2_S | 6 | 0.01 [0.00, 0.03] | 0.094 | 0.250 | – | equivalent | no significant reduction |

Missing levers counted at p = 1: L1, L2, L3, L4a, L5, L6, L7, L8.

## Mechanical vs behavioural Delta C (levers)

_(no lever contrasts)_

## Empirical best lever (descriptive split sample)

Status: off: 0 selection and 0 estimation trunks, fewer than 3 per half (D-22). Selected on odd-seed trunks (–): none; estimated on even-seed trunks (–): – [–, –]. Descriptive only, no decision. sigma_d (realised, pooled lever-minus-sham) = –.
