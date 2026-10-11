# Analysis — AC-H

Registry `configs\analysis\registered_arms.yaml` (sha256 f149e6e50788); windows: branch 111–150, trunk 71–100; runs analysed: 4; excluded: 0; ignored (unregistered): 0; bootstrap B = 10000, seed 20261010.

Primary estimand: delta_p: incumbents' CAPPED price index min(Delta_p, 1) with the pre-lever (trunk) benchmarks (D-16; raw index reported as delta_p_raw). Tests: one-sided (alternative='greater'): H1 is Delta(sham) - Delta(arm) > 0, a reduction of collusion; BCa intervals are two-sided 95 %; TOST is two one-sided tests by construction.

Provisional (closed-form fringe) benchmarks: none; re-pin with scripts/analyze.py --make-benchmarks before the registered analysis.

## Arms

| arm | n | Delta [95% CI] |
| :--- | ---: | :--- |
| AC-H | 4 | 0.32 [0.09, 0.51] |

## H1 — primary lever L2 (no multiplicity; capped index, both scorings)

Gate 2 outcome: not tested: no paired contrast.

n = 0 qualified trunks; ΔC_p (trunk scoring) = – [–, –] (BCa 95 %); one-sided p = – (Wilcoxon sensitivity 1.000); own scoring ΔC_p = – [–, –], one-sided p = –; MQC – pp. Label (D-17): no significant reduction. Supported: no. Practical significance (D-19): not applicable (one-sided 95 % lower bound –; 90 % t interval – [–, –]; BCa 90 % – [–, –]); TOST ±0.20: not shown.

## Empirical best lever (descriptive split sample)

Status: off: 0 selection and 0 estimation trunks, fewer than 3 per half (D-22). Selected on odd-seed trunks (–): none; estimated on even-seed trunks (–): – [–, –]. Descriptive only, no decision. sigma_d (realised, pooled lever-minus-sham) = –.
