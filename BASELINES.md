# Confusion Probe — Baseline Comparison

Generated: 2026-05-18T13:02:58.457788

Records: 364 | Activations: True

## Results

```
Method                                         AUC     ±  PR-AUC   Δ ref  p_perm
----------------------------------------------------------------------------------
B0_strat L50%: Qwen probe (StratifiedKFold   0.872 0.015   0.847  +0.040       —
B0 L50%: Qwen probe (GroupKFold, layer 50%   0.831 0.039   0.804  +0.000   0.000
S1: Structural features only                 0.807 0.026   0.769               —
T1: TF-IDF on target text                    0.775 0.041   0.789               —
T2: TF-IDF on full transcript                0.843 0.042   0.862           0.000
H1: Hedge lexicon counts                     0.571 0.026   0.576               —
B1 L50%: Qwen + structural (concat)          0.839 0.029   0.808  +0.008       —
B2 L50%: Qwen probe after partialing struc   0.735 0.035   0.706  -0.096       —
B0_L25 L25%: Qwen probe GroupKFold layer 2   0.814 0.037   0.808               —
B0_L75 L75%: Qwen probe GroupKFold layer 7   0.791 0.022   0.771               —
A1_B0 L50%: Qwen probe, length-matched sub   0.908 0.050   0.910  +0.077       —
A1_S1: Structural only, length-matched       0.884 0.038   0.880               —
A1_T2: TF-IDF full transcript, length-matc   0.840 0.116   0.863               —
A2_agent_caveat_B0 L50%: Qwen probe, agent   0.880 0.043   0.710  +0.048       —
A2_agent_caveat_T1: TF-IDF target, agent_c   0.783 0.087   0.630               —
A2_missing_context_B0 L50%: Qwen probe, mi   0.862 0.039   0.836  +0.031       —
A2_missing_context_T1: TF-IDF target, miss   0.904 0.027   0.880               —
A2_user_doubt_B0 L50%: Qwen probe, user_do   0.870 0.074   0.492  +0.039       —
A2_user_doubt_T1: TF-IDF target, user_doub   0.707 0.065   0.201               —
E1_target: Sentence-transformer on target    0.757 0.064   0.729               —
E1_full: Sentence-transformer on full tran   0.731 0.045   0.739               —
Z1: Zero-shot Qwen3-4B judge P(yes)          0.732 0.056   0.741               —
```

## Interpretation
Reference: **Qwen probe GroupKFold layer 50% (B0) = 0.831±0.039**
- **Thread leakage check:** StratifiedKFold 0.872 vs GroupKFold 0.831 (Δ=+0.040). GroupKFold lowers AUC — prior result was inflated by repeated threads.
- **T2 TF-IDF full transcript: 0.843** (Δ vs B0 = -0.012). **Probe ≈ bag-of-words** on the same serialized text.
- **Structural only (S1): 0.807** — 97% of probe AUC without reading text.
- **B2 residualized activations: 0.735** (drop 0.096 vs B0). Large drop — structural confounds are a major component.
- **Length-matched subset:** B0=0.908, T2=0.840 (n=162).
- **Sentence-transformer (E1 full): 0.731**
- **Zero-shot Qwen judge (Z1): 0.732**
- **user_doubt subset** (hardest category — label depends on user reaction):
  - Qwen B0: 0.870 (14 positives)
  - TF-IDF T1: 0.707
- **Permutation p-value (B0): 0.0000** (empirical, 200 shuffles).
## Decision criteria

| Criterion | Outcome |
|-----------|--------|
| T2 within 0.02 of B0 | Probe ≈ bag-of-words |
| B2 collapses (<0.60) | Probe rode on structural confounds |
| A1 length-matched collapses | Same |
| B0 beats T2+E1 by ≥0.04 on A1 and user_doubt | Non-text signal survives |
