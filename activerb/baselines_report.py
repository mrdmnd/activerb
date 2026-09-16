"""Generate BASELINES.md interpretation from baseline results."""

from __future__ import annotations

from typing import Any

from activerb.baselines_eval import EvalResult, format_results_table

AUC_NEAR_TIE = 0.02
AUC_MODEST_GAP = 0.04
AUC_COLLAPSE = 0.6
STRUCTURAL_DROP = 0.05


def _get(results: list[EvalResult], method_id: str) -> EvalResult | None:
    return next((r for r in results if r.method_id == method_id), None)


def interpret_results(results: list[EvalResult]) -> str:  # noqa: PLR0912
    b0 = _get(results, "B0")
    b0_strat = _get(results, "B0_strat")
    t2 = _get(results, "T2")
    s1 = _get(results, "S1")
    b2 = _get(results, "B2")
    a1_b0 = _get(results, "A1_B0")
    a1_t2 = _get(results, "A1_T2")
    e1_full = _get(results, "E1_full")
    z1 = _get(results, "Z1")

    lines = ["## Interpretation\n"]

    if b0 is None:
        lines.append(
            "Qwen activation baselines were not run (missing `features_v2.pkl`). "
            "Text/structural baselines below still test redundancy with bag-of-words.\n"
        )
        if t2:
            lines.append(
                f"- **TF-IDF full transcript (T2): AUC={t2.auc_mean:.3f}** — compare to reported "
                f"Qwen probe AUC≈0.875 from REPORT.md.\n"
            )
        if s1:
            lines.append(
                f"- **Structural features (S1): AUC={s1.auc_mean:.3f}** — length/position/tool-count "
                "alone explain much of the label.\n"
            )
        return "".join(lines)

    ref_auc = b0.auc_mean
    lines.append(f"Reference: **Qwen probe GroupKFold layer 50% (B0) = {ref_auc:.3f}±{b0.auc_std:.3f}**\n")

    if b0_strat:
        delta_leak = b0_strat.auc_mean - ref_auc
        lines.append(
            f"- **Thread leakage check:** StratifiedKFold {b0_strat.auc_mean:.3f} vs GroupKFold "
            f"{ref_auc:.3f} (Δ={delta_leak:+.3f}). "
        )
        if abs(delta_leak) < AUC_NEAR_TIE:
            lines.append("Thread leakage is small.\n")
        else:
            lines.append("GroupKFold lowers AUC — prior result was inflated by repeated threads.\n")

    if t2:
        gap = ref_auc - t2.auc_mean
        lines.append(f"- **T2 TF-IDF full transcript: {t2.auc_mean:.3f}** (Δ vs B0 = {gap:+.3f}). ")
        if abs(gap) <= AUC_NEAR_TIE:
            lines.append("**Probe ≈ bag-of-words** on the same serialized text.\n")
        elif gap > AUC_MODEST_GAP:
            lines.append("Qwen adds modest signal beyond TF-IDF.\n")
        else:
            lines.append("Inconclusive — small gap.\n")

    if s1:
        lines.append(
            f"- **Structural only (S1): {s1.auc_mean:.3f}** — "
            f"{100 * s1.auc_mean / max(ref_auc, 1e-6):.0f}% of probe AUC without reading text.\n"
        )

    if b2:
        drop = ref_auc - b2.auc_mean
        lines.append(
            f"- **B2 residualized activations: {b2.auc_mean:.3f}** (drop {drop:.3f} vs B0). "
        )
        if b2.auc_mean < AUC_COLLAPSE:
            lines.append("Probe collapses after partialling structure — **falsified as length/position**.\n")
        elif drop > STRUCTURAL_DROP:
            lines.append("Large drop — structural confounds are a major component.\n")
        else:
            lines.append("Small drop — activations carry signal beyond structure.\n")

    if a1_b0 and a1_t2:
        lines.append(
            f"- **Length-matched subset:** B0={a1_b0.auc_mean:.3f}, T2={a1_t2.auc_mean:.3f} "
            f"({a1_b0.notes}).\n"
        )

    if e1_full:
        lines.append(f"- **Sentence-transformer (E1 full): {e1_full.auc_mean:.3f}**\n")
    if z1:
        lines.append(f"- **Zero-shot Qwen judge (Z1): {z1.auc_mean:.3f}**\n")

    ud = _get(results, "A2_user_doubt_B0")
    ud_t1 = _get(results, "A2_user_doubt_T1")
    if ud or ud_t1:
        lines.append("- **user_doubt subset** (hardest category — label depends on user reaction):\n")
        if ud:
            lines.append(f"  - Qwen B0: {ud.auc_mean:.3f} ({ud.notes})\n")
        if ud_t1:
            lines.append(f"  - TF-IDF T1: {ud_t1.auc_mean:.3f}\n")

    if b0.permutation_p is not None:
        lines.append(
            f"- **Permutation p-value (B0): {b0.permutation_p:.4f}** "
            f"(empirical, 200 shuffles).\n"
        )

    return "".join(lines)


def render_baselines_markdown(payload: dict[str, Any]) -> str:
    results = [EvalResult(**row) for row in payload["results"]]
    ref = _get(results, "B0")
    ref_auc = ref.auc_mean if ref else None
    parts = [
        "# Confusion Probe — Baseline Comparison\n\n",
        f"Generated: {payload.get('timestamp', 'unknown')}\n\n",
        f"Records: {payload.get('n_records', '?')} | ",
        f"Activations: {payload.get('has_activations', False)}\n\n",
        "## Results\n\n",
        "```\n",
        format_results_table(results, reference_auc=ref_auc),
        "\n```\n\n",
        interpret_results(results),
        "## Decision criteria\n\n",
        "| Criterion | Outcome |\n",
        "|-----------|--------|\n",
        "| T2 within 0.02 of B0 | Probe ≈ bag-of-words |\n",
        "| B2 collapses (<0.60) | Probe rode on structural confounds |\n",
        "| A1 length-matched collapses | Same |\n",
        "| B0 beats T2+E1 by ≥0.04 on A1 and user_doubt | Non-text signal survives |\n",
    ]
    return "".join(parts)
