"""Run confusion-probe disprove-me baselines (CPU).

Reads features from a pickle (Modal extraction or local build) and runs
text/structural/Qwen baselines with GroupKFold by thread_id.

Usage:
    # Text-only baselines from dataset (no GPU activations):
    uv run python baselines_run.py --dataset datasets/confusion_dataset.jsonl

    # Full baselines after Modal extraction:
    modal volume get activerb-outputs /probe/features_v2.pkl ./runs/features_v2.pkl
    uv run python baselines_run.py --features runs/features_v2.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from activerb.baselines_eval import (
    MIN_SUBSET_FOR_CV,
    MIN_SUBSET_FOR_LOO,
    EvalResult,
    evaluate_array,
    evaluate_array_structural_residualized,
    evaluate_tfidf,
    format_results_table,
    length_matched_indices,
    make_logreg_pipeline,
    permutation_p_value,
)
from activerb.baselines_report import render_baselines_markdown
from activerb.probe_features import (
    LAYERS,
    FeatureRecord,
    build_metadata_records,
    load_dataset_jsonl,
    structural_vector,
)

LAYER_PCT = {9: 25, 18: 50, 27: 75}


def records_from_pickle(path: Path) -> list[FeatureRecord]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    return [
        FeatureRecord(
            thread_id=row["thread_id"],
            label=int(row["label"]),
            confusion_type=row.get("confusion_type"),
            meta=row["meta"],
            activations={int(k): np.asarray(v) for k, v in row["activations"].items()},
        )
        for row in payload["records"]
    ]


def arrays_from_records(
    records: list[FeatureRecord],
    indices: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[int, np.ndarray],
    list[FeatureRecord],
]:
    if indices is None:
        indices = np.arange(len(records))
    subset = [records[i] for i in indices]
    y = np.array([r.label for r in subset], dtype=int)
    groups = np.array([r.thread_id for r in subset])
    structural = np.array([structural_vector(r.meta) for r in subset], dtype=np.float64)
    hedge = np.array([r.hedge_vector for r in subset], dtype=np.float64)
    activations: dict[int, np.ndarray] = {}
    if subset and subset[0].activations:
        for layer in LAYERS:
            activations[layer] = np.stack([r.activations[layer] for r in subset])
    return y, groups, structural, hedge, activations, subset


def run_all_baselines(  # noqa: PLR0912
    records: list[FeatureRecord],
    *,
    run_permutations: bool = True,
    reference_layer: int = 18,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    y, groups, structural, hedge, activations, _ = arrays_from_records(records)
    target_texts = [r.meta["target_text"] for r in records]
    full_texts = [r.meta["full_text"] for r in records]

    # Reference: Qwen with StratifiedKFold (replicates original probe protocol)
    X_ref = activations.get(reference_layer)
    if X_ref is not None and X_ref.shape[0] == len(records):
        results.append(
            evaluate_array(
                "B0_strat",
                "Qwen probe (StratifiedKFold, layer 50%)",
                X_ref,
                y,
                groups,
                layer_pct=50,
                cv_type="stratified",
            )
        )
        b0 = evaluate_array(
            "B0",
            "Qwen probe (GroupKFold, layer 50%)",
            X_ref,
            y,
            groups,
            layer_pct=50,
        )
        results.append(b0)
        reference_auc = b0.auc_mean
    else:
        logger.warning("No activations — skipping Qwen probe baselines (B0)")
        reference_auc = None

    results.append(
        evaluate_array(
            "S1",
            "Structural features only",
            structural,
            y,
            groups,
            use_pca=False,
        )
    )
    results.append(
        evaluate_tfidf("T1", "TF-IDF on target text", target_texts, y, groups)
    )
    results.append(
        evaluate_tfidf("T2", "TF-IDF on full transcript", full_texts, y, groups)
    )
    results.append(
        evaluate_array(
            "H1",
            "Hedge lexicon counts",
            hedge,
            y,
            groups,
            use_pca=False,
        )
    )

    if X_ref is not None:
        results.append(
            evaluate_array(
                "B1",
                "Qwen + structural (concat)",
                np.hstack([X_ref, structural]),
                y,
                groups,
                layer_pct=50,
            )
        )
        results.append(
            evaluate_array_structural_residualized(
                "B2",
                "Qwen probe after partialing structural features",
                X_ref,
                structural,
                y,
                groups,
                layer_pct=50,
                notes="Collapse vs B0 => probe rode on length/position confounds",
            )
        )

        for layer, pct in zip(LAYERS, (25, 50, 75)):
            if layer == reference_layer:
                continue
            results.append(
                evaluate_array(
                    f"B0_L{pct}",
                    f"Qwen probe GroupKFold layer {pct}%",
                    activations[layer],
                    y,
                    groups,
                    layer_pct=pct,
                )
            )

    # A1: length-matched subset
    matched_idx = length_matched_indices(records)
    logger.info(f"Length-matched subset: {len(matched_idx)} examples")
    _, _, struct_m, _, acts_m, subset_m = arrays_from_records(records, matched_idx)
    texts_m = [r.meta["full_text"] for r in subset_m]
    y_m = np.array([r.label for r in subset_m], dtype=int)
    groups_m = np.array([r.thread_id for r in subset_m])
    if X_ref is not None:
        results.append(
            evaluate_array(
                "A1_B0",
                "Qwen probe, length-matched subset",
                acts_m[reference_layer],
                y_m,
                groups_m,
                layer_pct=50,
                notes=f"n={len(matched_idx)}",
            )
        )
    results.append(
        evaluate_array(
            "A1_S1",
            "Structural only, length-matched",
            struct_m,
            y_m,
            groups_m,
            use_pca=False,
            notes=f"n={len(matched_idx)}",
        )
    )
    results.append(
        evaluate_tfidf(
            "A1_T2",
            "TF-IDF full transcript, length-matched",
            texts_m,
            y_m,
            groups_m,
            notes=f"n={len(matched_idx)}",
        )
    )

    # A2: per confusion type (positives subset vs all negatives)
    neg_records = [r for r in records if r.label == 0]
    for ctype in ("agent_caveat", "missing_context", "user_doubt"):
        pos_records = [r for r in records if r.label == 1 and r.confusion_type == ctype]
        if not pos_records:
            continue
        subset_records = pos_records + neg_records
        y_sub = np.array([r.label for r in subset_records], dtype=int)
        groups_sub = np.array([r.thread_id for r in subset_records])
        has_acts = bool(pos_records[0].activations)
        if X_ref is not None and has_acts:
            X_sub = np.stack([r.activations[reference_layer] for r in subset_records])
            use_loo = len(subset_records) < MIN_SUBSET_FOR_LOO
            if use_loo:
                from sklearn.model_selection import LeaveOneOut, cross_validate

                pipe = make_logreg_pipeline(X_sub.shape[1], len(y_sub))
                cv = LeaveOneOut()
                scores = cross_validate(
                    pipe, X_sub, y_sub, cv=cv, scoring=["roc_auc"]
                )["test_roc_auc"]
                results.append(
                    EvalResult(
                        method_id=f"A2_{ctype}_B0",
                        description=f"Qwen probe, {ctype} vs negatives",
                        auc_mean=float(np.nanmean(scores)),
                        auc_std=float(np.nanstd(scores)),
                        pr_auc_mean=float("nan"),
                        pr_auc_std=float("nan"),
                        acc_mean=float("nan"),
                        acc_std=float("nan"),
                        n_examples=len(y_sub),
                        cv="loo",
                        layer_pct=50,
                        notes=f"{len(pos_records)} positives",
                    )
                )
            else:
                results.append(
                    evaluate_array(
                        f"A2_{ctype}_B0",
                        f"Qwen probe, {ctype} vs negatives",
                        X_sub,
                        y_sub,
                        groups_sub,
                        layer_pct=50,
                        notes=f"{len(pos_records)} positives",
                    )
                )
        texts_sub = [r.meta["target_text"] for r in subset_records]
        if len(subset_records) >= MIN_SUBSET_FOR_CV:
            results.append(
                evaluate_tfidf(
                    f"A2_{ctype}_T1",
                    f"TF-IDF target, {ctype} vs negatives",
                    texts_sub,
                    y_sub,
                    groups_sub,
                    notes=f"{len(pos_records)} positives",
                )
            )

    # P1: permutation null for B0 and T2
    if run_permutations and X_ref is not None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        t2 = results[[r.method_id for r in results].index("T2")]
        b0_group = next(r for r in results if r.method_id == "B0")
        X_t2 = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=20_000, sublinear_tf=True
        ).fit_transform(full_texts).toarray()
        b0_perm_p = permutation_p_value(X_ref, y, groups, b0_group.auc_mean)
        t2_perm_p = permutation_p_value(X_t2, y, groups, t2.auc_mean, use_pca=False)
        updated: list[EvalResult] = []
        for r in results:
            if r.method_id == "B0":
                updated.append(
                    EvalResult(**{**r.to_dict(), "permutation_p": b0_perm_p})
                )
            elif r.method_id == "T2":
                updated.append(
                    EvalResult(**{**r.to_dict(), "permutation_p": t2_perm_p})
                )
            else:
                updated.append(r)
        results = updated

    if reference_auc is not None:
        logger.info("\n" + format_results_table(results, reference_auc=reference_auc))
    else:
        logger.info("\n" + format_results_table(results))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run confusion probe baselines")
    parser.add_argument(
        "--features",
        type=Path,
        help="Pickle from modal_extract_features.py (includes activations)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/confusion_dataset.jsonl"),
        help="JSONL dataset (text/structural only if no --features)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/baselines_<timestamp>)",
    )
    parser.add_argument(
        "--no-permutations",
        action="store_true",
        help="Skip label-permutation null (faster)",
    )
    parser.add_argument(
        "--merge-gpu",
        type=Path,
        help="Merge results from modal_baselines_gpu.py into --out-dir after run",
    )
    args = parser.parse_args()

    if args.features and args.features.exists():
        records = records_from_pickle(args.features)
        logger.info(f"Loaded {len(records)} records from {args.features}")
    else:
        examples = load_dataset_jsonl(str(args.dataset))
        records = build_metadata_records(examples)
        logger.info(
            f"Built {len(records)} metadata-only records from {args.dataset} "
            "(no Qwen activations — B0/B1/B2 skipped)"
        )

    out_dir = args.out_dir or Path("runs") / f"baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_all_baselines(records, run_permutations=not args.no_permutations)
    out_path = out_dir / "results_comparison.json"
    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "n_records": len(records),
        "has_activations": bool(records and records[0].activations),
        "results": [r.to_dict() for r in results],
    }
    if args.merge_gpu and args.merge_gpu.exists():
        gpu_payload = json.loads(args.merge_gpu.read_text(encoding="utf-8"))
        for row in gpu_payload.get("results", []):
            fields = EvalResult.__dataclass_fields__
            results.append(EvalResult(**{k: row[k] for k in fields if k in row}))
        payload["results"] = [r.to_dict() for r in results]
        logger.info(f"Merged GPU baselines from {args.merge_gpu}")

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Wrote {out_path}")

    md_text = render_baselines_markdown(payload)
    md_path = out_dir / "BASELINES.md"
    md_path.write_text(md_text, encoding="utf-8")
    root_md = Path("BASELINES.md")
    root_md.write_text(md_text, encoding="utf-8")
    logger.info(f"Wrote {md_path} and {root_md}")


if __name__ == "__main__":
    main()
