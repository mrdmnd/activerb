"""Shared CV evaluation for confusion-probe baselines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CV_SEED = 42
N_SPLITS = 5
N_PERMUTATIONS = 200
MIN_PCA_COMPONENTS = 2
MIN_MATCHED_SUBSET = 50
REFERENCE_LAYER_PCT = 50
MIN_SUBSET_FOR_CV = 20
MIN_SUBSET_FOR_LOO = 30


@dataclass(frozen=True)
class EvalResult:
    method_id: str
    description: str
    auc_mean: float
    auc_std: float
    pr_auc_mean: float
    pr_auc_std: float
    acc_mean: float
    acc_std: float
    n_examples: int
    cv: str
    layer_pct: int | None = None
    permutation_p: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "description": self.description,
            "auc_mean": self.auc_mean,
            "auc_std": self.auc_std,
            "pr_auc_mean": self.pr_auc_mean,
            "pr_auc_std": self.pr_auc_std,
            "acc_mean": self.acc_mean,
            "acc_std": self.acc_std,
            "n_examples": self.n_examples,
            "cv": self.cv,
            "layer_pct": self.layer_pct,
            "permutation_p": self.permutation_p,
            "notes": self.notes,
        }


def _youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    thresholds = np.unique(y_score)
    if len(thresholds) == 0:
        return 0.5
    best_t, best_j = 0.5, -1.0
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        tn = np.sum((pred == 0) & (y_true == 0))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


def make_logreg_pipeline(
    n_features: int,
    n_samples: int,
    *,
    use_pca: bool = True,
) -> Pipeline:
    n_components = min(64, n_samples - 1, n_features)
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if use_pca and n_components >= MIN_PCA_COMPONENTS:
        steps.append(("pca", PCA(n_components=n_components)))
    steps.append(
        (
            "clf",
            LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"),
        )
    )
    return Pipeline(steps)


def cross_validate_classifier(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    *,
    use_pca: bool = True,
    cv_type: Literal["group", "stratified"] = "group",
) -> tuple[list[float], list[float], list[float]]:
    n, d = X.shape
    pipeline = make_logreg_pipeline(d, n, use_pca=use_pca)
    if cv_type == "group" and groups is not None:
        cv = GroupKFold(n_splits=N_SPLITS)
        splits = cv.split(X, y, groups=groups)
    else:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
        splits = cv.split(X, y)

    aucs: list[float] = []
    pr_aucs: list[float] = []
    accs: list[float] = []

    for train_idx, test_idx in splits:
        model = clone(pipeline)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        y_test = y[test_idx]
        aucs.append(float(roc_auc_score(y_test, proba)))
        pr_aucs.append(float(average_precision_score(y_test, proba)))
        thr = _youden_threshold(y_test, proba)
        accs.append(float(accuracy_score(y_test, (proba >= thr).astype(int))))

    return aucs, pr_aucs, accs


def evaluate_array_structural_residualized(
    method_id: str,
    description: str,
    X: np.ndarray,
    structural: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    layer_pct: int | None = None,
    notes: str = "",
) -> EvalResult:
    """B2: within each fold, partial structural features out of activations, then probe."""
    n, d = X.shape
    pipeline = make_logreg_pipeline(d, n, use_pca=True)
    cv = GroupKFold(n_splits=N_SPLITS)
    aucs: list[float] = []
    pr_aucs: list[float] = []
    accs: list[float] = []

    for train_idx, test_idx in cv.split(X, y, groups=groups):
        reg = Ridge(alpha=10.0)
        reg.fit(structural[train_idx], X[train_idx])
        x_train_res = X[train_idx] - reg.predict(structural[train_idx])
        x_test_res = X[test_idx] - reg.predict(structural[test_idx])
        model = clone(pipeline)
        model.fit(x_train_res, y[train_idx])
        proba = model.predict_proba(x_test_res)[:, 1]
        y_test = y[test_idx]
        aucs.append(float(roc_auc_score(y_test, proba)))
        pr_aucs.append(float(average_precision_score(y_test, proba)))
        thr = _youden_threshold(y_test, proba)
        accs.append(float(accuracy_score(y_test, (proba >= thr).astype(int))))

    return EvalResult(
        method_id=method_id,
        description=description,
        auc_mean=float(np.mean(aucs)),
        auc_std=float(np.std(aucs)),
        pr_auc_mean=float(np.mean(pr_aucs)),
        pr_auc_std=float(np.std(pr_aucs)),
        acc_mean=float(np.mean(accs)),
        acc_std=float(np.std(accs)),
        n_examples=len(y),
        cv="group",
        layer_pct=layer_pct,
        notes=notes,
    )


def evaluate_array(
    method_id: str,
    description: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    use_pca: bool = True,
    cv_type: Literal["group", "stratified"] = "group",
    layer_pct: int | None = None,
    notes: str = "",
) -> EvalResult:
    aucs, pr_aucs, accs = cross_validate_classifier(
        X, y, groups, use_pca=use_pca, cv_type=cv_type
    )
    return EvalResult(
        method_id=method_id,
        description=description,
        auc_mean=float(np.mean(aucs)),
        auc_std=float(np.std(aucs)),
        pr_auc_mean=float(np.mean(pr_aucs)),
        pr_auc_std=float(np.std(pr_aucs)),
        acc_mean=float(np.mean(accs)),
        acc_std=float(np.std(accs)),
        n_examples=len(y),
        cv=cv_type,
        layer_pct=layer_pct,
        notes=notes,
    )


def evaluate_tfidf(
    method_id: str,
    description: str,
    texts: Sequence[str],
    y: np.ndarray,
    groups: np.ndarray,
    *,
    cv_type: Literal["group", "stratified"] = "group",
    notes: str = "",
) -> EvalResult:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=20_000,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts).toarray()
    return evaluate_array(
        method_id,
        description,
        X,
        y,
        groups,
        use_pca=False,
        cv_type=cv_type,
        notes=notes,
    )


def permutation_p_value(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    observed_auc: float,
    *,
    n_perm: int = N_PERMUTATIONS,
    use_pca: bool = True,
    seed: int = CV_SEED,
) -> float:
    rng = np.random.default_rng(seed)
    null_aucs: list[float] = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        aucs, _, _ = cross_validate_classifier(X, y_perm, groups, use_pca=use_pca)
        null_aucs.append(float(np.mean(aucs)))
    return float(np.mean(np.array(null_aucs) >= observed_auc))


def oof_predict_proba(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    use_pca: bool = True,
) -> np.ndarray:
    """Out-of-fold positive-class probabilities for residualization."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    pipeline = make_logreg_pipeline(X.shape[1], n, use_pca=use_pca)
    cv = GroupKFold(n_splits=N_SPLITS)
    for train_idx, test_idx in cv.split(X, y, groups=groups):
        model = clone(pipeline)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return oof


def length_matched_indices(
    records: Sequence[Any],
    *,
    seed: int = CV_SEED,
) -> np.ndarray:
    """Bin-stratified subsample to balance target_text_chars and n_messages."""
    rng = np.random.default_rng(seed)
    pos_idx = [i for i, r in enumerate(records) if r.label == 1]
    neg_idx = [i for i, r in enumerate(records) if r.label == 0]
    n = min(len(pos_idx), len(neg_idx))

    def bins_for(indices: list[int], key: str, n_bins: int = 5) -> dict[int, list[int]]:
        vals = [records[i].meta[key] for i in indices]
        edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        out: dict[int, list[int]] = {b: [] for b in range(len(edges) - 1)}
        for i in indices:
            v = records[i].meta[key]
            b = int(np.searchsorted(edges[1:], v, side="right"))
            b = min(b, len(edges) - 2)
            out[b].append(i)
        return out

    pos_bins_chars = bins_for(pos_idx, "target_text_chars")
    neg_bins_chars = bins_for(neg_idx, "target_text_chars")
    selected: list[int] = []
    for b in pos_bins_chars:
        pos_pool = pos_bins_chars[b]
        neg_pool = neg_bins_chars.get(b, neg_idx)
        k = min(len(pos_pool), len(neg_pool), max(1, n // 5))
        if not pos_pool or not neg_pool:
            continue
        selected.extend(rng.choice(pos_pool, size=k, replace=False).tolist())
        selected.extend(rng.choice(neg_pool, size=k, replace=False).tolist())
    if len(selected) < MIN_MATCHED_SUBSET:
        # fallback: match on median chars
        med = float(np.median([records[i].meta["target_text_chars"] for i in range(len(records))]))
        pos_near = sorted(pos_idx, key=lambda i: abs(records[i].meta["target_text_chars"] - med))[:n]
        neg_near = sorted(neg_idx, key=lambda i: abs(records[i].meta["target_text_chars"] - med))[:n]
        selected = pos_near + neg_near
    return np.array(sorted(set(selected)), dtype=int)


def format_results_table(results: list[EvalResult], reference_auc: float | None = None) -> str:
    lines = [
        f"{'Method':<42} {'AUC':>7} {'±':>5} {'PR-AUC':>7} {'Δ ref':>7} {'p_perm':>7}",
        "-" * 82,
    ]
    for r in results:
        delta = ""
        if reference_auc is not None and r.layer_pct == REFERENCE_LAYER_PCT:
            delta = f"{r.auc_mean - reference_auc:+.3f}"
        p = f"{r.permutation_p:.3f}" if r.permutation_p is not None else "—"
        layer_note = f" L{r.layer_pct}%" if r.layer_pct is not None else ""
        name = f"{r.method_id}{layer_note}: {r.description}"[:42]
        lines.append(
            f"{name:<42} {r.auc_mean:>7.3f} {r.auc_std:>5.3f} {r.pr_auc_mean:>7.3f} {delta:>7} {p:>7}"
        )
    return "\n".join(lines)
