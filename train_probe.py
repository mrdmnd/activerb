"""Train a linear probe to detect confusion from Qwen3-4B hidden states.

Pipeline:
  1. Load confusion_dataset.jsonl
  2. Serialize each example's messages to text, tracking target message token span
  3. Run forward passes through Qwen3-4B (MPS) — one example at a time
  4. Mean-pool activations at target agent message positions for each layer
  5. Train logistic regression (5-fold CV) per layer, report AUC

Activations are cached to disk so you can iterate on the probe without
re-running inference.

Usage:
    uv run python train_probe.py
    uv run python train_probe.py --no-cache   # re-extract activations
    uv run python train_probe.py --user-doubt-only  # cleaner labels only
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

DATASET_PATH = "datasets/confusion_dataset.jsonl"
CACHE_PATH = "datasets/activations_cache.pkl"
MODEL_ID = "Qwen/Qwen3-4B"

# Layers to probe (indices into Qwen3-4B's 36 layers)
LAYER_PERCENTS = [25, 50, 75]
LAYERS = [int(36 * p / 100) for p in LAYER_PERCENTS]  # 9, 18, 27

MAX_TOKENS = 4096       # truncate context to last N tokens (MPS memory)
MAX_BLOCK_LEN = 400     # max chars per content block (tool results can be huge)

# ── Serialization ─────────────────────────────────────────────────────────────


def serialize_block(block: dict) -> str:
    """Convert one content block to plain text."""
    btype = block.get("type", "")
    if btype == "text":
        return block.get("text", "").strip()[:MAX_BLOCK_LEN]
    if btype == "thinking":
        return f"<thinking>{block.get('text', '').strip()[:MAX_BLOCK_LEN]}</thinking>"
    if btype == "tool_call":
        args = str(block.get("arguments", ""))[:200]
        return f"[call {block.get('toolName', '')}({args})]"
    if btype == "tool_result":
        result = json.dumps(block.get("result", {}))[:MAX_BLOCK_LEN]
        return f"[result {block.get('toolName', '')} → {result}]"
    return ""


def serialize_message(msg: dict) -> str:
    role_label = "User" if msg["role"] == "user" else "Assistant"
    parts = [serialize_block(b) for b in msg.get("content", []) if isinstance(b, dict)]
    body = "\n".join(p for p in parts if p)
    return f"{role_label}: {body}\n\n"


def build_context(example: dict) -> tuple[str, str]:
    """Return (prefix_text, target_text) where prefix is everything before the
    target agent message and target_text is the target message itself."""
    messages = example["messages"]
    target_id = example["target_agent_message_id"]

    prefix_msgs = []
    target_msg = None
    for msg in messages:
        if msg["id"] == target_id:
            target_msg = msg
            break
        prefix_msgs.append(msg)

    if target_msg is None:
        # Fallback: use last agent message
        for msg in reversed(messages):
            if msg["role"] == "agent":
                target_msg = msg
                break
        prefix_msgs = [m for m in messages if m is not target_msg]

    prefix_text = "".join(serialize_message(m) for m in prefix_msgs)
    target_text = serialize_message(target_msg) if target_msg else ""
    return prefix_text, target_text


# ── Activation extraction ─────────────────────────────────────────────────────


def extract_one(
    model: torch.nn.Module,
    tokenizer: object,
    prefix_text: str,
    target_text: str,
    layers: list[int],
    device: torch.device,
) -> dict[int, np.ndarray] | None:
    """
    Run one forward pass. Returns mean-pooled hidden states at target token
    positions for each requested layer. Returns None if target is empty.
    """
    if not target_text.strip():
        return None

    # Tokenize separately to find the boundary
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    if not target_ids:
        return None

    full_ids = prefix_ids + target_ids

    # Truncate to last MAX_TOKENS tokens (keeping target at the end)
    if len(full_ids) > MAX_TOKENS:
        drop = len(full_ids) - MAX_TOKENS
        if drop >= len(prefix_ids):
            # Even truncating all of prefix isn't enough — truncate target too
            target_ids = target_ids[-(MAX_TOKENS):]
            prefix_ids = []
        else:
            prefix_ids = prefix_ids[drop:]
        full_ids = prefix_ids + target_ids

    target_start = len(prefix_ids)
    target_end = len(full_ids)

    if target_start >= target_end:
        return None

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    # Collect activations via forward hooks
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_idx: int):  # noqa: ANN202
        def _hook(_module: object, _args: object, output: object) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach().float().cpu()
        return _hook

    for layer_idx in layers:
        submodule = model.model.layers[layer_idx]
        handles.append(submodule.register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.inference_mode():
            model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()

    result = {}
    for layer_idx in layers:
        if layer_idx not in captured:
            continue
        acts = captured[layer_idx][0, target_start:target_end, :]  # [T, D]
        if acts.shape[0] == 0:
            continue
        result[layer_idx] = acts.mean(dim=0).numpy()  # [D]

    return result if result else None


def build_activation_matrix(
    examples: list[dict],
    model: torch.nn.Module,
    tokenizer: object,
    layers: list[int],
    device: torch.device,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Returns:
        activations: {layer_idx: (N, D) array}
        labels: (N,) int array  0=not_confused, 1=confused
    """
    layer_acts: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    kept_labels = []
    skipped = 0

    for ex in tqdm(examples, desc="Extracting activations"):
        prefix_text, target_text = build_context(ex)
        result = extract_one(model, tokenizer, prefix_text, target_text, layers, device)
        if result is None or len(result) < len(layers):
            skipped += 1
            continue
        for layer_idx in layers:
            layer_acts[layer_idx].append(result[layer_idx])
        kept_labels.append(1 if ex["label"] == "confused" else 0)

    if skipped:
        logger.warning(f"Skipped {skipped} examples (empty target message)")

    return (
        {layer: np.stack(layer_acts[layer]) for layer in layers},
        np.array(kept_labels, dtype=int),
    )


# ── Probe training ────────────────────────────────────────────────────────────


def train_probe(X: np.ndarray, y: np.ndarray, layer_name: str) -> None:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    n, d = X.shape
    logger.info(f"Layer {layer_name}: {n} examples, {d} dims, {y.sum()} confused / {(1 - y).sum()} not_confused")

    # PCA to 64 dims before logistic regression (avoids overfitting with small N)
    n_components = min(64, n - 1, d)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = cross_validate(pipeline, X, y, cv=cv, scoring=["roc_auc", "accuracy"], return_train_score=False)

    auc = results["test_roc_auc"]
    acc = results["test_accuracy"]
    logger.info(
        f"  AUC  = {auc.mean():.3f} ± {auc.std():.3f}  "
        f"  Acc  = {acc.mean():.3f} ± {acc.std():.3f}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true", help="Re-extract activations even if cache exists")
    parser.add_argument("--user-doubt-only", action="store_true", help="Only use user_doubt confused examples")
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    with Path(DATASET_PATH).open(encoding="utf-8") as f:
        all_examples = [json.loads(line) for line in f]

    if args.user_doubt_only:
        examples = [
            e for e in all_examples
            if e["label"] == "not_confused" or e["confusion_type"] == "user_doubt"
        ]
        logger.info(f"user_doubt_only: {len(examples)} examples")
    else:
        examples = all_examples
        logger.info(f"All examples: {len(examples)}")

    # ── Load or build activations ─────────────────────────────────────────────
    cache_path = Path(CACHE_PATH if not args.user_doubt_only else CACHE_PATH.replace(".pkl", "_user_doubt.pkl"))

    if not args.no_cache and cache_path.exists():
        logger.info(f"Loading cached activations from {cache_path}")
        with Path(cache_path).open("rb") as f:
            activations, labels = pickle.load(f)
    else:
        logger.info(f"Loading {MODEL_ID} on MPS…")
        from activerb.model import load_model
        device = torch.device("mps")
        model, tokenizer = load_model(MODEL_ID, device=device, dtype=torch.float16)
        model.eval()

        activations, labels = build_activation_matrix(examples, model, tokenizer, LAYERS, device)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with Path(cache_path).open("wb") as f:
            pickle.dump((activations, labels), f)
        logger.info(f"Activations cached to {cache_path}")

        del model
        torch.mps.empty_cache()

    # ── Train probe per layer ─────────────────────────────────────────────────
    logger.info(f"\nProbe results ({labels.sum()} confused / {(1 - labels).sum()} not_confused):")
    for layer_idx, pct in zip(LAYERS, LAYER_PERCENTS):
        train_probe(activations[layer_idx], labels, f"{pct}% (layer {layer_idx})")

    # ── User-doubt subset if running full ─────────────────────────────────────
    if not args.user_doubt_only:
        # Find which examples in our kept set are user_doubt
        # (need to rebuild the kept-example mapping)
        logger.info("\nRe-running probe on user_doubt confused examples only…")
        with Path(DATASET_PATH).open(encoding="utf-8") as f:
            full_examples = [json.loads(line) for line in f]

        ud_examples = [e for e in full_examples if e["label"] == "not_confused" or e["confusion_type"] == "user_doubt"]
        ud_ids = {e["thread_id"] + str(e["target_agent_message_id"]) for e in ud_examples}

        # Rebuild full example list in same order we extracted (skip same as above)
        kept = []
        for ex in examples:
            key = ex["thread_id"] + str(ex["target_agent_message_id"])
            kept.append(key in ud_ids)
        mask = np.array(kept[:labels.shape[0]], dtype=bool)

        MIN_USER_DOUBT = 10
        if mask.sum() >= MIN_USER_DOUBT:
            logger.info(f"User-doubt subset: {mask.sum()} examples")
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import LeaveOneOut, cross_val_score
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            for layer_idx, pct in zip(LAYERS, LAYER_PERCENTS):
                X_sub = activations[layer_idx][mask]
                y_sub = labels[mask]
                n = len(y_sub)
                n_components = min(16, n - 1, X_sub.shape[1])
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=n_components)),
                    ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
                ])
                scores = cross_val_score(pipeline, X_sub, y_sub, cv=LeaveOneOut(), scoring="roc_auc")
                logger.info(f"  Layer {pct}% LOO AUC = {scores.mean():.3f} ± {scores.std():.3f}")
        else:
            logger.warning(f"Only {mask.sum()} user_doubt examples after filtering — skipping subset probe")


if __name__ == "__main__":
    main()
