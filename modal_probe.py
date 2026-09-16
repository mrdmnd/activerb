"""Run the confusion linear probe on Modal GPU.

Mounts the local dataset, extracts activations from Qwen3-4B on an H100,
trains a logistic regression probe per layer, saves results to the outputs volume.

Usage:
    modal run --detach modal_probe.py
    modal run --detach modal_probe.py --no-cache   # re-extract activations
"""

import modal

hf_cache_vol = modal.Volume.from_name("activerb-hf-cache", create_if_missing=False)
outputs_vol = modal.Volume.from_name("activerb-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.50.0",
        "accelerate>=1.0.0",
        "loguru>=0.7.3",
        "tqdm>=4.67.3",
        "numpy",
        "scikit-learn",
    )
    .add_local_python_source("activerb")
    .add_local_file("datasets/confusion_dataset.jsonl", remote_path="/data/confusion_dataset.jsonl")
)

app = modal.App("activerb-probe", image=image)

MODEL_ID = "Qwen/Qwen3-4B"
HF_CACHE_DIR = "/hf-cache"
OUTPUT_DIR = "/outputs"
DATASET_PATH = "/data/confusion_dataset.jsonl"
CACHE_REMOTE_PATH = "/outputs/probe/activations_cache.pkl"

LAYER_PERCENTS = [25, 50, 75]
LAYERS = [int(36 * p / 100) for p in LAYER_PERCENTS]  # 9, 18, 27
MAX_TOKENS = 4096
MAX_BLOCK_LEN = 400


@app.function(
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        OUTPUT_DIR: outputs_vol,
    },
)
def run_probe(no_cache: bool = False) -> None:
    import json
    import os
    import pickle
    from pathlib import Path

    import numpy as np
    import torch
    from loguru import logger
    from tqdm import tqdm

    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_DIR}/transformers"

    device = torch.device("cuda")

    # ── Serialization ──────────────────────────────────────────────────────────

    def serialize_block(block: dict) -> str:
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
        messages = example["messages"]
        target_id = example["target_agent_message_id"]
        prefix_msgs, target_msg = [], None
        for msg in messages:
            if msg["id"] == target_id:
                target_msg = msg
                break
            prefix_msgs.append(msg)
        if target_msg is None:
            for msg in reversed(messages):
                if msg["role"] == "agent":
                    target_msg = msg
                    break
            prefix_msgs = [m for m in messages if m is not target_msg]
        prefix_text = "".join(serialize_message(m) for m in prefix_msgs)
        target_text = serialize_message(target_msg) if target_msg else ""
        return prefix_text, target_text

    # ── Activation extraction ──────────────────────────────────────────────────

    def extract_one(model: object, tokenizer: object, prefix_text: str, target_text: str) -> dict[int, np.ndarray] | None:
        if not target_text.strip():
            return None
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        if not target_ids:
            return None
        full_ids = prefix_ids + target_ids
        if len(full_ids) > MAX_TOKENS:
            drop = len(full_ids) - MAX_TOKENS
            prefix_ids = prefix_ids[drop:] if drop < len(prefix_ids) else []
            target_ids = target_ids[-(MAX_TOKENS):] if drop >= len(prefix_ids) else target_ids
            full_ids = prefix_ids + target_ids
        target_start = len(prefix_ids)
        target_end = len(full_ids)
        if target_start >= target_end:
            return None

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int):  # noqa: ANN202
            def _hook(_m: object, _a: object, output: object) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = hidden.detach().float().cpu()
            return _hook

        handles = [model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx)) for layer_idx in LAYERS]
        try:
            with torch.inference_mode():
                model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()

        result = {}
        for layer_idx in LAYERS:
            if layer_idx not in captured:
                continue
            acts = captured[layer_idx][0, target_start:target_end, :]
            if acts.shape[0] == 0:
                continue
            result[layer_idx] = acts.mean(dim=0).numpy()
        return result if result else None

    # ── Load dataset ───────────────────────────────────────────────────────────

    with Path(DATASET_PATH).open(encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(examples)} examples from dataset")

    # ── Load or build activations ──────────────────────────────────────────────

    cache_path = Path(CACHE_REMOTE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not no_cache and cache_path.exists():
        logger.info("Loading cached activations from volume…")
        with cache_path.open("rb") as f:
            activations, labels = pickle.load(f)
    else:
        logger.info(f"Loading {MODEL_ID}…")
        from activerb.model import load_model
        model, tokenizer = load_model(MODEL_ID, device=device, dtype=torch.bfloat16)
        model.eval()

        layer_acts: dict[int, list] = {layer: [] for layer in LAYERS}
        kept_labels = []
        skipped = 0

        for ex in tqdm(examples, desc="Extracting activations"):
            prefix_text, target_text = build_context(ex)
            result = extract_one(model, tokenizer, prefix_text, target_text)
            if result is None or len(result) < len(LAYERS):
                skipped += 1
                continue
            for layer_idx in LAYERS:
                layer_acts[layer_idx].append(result[layer_idx])
            kept_labels.append(1 if ex["label"] == "confused" else 0)

        logger.info(f"Extracted {len(kept_labels)} examples ({skipped} skipped)")
        activations = {layer: np.stack(layer_acts[layer]) for layer in LAYERS}
        labels = np.array(kept_labels, dtype=int)

        with cache_path.open("wb") as f:
            pickle.dump((activations, labels), f)
        outputs_vol.commit()
        logger.info("Activations cached to volume")

        del model
        torch.cuda.empty_cache()

    # ── Train probe ────────────────────────────────────────────────────────────

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    results_summary = {}

    logger.info(f"\nFull dataset probe ({labels.sum()} confused / {(1 - labels).sum()} not_confused):")
    for layer_idx, pct in zip(LAYERS, LAYER_PERCENTS):
        X, y = activations[layer_idx], labels
        n, d = X.shape
        n_components = min(64, n - 1, d)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components)),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=["roc_auc", "accuracy"])
        auc = cv_results["test_roc_auc"]
        acc = cv_results["test_accuracy"]
        key = f"layer_{pct}pct"
        results_summary[key] = {"auc_mean": float(auc.mean()), "auc_std": float(auc.std()), "acc_mean": float(acc.mean())}
        logger.info(f"  Layer {pct}%: AUC={auc.mean():.3f}±{auc.std():.3f}  Acc={acc.mean():.3f}±{acc.std():.3f}")

    # ── User-doubt subset (LOO) ────────────────────────────────────────────────

    logger.info("\nUser-doubt subset (leave-one-out):")
    with Path(DATASET_PATH).open(encoding="utf-8") as f:
        all_examples = [json.loads(line) for line in f]

    ud_ids = {
        e["thread_id"] + str(e["target_agent_message_id"])
        for e in all_examples
        if e["label"] == "not_confused" or e["confusion_type"] == "user_doubt"
    }
    kept = [e["thread_id"] + str(e["target_agent_message_id"]) in ud_ids for e in examples]
    mask = np.array(kept[:labels.shape[0]], dtype=bool)

    logger.info(f"  User-doubt mask: {mask.sum()} examples ({labels[mask].sum()} confused)")
    MIN_USER_DOUBT = 10
    if mask.sum() >= MIN_USER_DOUBT:
        for layer_idx, pct in zip(LAYERS, LAYER_PERCENTS):
            X_sub, y_sub = activations[layer_idx][mask], labels[mask]
            n = len(y_sub)
            n_components = min(16, n - 1, X_sub.shape[1])
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
            ])
            scores = cross_val_score(pipeline, X_sub, y_sub, cv=LeaveOneOut(), scoring="roc_auc")
            key = f"user_doubt_layer_{pct}pct"
            results_summary[key] = {"auc_mean": float(scores.mean()), "auc_std": float(scores.std())}
            logger.info(f"  Layer {pct}% LOO AUC={scores.mean():.3f}±{scores.std():.3f}")

    # ── Save results ───────────────────────────────────────────────────────────

    import json as _json
    out = Path(OUTPUT_DIR) / "probe" / "probe_results.json"
    out.write_text(_json.dumps(results_summary, indent=2))
    outputs_vol.commit()
    logger.info(f"\nResults saved to {out}")
    logger.info(f"Summary: {results_summary}")


@app.local_entrypoint()
def main(no_cache: bool = False) -> None:
    run_probe.remote(no_cache=no_cache)
