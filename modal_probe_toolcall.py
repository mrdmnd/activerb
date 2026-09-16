"""Tool-call-point confusion probe on Modal H100.

Tests how early in a conversation we can detect agent confusion by probing
activations at tool call token positions rather than at the final text response.

Four probe conditions compared:
  final_response  - original baseline (activations at final agent text response)
  last_tool_call  - last tool-call-containing agent turn before the final response
  first_tool_call - first tool-call-containing agent turn in the thread
  all_tool_calls  - activations mean-pooled across ALL tool-call turns

Usage:
    modal run --detach modal_probe_toolcall.py
    modal run --detach modal_probe_toolcall.py --no-cache
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

app = modal.App("activerb-probe-toolcall", image=image)

MODEL_ID = "Qwen/Qwen3-4B"
HF_CACHE_DIR = "/hf-cache"
OUTPUT_DIR = "/outputs"
DATASET_PATH = "/data/confusion_dataset.jsonl"
CACHE_REMOTE_PATH = "/outputs/probe_toolcall/activations_cache.pkl"

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
            return f"[result {block.get('toolName', '')} -> {result}]"
        return ""

    def serialize_message(msg: dict) -> str:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        parts = [serialize_block(b) for b in msg.get("content", []) if isinstance(b, dict)]
        body = "\n".join(p for p in parts if p)
        return f"{role_label}: {body}\n\n"

    def serialize_tool_calls_only(msg: dict) -> str:
        """Serialize only the tool_call blocks from a message."""
        parts = []
        for b in msg.get("content", []):
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_call":
                args = str(b.get("arguments", ""))[:200]
                parts.append(f"[call {b.get('toolName', '')}({args})]")
        return "\n".join(parts)

    def has_tool_call(msg: dict) -> bool:
        return any(
            isinstance(b, dict) and b.get("type") == "tool_call"
            for b in msg.get("content", [])
        )

    # ── Context builders ───────────────────────────────────────────────────────

    def build_final_response_context(example: dict) -> tuple[str, str] | None:
        """(prefix, target) where target = full target agent message."""
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
            return None
        return (
            "".join(serialize_message(m) for m in prefix_msgs),
            serialize_message(target_msg),
        )

    def build_tool_call_contexts(example: dict) -> list[tuple[str, str]]:
        """
        Returns list of (prefix_text, target_text) pairs, one per tool-call agent turn,
        in chronological order. target_text contains ONLY the tool_call blocks from that
        turn — not thinking blocks or tool results. prefix_text is everything before.
        """
        messages = example["messages"]
        target_id = example["target_agent_message_id"]

        results = []
        accumulated: list[dict] = []

        for msg in messages:
            if msg["id"] == target_id:
                # Check for tool calls within the target message itself (before text blocks)
                if msg.get("role") in ("agent", "assistant") and has_tool_call(msg):
                    tc_text = serialize_tool_calls_only(msg)
                    if tc_text:
                        prefix = "".join(serialize_message(m) for m in accumulated)
                        results.append((prefix, tc_text))
                break

            if msg.get("role") in ("agent", "assistant") and has_tool_call(msg):
                tc_text = serialize_tool_calls_only(msg)
                if tc_text:
                    prefix = "".join(serialize_message(m) for m in accumulated)
                    results.append((prefix, tc_text))

            accumulated.append(msg)

        return results  # chronological order; [0]=first, [-1]=last

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
            if drop >= len(prefix_ids):
                target_ids = target_ids[-MAX_TOKENS:]
                prefix_ids = []
            else:
                prefix_ids = prefix_ids[drop:]
            full_ids = prefix_ids + target_ids
        target_start = len(prefix_ids)
        target_end = len(full_ids)
        if target_start >= target_end:
            return None

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        captured: dict[int, torch.Tensor] = {}

        def make_hook(layer_idx: int):  # noqa: ANN202
            def _hook(_m: object, _a: object, output: object) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = hidden.detach().float().cpu()
            return _hook

        handles = [model.model.layers[li].register_forward_hook(make_hook(li)) for li in LAYERS]
        try:
            with torch.inference_mode():
                model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()

        result = {}
        for li in LAYERS:
            if li not in captured:
                continue
            acts = captured[li][0, target_start:target_end, :]
            if acts.shape[0] == 0:
                continue
            result[li] = acts.mean(dim=0).numpy()
        return result if result else None

    # ── Load dataset ───────────────────────────────────────────────────────────

    with Path(DATASET_PATH).open(encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(examples)} examples")

    # ── Load or build activations ──────────────────────────────────────────────

    cache_path = Path(CACHE_REMOTE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not no_cache and cache_path.exists():
        logger.info("Loading cached activations from volume...")
        with cache_path.open("rb") as f:
            cache = pickle.load(f)
    else:
        logger.info(f"Loading {MODEL_ID}...")
        from activerb.model import load_model
        model, tokenizer = load_model(MODEL_ID, device=device, dtype=torch.bfloat16)
        model.eval()

        # Per condition: {layer_idx: [vectors]}
        condition_acts: dict[str, dict[int, list]] = {
            "final_response": {li: [] for li in LAYERS},
            "first_tool_call": {li: [] for li in LAYERS},
            "last_tool_call": {li: [] for li in LAYERS},
            "all_tool_calls": {li: [] for li in LAYERS},
        }
        condition_labels: dict[str, list] = {k: [] for k in condition_acts}
        skipped = 0

        for ex in tqdm(examples, desc="Extracting activations"):
            label = 1 if ex["label"] == "confused" else 0

            # ── final_response ─────────────────────────────────────────────────
            ctx = build_final_response_context(ex)
            if ctx is not None:
                result = extract_one(model, tokenizer, ctx[0], ctx[1])
                if result and len(result) == len(LAYERS):
                    for li in LAYERS:
                        condition_acts["final_response"][li].append(result[li])
                    condition_labels["final_response"].append(label)

            # ── tool call contexts ─────────────────────────────────────────────
            tc_contexts = build_tool_call_contexts(ex)

            if not tc_contexts:
                skipped += 1
                continue

            # first_tool_call
            result_first = extract_one(model, tokenizer, tc_contexts[0][0], tc_contexts[0][1])
            if result_first and len(result_first) == len(LAYERS):
                for li in LAYERS:
                    condition_acts["first_tool_call"][li].append(result_first[li])
                condition_labels["first_tool_call"].append(label)

            # last_tool_call
            result_last = extract_one(model, tokenizer, tc_contexts[-1][0], tc_contexts[-1][1])
            if result_last and len(result_last) == len(LAYERS):
                for li in LAYERS:
                    condition_acts["last_tool_call"][li].append(result_last[li])
                condition_labels["last_tool_call"].append(label)

            # all_tool_calls: extract each and mean-pool across turns
            layer_vecs: dict[int, list[np.ndarray]] = {li: [] for li in LAYERS}
            all_ok = True
            for prefix, tc_text in tc_contexts:
                r = extract_one(model, tokenizer, prefix, tc_text)
                if r is None or len(r) < len(LAYERS):
                    all_ok = False
                    break
                for li in LAYERS:
                    layer_vecs[li].append(r[li])
            if all_ok and all(layer_vecs[li] for li in LAYERS):
                for li in LAYERS:
                    condition_acts["all_tool_calls"][li].append(np.stack(layer_vecs[li]).mean(axis=0))
                condition_labels["all_tool_calls"].append(label)

        logger.info(f"Examples with no tool calls: {skipped}")

        # Stack into arrays
        cache = {}
        for condition in condition_acts:
            n = len(condition_labels[condition])
            if n == 0:
                logger.warning(f"Condition {condition}: no examples — skipping")
                continue
            cache[condition] = {
                "activations": {li: np.stack(condition_acts[condition][li]) for li in LAYERS},
                "labels": np.array(condition_labels[condition], dtype=int),
            }
            logger.info(f"Condition {condition}: {n} examples ({cache[condition]['labels'].sum()} confused)")

        with cache_path.open("wb") as f:
            pickle.dump(cache, f)
        outputs_vol.commit()
        logger.info("Activations cached to volume")

        del model
        torch.cuda.empty_cache()

    # ── Train probes ───────────────────────────────────────────────────────────

    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    results_summary = {}

    for condition, data in cache.items():
        activations = data["activations"]
        labels = data["labels"]
        n_confused = int(labels.sum())
        n_total = len(labels)
        logger.info(f"\nCondition: {condition}  ({n_confused} confused / {n_total - n_confused} not_confused, n={n_total})")

        condition_results = {}
        for li, pct in zip(LAYERS, LAYER_PERCENTS):
            X, y = activations[li], labels
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
            condition_results[key] = {
                "auc_mean": float(auc.mean()),
                "auc_std": float(auc.std()),
                "acc_mean": float(acc.mean()),
                "n": n_total,
                "n_confused": n_confused,
            }
            logger.info(f"  Layer {pct}%: AUC={auc.mean():.3f}±{auc.std():.3f}  Acc={acc.mean():.3f}±{acc.std():.3f}")

        results_summary[condition] = condition_results

    # ── Save results ───────────────────────────────────────────────────────────

    import json as _json
    out = Path(OUTPUT_DIR) / "probe_toolcall" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(results_summary, indent=2))
    outputs_vol.commit()
    logger.info(f"\nResults saved to {out}")

    # Pretty summary table
    logger.info("\n── Summary (Layer 50%) ──────────────────────────────────")
    logger.info(f"{'Condition':<20} {'AUC':>6}  {'±':>5}  {'Acc':>6}  {'n':>5}")
    logger.info("-" * 55)
    for condition in ["final_response", "last_tool_call", "first_tool_call", "all_tool_calls"]:
        if condition not in results_summary:
            continue
        r = results_summary[condition].get("layer_50pct", {})
        logger.info(
            f"{condition:<20} {r.get('auc_mean', 0):.3f}  ±{r.get('auc_std', 0):.3f}  "
            f"{r.get('acc_mean', 0):.3f}  {r.get('n', 0):>5}"
        )


@app.local_entrypoint()
def main(no_cache: bool = False) -> None:
    run_probe.remote(no_cache=no_cache)
