"""Extract unified features for confusion-probe baseline experiments.

Saves thread metadata, raw text, structural features, hedge counts, and
Qwen3-4B activations to the Modal outputs volume.

Usage:
    modal run --detach modal_extract_features.py
    modal run --detach modal_extract_features.py --no-cache
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

app = modal.App("activerb-extract-features", image=image)

MODEL_ID = "Qwen/Qwen3-4B"
HF_CACHE_DIR = "/hf-cache"
OUTPUT_DIR = "/outputs"
DATASET_PATH = "/data/confusion_dataset.jsonl"
FEATURES_PATH = "/outputs/probe/features_v2.pkl"
MAX_TOKENS = 4096


@app.function(
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        OUTPUT_DIR: outputs_vol,
    },
)
def extract_features(no_cache: bool = False) -> None:
    import os
    import pickle
    from pathlib import Path

    import numpy as np
    import torch
    from loguru import logger
    from tqdm import tqdm

    from activerb.probe_features import (
        LAYERS,
        build_context,
        metadata_from_example,
    )

    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_DIR}/transformers"

    device = torch.device("cuda")
    out_path = Path(FEATURES_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not no_cache and out_path.exists():
        logger.info(f"Features already exist at {out_path}; use --no-cache to re-extract")
        return

    import json

    with Path(DATASET_PATH).open(encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(examples)} examples")

    from activerb.model import load_model

    model, tokenizer = load_model(MODEL_ID, device=device, dtype=torch.bfloat16)
    model.eval()

    def extract_activations(prefix_text: str, target_text: str) -> dict[int, np.ndarray] | None:
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
                prefix_ids = []
                target_ids = target_ids[-MAX_TOKENS:]
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

        handles = [
            model.model.layers[li].register_forward_hook(make_hook(li)) for li in LAYERS
        ]
        try:
            with torch.inference_mode():
                model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()

        result: dict[int, np.ndarray] = {}
        for li in LAYERS:
            if li not in captured:
                continue
            acts = captured[li][0, target_start:target_end, :]
            if acts.shape[0] == 0:
                continue
            result[li] = acts.mean(dim=0).numpy()
        return result if len(result) == len(LAYERS) else None

    records: list[dict] = []
    skipped = 0

    for ex in tqdm(examples, desc="Extracting"):
        prefix_text, target_text, _ = build_context(ex)
        prefix_tok = len(tokenizer.encode(prefix_text, add_special_tokens=False))
        target_tok = len(tokenizer.encode(target_text, add_special_tokens=False))
        meta = metadata_from_example(ex, target_token_count=target_tok)
        meta["prefix_tokens"] = prefix_tok

        acts = extract_activations(prefix_text, target_text)
        if acts is None:
            skipped += 1
            continue

        records.append(
            {
                "thread_id": ex["thread_id"],
                "label": meta["label"],
                "confusion_type": ex.get("confusion_type"),
                "meta": meta,
                "activations": {int(k): v for k, v in acts.items()},
            }
        )

    payload = {
        "model_id": MODEL_ID,
        "layers": list(LAYERS),
        "layer_percents": [25, 50, 75],
        "records": records,
        "skipped": skipped,
    }
    with out_path.open("wb") as f:
        pickle.dump(payload, f)
    outputs_vol.commit()
    logger.info(f"Saved {len(records)} records ({skipped} skipped) to {out_path}")


@app.local_entrypoint()
def main(no_cache: bool = False) -> None:
    extract_features.remote(no_cache=no_cache)
