"""GPU baselines: sentence-transformer and zero-shot Qwen judge.

Requires features_v2.pkl from modal_extract_features.py.

Usage:
    modal run --detach modal_baselines_gpu.py
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
        "sentence-transformers>=3.0.0",
        "loguru>=0.7.3",
        "numpy",
        "scikit-learn",
    )
    .add_local_python_source("activerb")
)

app = modal.App("activerb-baselines-gpu", image=image)

MODEL_ID = "Qwen/Qwen3-4B"
SENTENCE_MODEL = "BAAI/bge-small-en-v1.5"
HF_CACHE_DIR = "/hf-cache"
OUTPUT_DIR = "/outputs"
FEATURES_PATH = "/outputs/probe/features_v2.pkl"
GPU_RESULTS_PATH = "/outputs/probe/gpu_baselines.json"


@app.function(
    gpu="H100",
    timeout=60 * 60,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        OUTPUT_DIR: outputs_vol,
    },
)
def run_gpu_baselines() -> None:
    import json
    import os
    import pickle
    from pathlib import Path

    import numpy as np
    import torch
    from loguru import logger
    from sentence_transformers import SentenceTransformer

    from activerb.baselines_eval import evaluate_array

    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_DIR}/transformers"

    with Path(FEATURES_PATH).open("rb") as f:
        payload = pickle.load(f)
    records = payload["records"]
    logger.info(f"Loaded {len(records)} feature records")

    y = np.array([r["label"] for r in records], dtype=int)
    groups = np.array([r["thread_id"] for r in records])
    target_texts = [r["meta"]["target_text"] for r in records]
    full_texts = [r["meta"]["full_text"] for r in records]

    results: list[dict] = []

    # E1: sentence-transformer embeddings
    st_model = SentenceTransformer(SENTENCE_MODEL, device="cuda")
    for _field, texts, method_id, desc in [
        ("target", target_texts, "E1_target", "Sentence-transformer on target text"),
        ("full", full_texts, "E1_full", "Sentence-transformer on full transcript"),
    ]:
        emb = st_model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        X = np.asarray(emb, dtype=np.float64)
        ev = evaluate_array(method_id, desc, X, y, groups)
        results.append(ev.to_dict())
        logger.info(f"{method_id}: AUC={ev.auc_mean:.3f}±{ev.auc_std:.3f}")

    del st_model
    torch.cuda.empty_cache()

    # Z1: zero-shot Qwen judge — P(yes) from next-token logits
    from activerb.model import load_model

    device = torch.device("cuda")
    model, tokenizer = load_model(MODEL_ID, device=device, dtype=torch.bfloat16)
    model.eval()

    # Token ids for yes/no (first token of generation)
    yes_ids = tokenizer.encode(" yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" no", add_special_tokens=False)
    if not yes_ids or not no_ids:
        yes_ids = tokenizer.encode("yes", add_special_tokens=False)
        no_ids = tokenizer.encode("no", add_special_tokens=False)
    yes_id, no_id = yes_ids[0], no_ids[0]

    judge_prompt = (
        "You are evaluating whether an AI assistant was confused when responding to a user.\n"
        "Confusion means the assistant lacked context, hedged excessively, or misled due to uncertainty.\n\n"
        "Conversation:\n{conversation}\n\n"
        "Was the assistant confused in its most recent response? Answer with a single word: yes or no."
    )

    scores: list[float] = []
    for full in full_texts:
        messages = [{"role": "user", "content": judge_prompt.format(conversation=full[-3500:])}]
        try:
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        if isinstance(token_ids, list):
            ids_list = token_ids
        else:
            ids_list = token_ids["input_ids"]
            if hasattr(ids_list, "tolist"):
                ids_list = ids_list.tolist()
        input_ids = torch.tensor([ids_list], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=input_ids).logits[0, -1, :]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        p_yes = float(torch.exp(log_probs[yes_id]).item())
        p_no = float(torch.exp(log_probs[no_id]).item())
        scores.append(p_yes / max(p_yes + p_no, 1e-8))

    scores_arr = np.array(scores, dtype=np.float64).reshape(-1, 1)
    z1 = evaluate_array("Z1", "Zero-shot Qwen3-4B judge P(yes)", scores_arr, y, groups, use_pca=False)
    results.append(z1.to_dict())
    logger.info(f"Z1: AUC={z1.auc_mean:.3f}±{z1.auc_std:.3f}")

    out = Path(GPU_RESULTS_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    outputs_vol.commit()
    logger.info(f"Wrote {len(results)} GPU baseline results to {out}")


@app.local_entrypoint()
def main() -> None:
    run_gpu_baselines.remote()
