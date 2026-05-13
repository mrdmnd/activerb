"""Load the saved oracle checkpoint and run the enhanced three-baseline evaluation.

Usage:
    # Run against the latest checkpoint in the volume
    modal run --detach modal_eval_only.py

    # Run against a specific run dir
    modal run --detach modal_eval_only.py --run-dir 20260513_021015

Results are saved to the same run dir in the volume as eval_enhanced.json.
"""

import modal

# ── Reuse volumes from the main experiment ────────────────────────────────────

hf_cache_vol = modal.Volume.from_name("activerb-hf-cache", create_if_missing=False)
outputs_vol = modal.Volume.from_name("activerb-outputs", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.50.0",
        "peft>=0.14.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "loguru>=0.7.3",
        "tqdm>=4.67.3",
        "jaxtyping>=0.3.9",
        "numpy",
    )
    .add_local_python_source("activerb")
)

app = modal.App("activerb-eval", image=image)

MODEL_ID = "Qwen/Qwen3-4B"
HF_CACHE_DIR = "/hf-cache"
OUTPUT_DIR = "/outputs"

NUM_EVAL = 2_000       # classification examples to collect (1k pos + 1k neg target)
EVAL_BATCH_SIZE = 32
COLLECTION_BATCH_SIZE = 32


@app.function(
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        OUTPUT_DIR: outputs_vol,
    },
)
def run_eval(run_dir: str = "") -> None:
    import json
    import os
    from pathlib import Path

    import torch
    from loguru import logger
    from peft import PeftModel

    from activerb.datasets.classification import ClassificationConfig, build_classification_dataset
    from activerb.eval import evaluate
    from activerb.model import load_model

    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_DIR}/transformers"

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ── Find checkpoint ───────────────────────────────────────────────────────
    outputs_path = Path(OUTPUT_DIR)
    if run_dir:
        target = outputs_path / run_dir
    else:
        # Pick the most recent run dir that has a final checkpoint
        candidates = sorted(
            [d for d in outputs_path.iterdir() if d.is_dir() and (d / "checkpoints" / "final").exists()],
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(f"No run dirs with checkpoints found in {OUTPUT_DIR}")
        target = candidates[0]

    checkpoint_path = target / "checkpoints" / "final"
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # ── Load base model + LoRA adapter ───────────────────────────────────────
    logger.info(f"Loading {MODEL_ID}…")
    model, tokenizer = load_model(MODEL_ID, device=device, dtype=dtype)
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()
    logger.info("Checkpoint loaded.")

    # ── Collect balanced classification eval set ──────────────────────────────
    # Collect 2x what we need so we can balance positive/negative labels
    cls_cfg = ClassificationConfig(
        min_k_acts=1,
        max_k_acts=20,
        min_end_offset=-1,
        max_end_offset=-5,
        max_length=512,
        layer_percents=[25, 50, 75],
        hf_dataset="stanfordnlp/sst2",
        hf_split="validation",   # use validation split — unseen during training
        collection_batch_size=COLLECTION_BATCH_SIZE,
    )
    logger.info(f"Collecting {NUM_EVAL} classification eval examples (validation split)…")
    eval_data = build_classification_dataset(
        subject_model=model,
        tokenizer=tokenizer,
        cfg=cls_cfg,
        num_examples=NUM_EVAL,
        device=device,
        seed=99,
    )

    pos = [ex for ex in eval_data if ex.target_output == "positive"]
    neg = [ex for ex in eval_data if ex.target_output == "negative"]
    logger.info(f"Eval set: {len(pos)} positive, {len(neg)} negative")

    # ── Run enhanced eval ─────────────────────────────────────────────────────
    logger.info("Running enhanced evaluation (oracle / random / shuffled / label-swapped)…")
    results = evaluate(
        oracle_model=model,
        eval_data=eval_data,
        tokenizer=tokenizer,
        device=device,
        hook_layer=1,
        batch_size=EVAL_BATCH_SIZE,
        seed=0,
    )

    logger.info(f"Results: {results}")

    out_file = target / "eval_enhanced.json"
    out_file.write_text(json.dumps(results, indent=2))
    outputs_vol.commit()
    logger.info(f"Saved to {out_file}")


@app.local_entrypoint()
def main(run_dir: str = "") -> None:
    """Run the enhanced evaluation against a saved checkpoint.

    Args:
        run_dir: Optional run directory name (e.g. 20260513_021015).
                 If omitted, uses the most recent checkpoint.
    """
    print(f"Launching eval against {run_dir if run_dir else 'latest checkpoint'}…")
    run_eval.remote(run_dir=run_dir)
    print("Done. Fetch results with: modal volume get activerb-outputs /. ./modal_results")
