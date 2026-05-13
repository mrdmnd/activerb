"""Modal app for running the activation oracle experiment on cloud GPUs.

Usage:
    # Authenticate once
    modal token new

    # Run the full experiment (A100, ~2 hours, ~$3-5)
    modal run modal_experiment.py

    # Download results when done
    modal volume get activerb-outputs /runs .

The HuggingFace model cache is stored in a persistent Modal volume so
subsequent runs don't re-download Qwen3-4B.
"""

import modal

# ── Persistent volumes ────────────────────────────────────────────────────────

# Caches HuggingFace model weights across runs (~8 GB for Qwen3-4B)
hf_cache_vol = modal.Volume.from_name("activerb-hf-cache", create_if_missing=True)

# Stores experiment outputs (checkpoints, logs, sample examples)
outputs_vol = modal.Volume.from_name("activerb-outputs", create_if_missing=True)

# ── Container image ───────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "torchvision",
        "transformers>=4.50.0",
        "peft>=0.14.0",
        "datasets>=3.0.0",
        "accelerate>=1.0.0",
        "loguru>=0.7.3",
        "tqdm>=4.67.3",
        "jaxtyping>=0.3.9",
        "numpy",
    )
    .add_local_python_source("activerb")  # mounts our local package into the container
)

app = modal.App("activerb-experiment", image=image)

# ── Experiment settings ───────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-4B"

# Scale settings — adjust to taste / budget
NUM_EXAMPLES = 100_000      # paper scale; ~45 min data collection on A100
COLLECTION_BATCH_SIZE = 32  # larger batches = faster collection on GPU
TRAIN_BATCH_SIZE = 16       # matches paper default
GRADIENT_ACCUMULATION = 1
NUM_EPOCHS = 1
LOG_EVERY = 100
SAVE_STEPS = 1_000

# Fraction of training data that comes from classification task (rest = PastLens)
CLASSIFICATION_FRACTION = 0.5

# Paths inside the container
HF_CACHE_DIR = "/hf-cache"
OUTPUT_DIR = "/outputs"

# ── Main training function ────────────────────────────────────────────────────


@app.function(
    gpu="H100",          # ~$5.00/hr; faster than A100 for large batches
    timeout=60 * 60 * 4,  # 4-hour ceiling (data collection + training)
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        OUTPUT_DIR: outputs_vol,
    },
    secrets=[
        # Optional: set HF_TOKEN in a Modal secret for faster/private model downloads
        # modal secret create huggingface HF_TOKEN=hf_...
        # Then uncomment: modal.Secret.from_name("huggingface"),
    ],
)
def run_experiment(
    num_examples: int = NUM_EXAMPLES,
    collection_batch_size: int = COLLECTION_BATCH_SIZE,
    train_batch_size: int = TRAIN_BATCH_SIZE,
) -> None:
    import json
    import os
    from datetime import datetime
    from pathlib import Path

    import torch
    from loguru import logger

    from activerb.datasets.classification import ClassificationConfig, build_classification_dataset
    from activerb.datasets.past_lens import PastLensConfig, build_past_lens_dataset
    from activerb.eval import evaluate
    from activerb.model import load_model
    from activerb.train import TrainConfig, train_oracle

    # Point HuggingFace at the persistent cache volume
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE_DIR}/transformers"

    device = torch.device("cuda")
    dtype = torch.bfloat16

    run_dir = Path(OUTPUT_DIR) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run dir: {run_dir}")

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading {MODEL_ID}…")
    model, tokenizer = load_model(MODEL_ID, device=device, dtype=dtype)

    # ── Collect training data: 50% PastLens + 50% Classification ─────────────
    import random

    n_classification = int(num_examples * CLASSIFICATION_FRACTION)
    n_past_lens = num_examples - n_classification

    # Classification examples (SST-2 sentiment — label absent from prompt)
    cls_cfg = ClassificationConfig(
        min_k_acts=1,
        max_k_acts=20,
        min_end_offset=-1,
        max_end_offset=-5,
        max_length=512,
        layer_percents=[25, 50, 75],
        hf_dataset="stanfordnlp/sst2",
        hf_split="train",
        collection_batch_size=collection_batch_size,
    )
    # Collect 10% extra for the held-out eval set (eval = 5% of total)
    n_cls_total = int(n_classification * 1.1)
    logger.info(f"Collecting {n_cls_total} classification examples…")
    cls_all = build_classification_dataset(
        subject_model=model,
        tokenizer=tokenizer,
        cfg=cls_cfg,
        num_examples=n_cls_total,
        device=device,
        seed=42,
    )

    # PastLens examples (FineWeb)
    past_lens_cfg = PastLensConfig(
        min_k_tokens=1,
        max_k_tokens=20,
        min_k_acts=1,
        max_k_acts=20,
        max_length=512,
        layer_percents=[25, 50, 75],
        directions=["past", "future"],
        hf_dataset="HuggingFaceFW/fineweb",
        hf_config="default",
        hf_split="train",
        hf_text_key="text",
        collection_batch_size=collection_batch_size,
    )
    n_pl_total = int(n_past_lens * 1.1)
    logger.info(f"Collecting {n_pl_total} PastLens examples…")
    pl_all = build_past_lens_dataset(
        subject_model=model,
        tokenizer=tokenizer,
        cfg=past_lens_cfg,
        num_examples=n_pl_total,
        device=device,
        seed=43,
    )

    # Build train and eval splits, keeping task proportions in eval set
    cls_train = cls_all[:n_classification]
    cls_eval = cls_all[n_classification:]
    pl_train = pl_all[:n_past_lens]
    pl_eval = pl_all[n_past_lens:]

    training_data = cls_train + pl_train
    eval_data = cls_eval + pl_eval
    random.seed(42)
    random.shuffle(training_data)
    logger.info(
        f"Train: {len(training_data)} ({len(cls_train)} cls + {len(pl_train)} pastlens)  "
        f"Eval: {len(eval_data)}"
    )

    # Save sample examples for inspection
    samples = [
        {"layer": ex.layer, "num_positions": len(ex.positions), "target": ex.target_output}
        for ex in training_data[:10]
    ]
    (run_dir / "sample_examples.json").write_text(json.dumps(samples, indent=2))
    outputs_vol.commit()

    # ── Train oracle ──────────────────────────────────────────────────────────
    train_cfg = TrainConfig(
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules="all-linear",
        lr=1e-5,
        num_epochs=NUM_EPOCHS,
        batch_size=train_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        max_grad_norm=1.0,
        hook_layer=1,
        steering_coefficient=1.0,
        save_dir=str(run_dir / "checkpoints"),
        save_steps=SAVE_STEPS,
        log_every=LOG_EVERY,
        seed=42,
        layer_percents=[25, 50, 75],
    )

    # ── Eval before training (untrained baseline) ─────────────────────────────
    logger.info("Pre-training eval…")
    pre_results = evaluate(
        oracle_model=model,
        eval_data=eval_data,
        tokenizer=tokenizer,
        device=device,
        hook_layer=train_cfg.hook_layer,
        batch_size=train_batch_size * 2,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting oracle training…")
    trained_model = train_oracle(
        oracle_model=model,
        training_data=training_data,
        tokenizer=tokenizer,
        cfg=train_cfg,
        device=device,
        dtype=dtype,
        save_final=True,
    )

    # ── Eval after training ───────────────────────────────────────────────────
    logger.info("Post-training eval…")
    post_results = evaluate(
        oracle_model=trained_model,
        eval_data=eval_data,
        tokenizer=tokenizer,
        device=device,
        hook_layer=train_cfg.hook_layer,
        batch_size=train_batch_size * 2,
    )

    # Save results summary
    summary = {
        "num_train": len(training_data),
        "num_eval": len(eval_data),
        "pre_training": pre_results,
        "post_training": post_results,
        "improvement": {
            "oracle_loss_delta": pre_results["oracle_loss"] - post_results["oracle_loss"],
            "baseline_gap_delta": post_results["delta"] - pre_results["delta"],
        },
    }
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Results: {summary}")

    outputs_vol.commit()
    logger.info(f"Done. Results in Modal volume 'activerb-outputs' at {run_dir.name}/")


# ── Local entrypoint ──────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    num_examples: int = NUM_EXAMPLES,
    collection_batch_size: int = COLLECTION_BATCH_SIZE,
    train_batch_size: int = TRAIN_BATCH_SIZE,
) -> None:
    """Launch the experiment remotely.

    Examples:
        # Full paper-scale run
        modal run modal_experiment.py

        # Quick smoke test (500 examples, verify pipeline works)
        modal run modal_experiment.py --num-examples 500
    """
    print(f"Launching experiment: {num_examples} examples on H100…")
    run_experiment.remote(
        num_examples=num_examples,
        collection_batch_size=collection_batch_size,
        train_batch_size=train_batch_size,
    )
    print("Done. Fetch results with: modal volume get activerb-outputs /. ./modal_results")
