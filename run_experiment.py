"""Run the activation oracle replication experiment.

Trains an activation oracle on Qwen/Qwen3-4B using the PastLens task:
given hidden-state activations at K positions in the subject model, predict
the tokens that preceded (or followed) those positions.

Subject model = Oracle base model = Qwen/Qwen3-4B (same weights, oracle gets LoRA).
"""

import json
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

from activerb.datasets.past_lens import PastLensConfig, build_past_lens_dataset
from activerb.model import get_device, load_model
from activerb.train import TrainConfig, train_oracle

# ── Experiment settings ───────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-4B"

# Data collection
NUM_EXAMPLES = 5_000          # scale up to 100k to match the paper
COLLECTION_BATCH_SIZE = 8

# Training
TRAIN_BATCH_SIZE = 4
NUM_EPOCHS = 1
LOG_EVERY = 50
SAVE_STEPS = 500

RUN_DIR = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Run dir: {RUN_DIR}")

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading subject/oracle base model…")
    model, tokenizer = load_model(MODEL_ID, device=device)

    # ── Collect PastLens training data ────────────────────────────────────────
    past_lens_cfg = PastLensConfig(
        min_k_tokens=1,
        max_k_tokens=20,
        min_k_acts=1,
        max_k_acts=20,
        max_length=512,
        layer_percents=[25, 50, 75],
        directions=["past", "future"],
        # wikitext-103 (~500 MB, cached after first download)
        # Switch to hf_dataset="HuggingFaceFW/fineweb" for paper-scale training
        hf_dataset="Salesforce/wikitext",
        hf_config="wikitext-103-raw-v1",
        hf_split="train",
        hf_text_key="text",
        collection_batch_size=COLLECTION_BATCH_SIZE,
    )

    logger.info(f"Collecting {NUM_EXAMPLES} PastLens examples from FineWeb…")
    training_data = build_past_lens_dataset(
        subject_model=model,
        tokenizer=tokenizer,
        cfg=past_lens_cfg,
        num_examples=NUM_EXAMPLES,
        device=device,
        seed=42,
    )
    logger.info(f"Dataset ready: {len(training_data)} examples")

    # Log a few examples for inspection
    examples_log = []
    for ex in training_data[:5]:
        examples_log.append({
            "layer": ex.layer,
            "num_positions": len(ex.positions),
            "target": ex.target_output,
        })
    (RUN_DIR / "sample_examples.json").write_text(json.dumps(examples_log, indent=2))

    # ── Train oracle ──────────────────────────────────────────────────────────
    train_cfg = TrainConfig(
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules="all-linear",
        lr=1e-5,
        num_epochs=NUM_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        hook_layer=1,
        steering_coefficient=1.0,
        save_dir=str(RUN_DIR / "checkpoints"),
        save_steps=SAVE_STEPS,
        log_every=LOG_EVERY,
        seed=42,
        layer_percents=[25, 50, 75],
    )

    logger.info("Starting oracle training…")
    trained_model = train_oracle(
        oracle_model=model,
        training_data=training_data,
        tokenizer=tokenizer,
        cfg=train_cfg,
        device=device,
        dtype=torch.bfloat16,
        save_final=True,
    )

    logger.info(f"Done. Adapter saved to {train_cfg.save_dir}/final")


if __name__ == "__main__":
    main()
