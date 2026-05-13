"""Oracle training loop.

Trains a LoRA adapter on top of the oracle model so it learns to interpret
subject-model activations injected into its residual stream at layer 1.

Usage (single GPU):
    python -m activerb.train

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=<N> -m activerb.train
"""

import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger
from peft import LoraConfig, get_peft_model
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from activerb.data import BatchedExamples, TrainingExample, batch_examples
from activerb.hooks import steered_forward


@dataclass
class TrainConfig:
    """Hyperparameters for oracle training (mirrors the paper's defaults)."""

    # --- LoRA ---
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: str = "all-linear"

    # --- Optimisation ---
    lr: float = 1e-5
    num_epochs: int = 1
    batch_size: int = 4  # per-device (paper uses 16 global across 4 GPUs)
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # --- Oracle injection ---
    hook_layer: int = 1  # layer in oracle model where subject acts are injected
    steering_coefficient: float = 1.0

    # --- Logging / saving ---
    save_dir: str = "checkpoints"
    save_steps: int = 500
    log_every: int = 10

    # --- Misc ---
    seed: int = 42
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])


def _setup_lora(model: torch.nn.Module, cfg: TrainConfig) -> torch.nn.Module:
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    model.print_trainable_parameters()
    return model


def _forward_with_loss(
    oracle_model: torch.nn.Module,
    batch: BatchedExamples,
    cfg: TrainConfig,
) -> torch.Tensor:
    """Single oracle forward pass with steering hook; returns cross-entropy loss."""
    with steered_forward(
        oracle_model,
        layer=cfg.hook_layer,
        vectors=batch.steering_vectors,
        positions=batch.positions,
        coefficient=cfg.steering_coefficient,
    ):
        out = oracle_model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
    return out.loss


def train_oracle(
    oracle_model: torch.nn.Module,
    training_data: list[TrainingExample],
    tokenizer: object,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    save_final: bool = True,
) -> torch.nn.Module:
    """Train the oracle model with LoRA on the provided training examples.

    Args:
        oracle_model: The base oracle model (will be wrapped with LoRA).
        training_data: List of TrainingExamples with subject activations.
        tokenizer: The oracle tokenizer.
        cfg: Training hyperparameters.
        device: Target device.
        dtype: Model dtype.
        save_final: Whether to save the final LoRA adapter.

    Returns:
        The trained PEFT model.
    """
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    oracle_model = _setup_lora(oracle_model, cfg)
    oracle_model.enable_input_require_grads()
    oracle_model.train()

    optimizer = torch.optim.AdamW(oracle_model.parameters(), lr=cfg.lr)

    steps_per_epoch = max(1, len(training_data) // cfg.batch_size)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    global_step = 0
    accumulated_loss = 0.0

    for epoch in range(cfg.num_epochs):
        random.shuffle(training_data)
        optimizer.zero_grad()

        batches = range(0, len(training_data) - cfg.batch_size + 1, cfg.batch_size)
        for step_i, start in enumerate(tqdm(batches, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")):
            items = training_data[start : start + cfg.batch_size]
            batch = batch_examples(items, tokenizer, device)

            loss = _forward_with_loss(oracle_model, batch, cfg)
            (loss / cfg.gradient_accumulation_steps).backward()
            accumulated_loss += loss.item() / cfg.gradient_accumulation_steps

            is_update = (step_i + 1) % cfg.gradient_accumulation_steps == 0
            if is_update:
                clip_grad_norm_(oracle_model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % cfg.log_every == 0:
                    logger.info(
                        f"step={global_step} loss={accumulated_loss:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                accumulated_loss = 0.0

                if global_step > 0 and global_step % cfg.save_steps == 0:
                    ckpt = Path(cfg.save_dir) / f"step_{global_step}"
                    oracle_model.save_pretrained(str(ckpt))
                    logger.info(f"Saved checkpoint → {ckpt}")

                global_step += 1

    if save_final:
        final_path = Path(cfg.save_dir) / "final"
        oracle_model.save_pretrained(str(final_path))
        logger.info(f"Saved final adapter → {final_path}")

    return oracle_model
