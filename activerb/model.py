"""Model loading utilities for activation verbalization research."""

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_MODEL = "Qwen/Qwen3-4B"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    model_id: str = DEFAULT_MODEL,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal LM and its tokenizer onto the target device."""
    if device is None:
        device = get_device()

    logger.info(f"Loading {model_id} on {device} ({dtype})")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=str(device),
    )
    model.eval()

    logger.info(f"Loaded — {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    return model, tokenizer


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> str:
    """Run a single forward pass and return the decoded output."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )
    # strip the prompt tokens from the output
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)
