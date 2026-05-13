"""Training example construction for activation oracle training.

An oracle training example consists of:
  - A prompt asking about activations (e.g. "predict the previous 5 tokens")
  - K placeholder tokens (" ?") in the prompt where subject activations will be injected
  - The subject model's hidden states [K, D] to inject at those positions
  - The expected oracle response as the training target
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from transformers import PreTrainedTokenizerBase

# The paper uses " ?" (space + question mark) as the placeholder token.
# It must tokenize to a single token ID for position detection to work.
SPECIAL_TOKEN = " ?"


def get_introspection_prefix(layer: int, num_positions: int) -> str:
    """Build the prefix injected before the user prompt.

    Format matches the reference implementation:
        "Layer: {layer}\\n" + " ?" * num_positions + " \\n"
    """
    return f"Layer: {layer}\n" + SPECIAL_TOKEN * num_positions + " \n"


def _find_special_token_positions(
    token_ids: list[int],
    special_token_id: int,
    num_positions: int,
) -> list[int]:
    """Find the first `num_positions` occurrences of `special_token_id` in `token_ids`."""
    positions = []
    for i, tid in enumerate(token_ids):
        if tid == special_token_id:
            positions.append(i)
        if len(positions) == num_positions:
            break
    if len(positions) != num_positions:
        raise ValueError(
            f"Expected {num_positions} occurrences of special token {special_token_id!r}, "
            f"found {len(positions)}"
        )
    return positions


@dataclass
class TrainingExample:
    """A single oracle training example."""

    input_ids: list[int]
    labels: list[int]  # -100 for prompt tokens (ignored during loss)
    layer: int  # subject model layer the activations came from
    steering_vectors: Float[torch.Tensor, "k hidden"]  # subject hidden states
    positions: list[int]  # indices in input_ids where steering applies
    target_output: str  # human-readable target (for logging/eval)
    datapoint_type: str = "unknown"  # e.g. "past_lens", "latentqa"


def make_training_example(
    prompt: str,
    target: str,
    layer: int,
    acts: Float[torch.Tensor, "k hidden"],
    tokenizer: PreTrainedTokenizerBase,
    datapoint_type: str = "unknown",
) -> TrainingExample:
    """Build a TrainingExample from a prompt, target, and subject activations.

    The prefix (with K placeholder tokens) is prepended to `prompt`.
    The model is trained to predict `target` given the injection.
    """
    num_positions = acts.shape[0]
    prefix = get_introspection_prefix(layer, num_positions)

    full_prompt = prefix + prompt
    input_messages = [{"role": "user", "content": full_prompt}]

    def _apply(messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
        # enable_thinking=False is a Qwen3-specific kwarg; fall back gracefully.
        try:
            result = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            result = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        # transformers >= 4.50 may return BatchEncoding instead of a plain list
        if isinstance(result, list):
            return result
        return list(result["input_ids"])

    prompt_ids: list[int] = _apply(input_messages, add_generation_prompt=True)

    full_messages = [*input_messages, {"role": "assistant", "content": target}]
    full_ids: list[int] = _apply(full_messages, add_generation_prompt=False)

    # Mask the prompt portion (loss only on assistant tokens)
    labels = list(full_ids)
    for i in range(len(prompt_ids)):
        labels[i] = -100

    # Locate placeholder token positions
    special_token_ids = tokenizer.encode(SPECIAL_TOKEN, add_special_tokens=False)
    if len(special_token_ids) != 1:
        raise ValueError(
            f"SPECIAL_TOKEN {SPECIAL_TOKEN!r} must be a single token but got: {special_token_ids}"
        )
    positions = _find_special_token_positions(full_ids, special_token_ids[0], num_positions)

    return TrainingExample(
        input_ids=full_ids,
        labels=labels,
        layer=layer,
        steering_vectors=acts.cpu().detach(),
        positions=positions,
        target_output=target,
        datapoint_type=datapoint_type,
    )


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


@dataclass
class BatchedExamples:
    """Padded batch ready for the oracle forward pass."""

    input_ids: Int[torch.Tensor, "batch seq"]
    labels: Int[torch.Tensor, "batch seq"]
    attention_mask: torch.Tensor  # bool [batch, seq]
    steering_vectors: list[Float[torch.Tensor, "k hidden"]]  # one per batch element
    positions: list[list[int]]  # padded positions, one list per batch element


def batch_examples(
    examples: list[TrainingExample],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> BatchedExamples:
    """Left-pad a list of TrainingExamples into a single batch (mirrors construct_batch)."""
    max_len = max(len(e.input_ids) for e in examples)
    pad_id = tokenizer.pad_token_id

    input_ids_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    mask_rows: list[list[bool]] = []
    all_positions: list[list[int]] = []
    all_vectors: list[Float[torch.Tensor, "k hidden"]] = []

    for e in examples:
        pad = max_len - len(e.input_ids)
        input_ids_rows.append([pad_id] * pad + e.input_ids)
        label_rows.append([-100] * pad + e.labels)
        mask_rows.append([False] * pad + [True] * len(e.input_ids))
        all_positions.append([p + pad for p in e.positions])
        all_vectors.append(e.steering_vectors.to(device))

    return BatchedExamples(
        input_ids=torch.tensor(input_ids_rows, dtype=torch.long, device=device),
        labels=torch.tensor(label_rows, dtype=torch.long, device=device),
        attention_mask=torch.tensor(mask_rows, dtype=torch.bool, device=device),
        steering_vectors=all_vectors,
        positions=all_positions,
    )
