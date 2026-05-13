"""PastLens dataset: self-supervised activation interpretation.

The oracle is trained to predict tokens that preceded (or followed) a window
of activation positions in the subject model.  No labelled data is required —
supervision comes directly from the text itself.

Default data source: wikitext-103 (downloads once, then cached locally).
The paper uses FineWeb; set hf_dataset/hf_config/hf_text_key in PastLensConfig
to switch sources.
"""

import random
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from activerb.data import TrainingExample, make_training_example
from activerb.hooks import collect_activations_batch, layer_percent_to_layer


@dataclass
class PastLensConfig:
    """Configuration for PastLens dataset collection."""

    # Token prediction window
    min_k_tokens: int = 1
    max_k_tokens: int = 20
    # Number of activation positions to inject per example
    min_k_acts: int = 1
    max_k_acts: int = 20
    # Maximum sequence length for subject model forward passes
    max_length: int = 512
    # Which fractions of the subject model to extract activations from
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])
    # Mix of "past" (predict preceding tokens) and "future" (predict following tokens)
    directions: list[str] = field(default_factory=lambda: ["past", "future"])
    # HuggingFace dataset settings (wikitext caches locally; change to fineweb for paper scale)
    hf_dataset: str = "Salesforce/wikitext"
    hf_config: str = "wikitext-103-raw-v1"
    hf_split: str = "train"
    hf_text_key: str = "text"
    # Streaming batch size for subject model
    collection_batch_size: int = 8


def _text_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    hf_dataset: str,
    hf_config: str,
    hf_split: str,
    hf_text_key: str,
    max_length: int,
    min_tokens: int = 10,
):
    """Yield token-ID lists from any HuggingFace text dataset.

    The dataset is downloaded and cached on first use; subsequent runs are instant.
    For large corpora (FineWeb), use streaming=True by setting the env var
    HF_DATASETS_OFFLINE=0 and ensuring a stable network connection.
    """
    # Stream large corpora (FineWeb) to avoid downloading the whole thing;
    # cache small ones (wikitext) for fast repeated access.
    large_datasets = {"HuggingFaceFW/fineweb", "allenai/c4"}
    streaming = hf_dataset in large_datasets
    logger.info(
        f"Loading dataset {hf_dataset} ({hf_config or 'default'}, split={hf_split}, "
        f"streaming={streaming})…"
    )
    ds = load_dataset(hf_dataset, hf_config, split=hf_split, streaming=streaming)
    for sample in ds:
        text = sample[hf_text_key]
        if not text or not text.strip():
            continue
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        if len(ids) >= min_tokens:
            yield ids


def _make_example_from_batch(
    token_ids: list[int],
    acts_LD: torch.Tensor,   # [seq_len, hidden] after removing padding
    layer: int,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PastLensConfig,
) -> TrainingExample | None:
    """Build one PastLens TrainingExample from a single sequence and its activations."""
    L = len(token_ids)
    k_tokens = random.randint(cfg.min_k_tokens, cfg.max_k_tokens)
    k_acts = random.randint(cfg.min_k_acts, cfg.max_k_acts)
    direction = random.choice(cfg.directions)

    if direction == "past":
        # Need: k_tokens tokens before act window, k_acts activations, ≥1 token after
        if k_tokens + k_acts + 1 > L:
            return None
        act_start = random.randint(k_tokens, L - k_acts - 1)
        act_positions = list(range(act_start, act_start + k_acts))
        target_tokens = token_ids[act_start - k_tokens : act_start]
        prompt = f"Can you predict the previous {k_tokens} tokens that came before this?"
    else:
        # Need: ≥1 token before act window, k_acts activations, k_tokens after
        if k_tokens + k_acts + 1 > L:
            return None
        act_start = random.randint(1, L - k_acts - k_tokens)
        act_positions = list(range(act_start, act_start + k_acts))
        last = act_positions[-1]
        target_tokens = token_ids[last + 1 : last + 1 + k_tokens]
        prompt = f"Can you predict the next {k_tokens} tokens that come after this?"

    target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
    acts_KD = acts_LD[act_positions, :].clone()  # [k_acts, hidden]

    return make_training_example(
        prompt=prompt,
        target=target_text,
        layer=layer,
        acts=acts_KD,
        tokenizer=tokenizer,
        datapoint_type="past_lens",
    )


def build_past_lens_dataset(
    subject_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: PastLensConfig,
    num_examples: int,
    device: torch.device,
    seed: int = 42,
) -> list[TrainingExample]:
    """Collect `num_examples` PastLens training examples from the subject model.

    Runs the subject model in inference mode (no gradients) to extract activations
    from layers at the specified percentages of depth.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    layers = [layer_percent_to_layer(subject_model, p) for p in cfg.layer_percents]
    logger.info(f"Collecting PastLens acts from layers {layers} (percents {cfg.layer_percents})")

    data_gen = _text_token_ids(
        tokenizer,
        hf_dataset=cfg.hf_dataset,
        hf_config=cfg.hf_config,
        hf_split=cfg.hf_split,
        hf_text_key=cfg.hf_text_key,
        max_length=cfg.max_length,
    )
    examples: list[TrainingExample] = []

    pad_id = tokenizer.pad_token_id
    B = cfg.collection_batch_size

    with tqdm(total=num_examples, desc="PastLens") as pbar:
        buf: list[list[int]] = []
        for token_ids in data_gen:
            buf.append(token_ids)
            if len(buf) < B:
                continue

            # Build a left-padded batch for the subject model
            max_len = max(len(t) for t in buf)
            input_ids_list = [[pad_id] * (max_len - len(t)) + t for t in buf]
            input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
            attn_mask = input_ids.ne(pad_id)
            inputs = {"input_ids": input_ids, "attention_mask": attn_mask}

            # One forward pass collects all needed layers
            acts_by_layer = collect_activations_batch(subject_model, layers, inputs)

            layer = random.choice(layers)
            acts_BLD = acts_by_layer[layer]  # [B, L, D], float32

            for j, token_ids_j in enumerate(buf):
                pad = max_len - len(token_ids_j)
                # Strip padding: activations for non-padded tokens
                acts_LD = acts_BLD[j, pad:, :]  # [L_j, D]

                example = _make_example_from_batch(token_ids_j, acts_LD, layer, tokenizer, cfg)
                if example is None:
                    continue
                examples.append(example)
                pbar.update(1)
                if len(examples) >= num_examples:
                    break

            buf = []
            if len(examples) >= num_examples:
                break

    logger.info(f"Collected {len(examples)} PastLens examples")
    return examples
