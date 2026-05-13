"""Classification dataset: forces the oracle to read activations.

Unlike PastLens (which can be solved statistically), the oracle here must
predict a label (e.g. sentiment) that appears *nowhere* in its prompt.  The
only signal available is the subject-model activations injected at the
placeholder positions.  This creates genuine pressure to read them.

Default dataset: SST-2 (binary sentiment, ~67k examples, no download needed
beyond the HuggingFace hub).

Prompt format for oracle:
    "Answer with 'positive' or 'negative' only.
     What is the sentiment of this text?"

Target: "positive" or "negative"
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
class ClassificationConfig:
    """Configuration for classification dataset collection."""

    # Number of activation positions to inject per example
    min_k_acts: int = 1
    max_k_acts: int = 20
    # How far from the end of the tokenized text to place the activation window.
    # Negative offsets count from the last token.  E.g. min_end_offset=-1 means
    # the window can end at the last token; max_end_offset=-5 means it ends ≥5
    # tokens before the end (so there is context after the window).
    min_end_offset: int = -1
    max_end_offset: int = -5
    # Maximum sequence length for subject model
    max_length: int = 512
    # Which fractions of the subject model to extract activations from
    layer_percents: list[int] = field(default_factory=lambda: [25, 50, 75])
    # HuggingFace dataset settings
    hf_dataset: str = "stanfordnlp/sst2"
    hf_split: str = "train"
    # Streaming batch size for subject model
    collection_batch_size: int = 8


# Human-readable labels for SST-2 (0 = negative, 1 = positive)
_SST2_LABELS = {0: "negative", 1: "positive"}

_ORACLE_QUESTION = "Answer with 'positive' or 'negative' only.\nWhat is the sentiment of this text?"


def _load_sst2_examples(cfg: ClassificationConfig) -> list[tuple[str, str]]:
    """Return list of (sentence, label_str) from SST-2."""
    logger.info(f"Loading {cfg.hf_dataset} ({cfg.hf_split})…")
    ds = load_dataset(cfg.hf_dataset, split=cfg.hf_split)
    result: list[tuple[str, str]] = []
    for row in ds:
        sentence: str = row["sentence"]
        label_str: str = _SST2_LABELS[row["label"]]
        if sentence and sentence.strip():
            result.append((sentence, label_str))
    logger.info(f"Loaded {len(result)} SST-2 examples")
    return result


def build_classification_dataset(
    subject_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    cfg: ClassificationConfig,
    num_examples: int,
    device: torch.device,
    seed: int = 42,
) -> list[TrainingExample]:
    """Collect `num_examples` classification training examples.

    For each SST-2 sentence:
      1. Tokenize it and run the subject model to get hidden states.
      2. Pick a random window of K positions near the end of the sequence.
      3. Build an oracle training example whose prompt asks for sentiment
         but contains NO label — the oracle must read the activations.

    Returns a list of TrainingExample objects ready for oracle training.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    layers = [layer_percent_to_layer(subject_model, p) for p in cfg.layer_percents]
    logger.info(f"Collecting classification acts from layers {layers} (percents {cfg.layer_percents})")

    raw_examples = _load_sst2_examples(cfg)
    random.shuffle(raw_examples)

    pad_id = tokenizer.pad_token_id
    B = cfg.collection_batch_size

    examples: list[TrainingExample] = []

    # We cycle through the dataset (it may be smaller than num_examples)
    raw_cycle = raw_examples * (num_examples // len(raw_examples) + 2)

    with tqdm(total=num_examples, desc="Classification") as pbar:
        idx = 0
        while len(examples) < num_examples and idx < len(raw_cycle):
            # Fill a batch
            batch_texts: list[str] = []
            batch_labels: list[str] = []
            batch_token_ids: list[list[int]] = []

            while len(batch_texts) < B and idx < len(raw_cycle):
                sentence, label = raw_cycle[idx]
                idx += 1
                token_ids = tokenizer.encode(
                    sentence,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=cfg.max_length,
                )
                # Need enough tokens for at least max_k_acts positions + some margin
                if len(token_ids) < cfg.max_k_acts + 2:
                    continue
                batch_texts.append(sentence)
                batch_labels.append(label)
                batch_token_ids.append(token_ids)

            if not batch_texts:
                continue

            # Build left-padded batch for subject model
            max_len = max(len(t) for t in batch_token_ids)
            input_ids_list = [[pad_id] * (max_len - len(t)) + t for t in batch_token_ids]
            input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
            attn_mask = input_ids.ne(pad_id)
            inputs = {"input_ids": input_ids, "attention_mask": attn_mask}

            # One forward pass collects all needed layers
            acts_by_layer = collect_activations_batch(subject_model, layers, inputs)

            layer = random.choice(layers)
            acts_BLD = acts_by_layer[layer]  # [B, L, D], float32

            for j, (token_ids_j, label_j) in enumerate(zip(batch_token_ids, batch_labels)):
                L = len(token_ids_j)
                pad = max_len - L

                # Strip padding: activations for non-padded tokens
                acts_LD = acts_BLD[j, pad:, :]  # [L, D]

                # Pick a window of K positions ending somewhere near the end
                k_acts = random.randint(cfg.min_k_acts, min(cfg.max_k_acts, L))

                # end_offset is negative: -1 = last token, -5 = 5th from last
                # Clamp so we have room for k_acts positions before the end
                max_end = -1  # closest to end (least negative)
                min_end = -(L - k_acts)  # farthest from end (most negative), ensuring k_acts fits
                # Apply user-configured clamp
                min_end = max(min_end, cfg.max_end_offset)  # max_end_offset is more negative
                max_end = min(max_end, cfg.min_end_offset)  # min_end_offset is less negative

                if min_end > max_end:
                    # Sequence too short for the requested offset range; skip
                    continue

                end_offset = random.randint(min_end, max_end)  # e.g. -3
                end_pos = L + end_offset  # convert to absolute index (0-based)
                start_pos = end_pos - k_acts + 1
                if start_pos < 0:
                    continue

                act_positions = list(range(start_pos, end_pos + 1))
                acts_KD = acts_LD[act_positions, :].clone()  # [k_acts, hidden]

                example = make_training_example(
                    prompt=_ORACLE_QUESTION,
                    target=label_j,
                    layer=layer,
                    acts=acts_KD,
                    tokenizer=tokenizer,
                    datapoint_type="classification",
                )
                examples.append(example)
                pbar.update(1)
                if len(examples) >= num_examples:
                    break

    logger.info(f"Collected {len(examples)} classification examples")
    return examples
