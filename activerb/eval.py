"""Evaluation for the activation oracle.

Three baselines, in increasing strength:

  oracle_loss        — real subject activations, correct label
  random_loss        — random Gaussian vectors (same shape/norm), random direction
  shuffled_loss      — real activations from a *different* example in the batch
                       (controls for "real vs. fake" distributional confound)
  label_swapped_loss — real activations from an opposite-label example
                       (classification only; the gold-standard confound control)

The gaps that matter:
  shuffled_loss  - oracle_loss  → oracle reads activation *content*, not just geometry
  label_swapped_loss - oracle_loss → oracle reads *label-specific* content
"""

import random

import torch
from loguru import logger
from tqdm import tqdm

from activerb.data import TrainingExample, batch_examples
from activerb.hooks import steered_forward


def _run_batched(
    oracle_model: torch.nn.Module,
    batches: list[list[TrainingExample]],
    tokenizer: object,
    device: torch.device,
    hook_layer: int,
    steering_coefficient: float,
    override_vectors: list | None = None,  # if set, list of [K,D] tensors per example
    desc: str = "",
) -> float:
    """Run oracle over pre-grouped batches, optionally overriding steering vectors."""
    losses: list[float] = []
    flat_idx = 0
    for items in tqdm(batches, desc=desc, leave=False):
        batch = batch_examples(items, tokenizer, device)
        if override_vectors is not None:
            n = len(items)
            new_vecs = [override_vectors[flat_idx + i].to(device) for i in range(n)]
            batch.steering_vectors = new_vecs
            # Clip positions to match the override vector length (may differ when swapping labels)
            batch.positions = [pos[: vec.shape[0]] for pos, vec in zip(batch.positions, new_vecs)]
            flat_idx += n
        with steered_forward(oracle_model, hook_layer, batch.steering_vectors, batch.positions, steering_coefficient):
            loss = oracle_model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
            ).loss
        losses.append(loss.item())
    return sum(losses) / len(losses) if losses else float("nan")


@torch.no_grad()
def evaluate(
    oracle_model: torch.nn.Module,
    eval_data: list[TrainingExample],
    tokenizer: object,
    device: torch.device,
    hook_layer: int = 1,
    steering_coefficient: float = 1.0,
    batch_size: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate the oracle on a held-out set.

    Returns a dict with oracle_loss, random_loss, shuffled_loss,
    label_swapped_loss (classification examples only), and their deltas.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)
    oracle_model.eval()

    # Build batches once, reuse for all passes
    n = (len(eval_data) // batch_size) * batch_size  # drop last partial batch
    items_flat = eval_data[:n]
    batches = [items_flat[i : i + batch_size] for i in range(0, n, batch_size)]

    # ── 1. Oracle loss (real activations) ─────────────────────────────────────
    oracle_loss = _run_batched(oracle_model, batches, tokenizer, device, hook_layer, steering_coefficient, desc="Eval/oracle")

    # ── 2. Random baseline (same norm, random direction) ──────────────────────
    rand_vecs: list[torch.Tensor] = []
    for ex in items_flat:
        vecs = ex.steering_vectors.float()
        norms = vecs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        rand_dir = torch.randn_like(vecs)
        rand_dir /= rand_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        rand_vecs.append((rand_dir * norms).to(ex.steering_vectors.dtype))
    random_loss = _run_batched(
        oracle_model, batches, tokenizer, device, hook_layer, steering_coefficient,
        override_vectors=rand_vecs, desc="Eval/random",
    )

    # ── 3. Shuffled baseline (real acts, wrong example — rotate within batch) ─
    shuffled_vecs: list[torch.Tensor] = []
    for i in range(0, n, batch_size):
        chunk = [ex.steering_vectors for ex in items_flat[i : i + batch_size]]
        shuffled_vecs.extend([*chunk[1:], chunk[0]])  # shift by 1
    shuffled_loss = _run_batched(
        oracle_model, batches, tokenizer, device, hook_layer, steering_coefficient,
        override_vectors=shuffled_vecs, desc="Eval/shuffled",
    )

    results: dict[str, float] = {
        "oracle_loss": oracle_loss,
        "random_loss": random_loss,
        "shuffled_loss": shuffled_loss,
        "delta_vs_random": random_loss - oracle_loss,
        "delta_vs_shuffled": shuffled_loss - oracle_loss,
    }

    # ── 4. Label-swapped baseline (classification only) ───────────────────────
    cls_pos = [(i, ex) for i, ex in enumerate(items_flat) if ex.datapoint_type == "classification" and ex.target_output == "positive"]
    cls_neg = [(i, ex) for i, ex in enumerate(items_flat) if ex.datapoint_type == "classification" and ex.target_output == "negative"]

    if cls_pos and cls_neg:
        # Pair up positive ↔ negative, truncate to equal length
        n_pairs = min(len(cls_pos), len(cls_neg))
        rng.shuffle(cls_pos)
        rng.shuffle(cls_neg)
        cls_pos = cls_pos[:n_pairs]
        cls_neg = cls_neg[:n_pairs]

        # Build swapped-vector eval: positives get negative acts, negatives get positive acts
        swapped_examples: list[TrainingExample] = []
        swapped_vecs: list[torch.Tensor] = []
        for (_, pos_ex), (_, neg_ex) in zip(cls_pos, cls_neg):
            swapped_examples.append(pos_ex)
            swapped_vecs.append(neg_ex.steering_vectors)  # wrong label acts
            swapped_examples.append(neg_ex)
            swapped_vecs.append(pos_ex.steering_vectors)  # wrong label acts

        n_swap = (len(swapped_examples) // batch_size) * batch_size
        swap_batches = [swapped_examples[i : i + batch_size] for i in range(0, n_swap, batch_size)]
        swap_vecs_trunc = swapped_vecs[:n_swap]

        label_swapped_loss = _run_batched(
            oracle_model, swap_batches, tokenizer, device, hook_layer, steering_coefficient,
            override_vectors=swap_vecs_trunc, desc="Eval/label-swapped"
        )

        # Also compute oracle loss on the same subset for a fair comparison
        oracle_cls_batches = [swapped_examples[i : i + batch_size] for i in range(0, n_swap, batch_size)]
        oracle_cls_loss = _run_batched(oracle_model, oracle_cls_batches, tokenizer, device, hook_layer, steering_coefficient, desc="Eval/oracle-cls")

        results["label_swapped_loss"] = label_swapped_loss
        results["oracle_cls_loss"] = oracle_cls_loss
        results["delta_vs_label_swapped"] = label_swapped_loss - oracle_cls_loss
        results["n_label_swap_pairs"] = n_pairs
    else:
        logger.warning("Not enough classification examples for label-swapped baseline")

    oracle_model.train()

    logger.info(
        f"Eval — oracle: {oracle_loss:.4f}  random: {random_loss:.4f}  "
        f"shuffled: {shuffled_loss:.4f}  "
        f"Δ_random: {results['delta_vs_random']:+.4f}  "
        f"Δ_shuffled: {results['delta_vs_shuffled']:+.4f}"
        + (f"  Δ_label_swap: {results.get('delta_vs_label_swapped', float('nan')):+.4f}" if "delta_vs_label_swapped" in results else "")
    )
    return results
