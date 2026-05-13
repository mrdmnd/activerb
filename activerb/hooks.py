"""Activation extraction and residual-stream steering for oracle training.

Subject model: the model whose activations we want to interpret.
Oracle model:  the model trained to explain those activations.

The oracle receives subject activations injected at its residual stream
(layer 1, by default) at the K prefix placeholder positions.  Injection
rule (from the paper):

    oracle_resid[b, pos, :] = normalize(subject_vec[k]) * ‖oracle_resid[b, pos, :]‖ * coeff
"""

import contextlib
from collections.abc import Generator
from typing import Any

import torch
from jaxtyping import Float
from loguru import logger
from torch import nn
from transformers import PreTrainedModel

# ---------------------------------------------------------------------------
# Layer utilities
# ---------------------------------------------------------------------------


def get_num_layers(model: PreTrainedModel) -> int:
    """Return the number of transformer layers in a HuggingFace model."""
    cfg = model.config
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    raise ValueError(f"Cannot determine layer count from config: {cfg}")


def layer_percent_to_layer(model: PreTrainedModel, percent: int) -> int:
    """Convert a layer percentage (0–100) to a 0-indexed layer number."""
    return int(get_num_layers(model) * (percent / 100))


def get_residual_submodule(model: PreTrainedModel, layer: int) -> nn.Module:
    """Return the nn.Module for a specific transformer layer (Qwen3/Llama style)."""
    # Unwrap PEFT model if needed
    inner = model
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        inner = model.base_model.model

    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        return inner.model.layers[layer]
    if hasattr(inner, "transformer") and hasattr(inner.transformer, "h"):
        return inner.transformer.h[layer]
    raise ValueError(f"Cannot find layers for model type: {type(model)}")


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


class _EarlyStop(Exception):
    pass


def collect_activations(
    model: PreTrainedModel,
    layer: int,
    inputs: dict[str, torch.Tensor],
) -> Float[torch.Tensor, "batch seq hidden"]:
    """Run subject model forward pass and return hidden states after `layer`.

    Uses an early-exit hook so computation stops right after the target layer.
    """
    captured: list[torch.Tensor] = []

    def _hook(module: nn.Module, args: Any, output: Any) -> None:  # noqa: ANN401
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach().float())
        raise _EarlyStop()

    submodule = get_residual_submodule(model, layer)
    handle = submodule.register_forward_hook(_hook)
    try:
        with torch.inference_mode():
            model(**inputs)
    except _EarlyStop:
        pass
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError(f"Activation hook did not fire for layer {layer}")
    return captured[0]


def collect_activations_batch(
    model: PreTrainedModel,
    layers: list[int],
    inputs: dict[str, torch.Tensor],
) -> dict[int, Float[torch.Tensor, "batch seq hidden"]]:
    """Collect activations from multiple layers in a single forward pass.

    Returns a dict mapping layer index → hidden states [B, L, D].
    """
    remaining = set(layers)
    captured: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHook] = []

    max_layer = max(layers)

    def make_hook(layer_idx: int):  # noqa: ANN202
        def _hook(module: nn.Module, args: Any, output: Any) -> None:  # noqa: ANN401
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach().float()
            remaining.discard(layer_idx)
            if layer_idx == max_layer:
                raise _EarlyStop()

        return _hook

    for layer_idx in layers:
        submodule = get_residual_submodule(model, layer_idx)
        handles.append(submodule.register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.inference_mode():
            model(**inputs)
    except _EarlyStop:
        pass
    finally:
        for h in handles:
            h.remove()

    if remaining:
        raise RuntimeError(f"Activation hooks did not fire for layers: {remaining}")
    return captured


# ---------------------------------------------------------------------------
# Activation steering (oracle side)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def steered_forward(
    model: PreTrainedModel,
    layer: int,
    vectors: list[Float[torch.Tensor, "k hidden"] | None],
    positions: list[list[int]],
    coefficient: float = 1.0,
) -> Generator[None]:
    """Context manager: inject subject activations into oracle at `layer`.

    `vectors[b]` is a [K_b, D] tensor of subject hidden states to inject
    into batch element b at oracle token positions `positions[b]`.

    Injection rule:
        resid[b, pos, :] = normalize(vec) * ‖resid[b, pos, :]‖ * coefficient
    """

    def _hook(module: nn.Module, args: Any, output: Any) -> Any:  # noqa: ANN401
        hidden = output[0] if isinstance(output, tuple) else output
        B = hidden.shape[0]
        for b in range(B):
            vecs = vectors[b]
            if vecs is None or len(positions[b]) == 0:
                continue
            vecs_f = vecs.float().to(hidden.device)
            norms = vecs_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            unit_vecs = vecs_f / norms  # [K, D]
            for k, pos in enumerate(positions[b]):
                resid_norm = hidden[b, pos, :].float().norm()
                hidden[b, pos, :] = (unit_vecs[k] * resid_norm * coefficient).to(hidden.dtype)

        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    submodule = get_residual_submodule(model, layer)
    handle = submodule.register_forward_hook(_hook)
    logger.debug(f"Registered steering hook at layer {layer}")
    try:
        yield
    finally:
        handle.remove()
