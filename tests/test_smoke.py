import pytest
import torch

import activerb


def test_import() -> None:
    assert activerb.__version__ == "0.1.0"


# ---------------------------------------------------------------------------
# hooks.py
# ---------------------------------------------------------------------------


def test_get_num_layers() -> None:
    from transformers import AutoConfig, AutoModelForCausalLM

    from activerb.hooks import get_num_layers

    # Use a tiny GPT-2 config so we don't need a GPU or large download
    cfg = AutoConfig.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_config(cfg)
    n = get_num_layers(model)
    assert n == cfg.n_layer


def test_layer_percent_to_layer() -> None:
    from transformers import AutoConfig, AutoModelForCausalLM

    from activerb.hooks import layer_percent_to_layer

    cfg = AutoConfig.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_config(cfg)
    layer = layer_percent_to_layer(model, 50)
    assert 0 <= layer < cfg.n_layer


def test_collect_activations_tiny() -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from activerb.hooks import collect_activations

    cfg = AutoConfig.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer("hello world", return_tensors="pt")
    acts = collect_activations(model, layer=0, inputs=dict(inputs))

    assert acts.ndim == 3  # [batch, seq, hidden]
    assert acts.shape[0] == 1
    assert acts.shape[-1] == cfg.n_embd


def test_collect_activations_batch_tiny() -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from activerb.hooks import collect_activations_batch

    cfg = AutoConfig.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(["hello world", "foo bar baz"], return_tensors="pt", padding=True)
    layers = [0, 1]
    acts = collect_activations_batch(model, layers, dict(inputs))

    assert set(acts.keys()) == {0, 1}
    for layer_idx in layers:
        a = acts[layer_idx]
        assert a.ndim == 3
        assert a.shape[0] == 2
        assert a.shape[-1] == cfg.n_embd


# ---------------------------------------------------------------------------
# data.py
# ---------------------------------------------------------------------------


def test_get_introspection_prefix() -> None:
    from activerb.data import SPECIAL_TOKEN, get_introspection_prefix

    prefix = get_introspection_prefix(layer=5, num_positions=3)
    assert "Layer: 5" in prefix
    assert prefix.count(SPECIAL_TOKEN) == 3


QWEN3_TOKENIZER = "Qwen/Qwen3-4B"


@pytest.fixture(scope="module")
def qwen3_tokenizer():
    """Qwen3 tokenizer — required for the ' ?' special token to work correctly."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER)
    tok.padding_side = "left"
    return tok


def test_make_training_example_qwen3(qwen3_tokenizer) -> None:
    from activerb.data import make_training_example

    k, d = 3, 2048
    acts = torch.randn(k, d)
    example = make_training_example(
        prompt="What came before?",
        target="foo bar baz",
        layer=2,
        acts=acts,
        tokenizer=qwen3_tokenizer,
    )

    assert len(example.positions) == k
    assert len(example.input_ids) == len(example.labels)
    assert -100 in example.labels
    assert any(lb != -100 for lb in example.labels)


def test_batch_examples_qwen3(qwen3_tokenizer) -> None:
    from activerb.data import batch_examples, make_training_example

    device = torch.device("cpu")
    examples = []
    for i in range(3):
        k = i + 1
        acts = torch.randn(k, 2048)
        ex = make_training_example(
            prompt=f"Predict tokens (example {i})",
            target=f"target {i}",
            layer=1,
            acts=acts,
            tokenizer=qwen3_tokenizer,
        )
        examples.append(ex)

    batch = batch_examples(examples, qwen3_tokenizer, device)
    B = len(examples)
    assert batch.input_ids.shape[0] == B
    assert batch.labels.shape[0] == B
    assert batch.attention_mask.shape[0] == B
    assert len(batch.steering_vectors) == B
    assert len(batch.positions) == B


# ---------------------------------------------------------------------------
# steered_forward (integration: hook fires and modifies residuals)
# ---------------------------------------------------------------------------


def test_steered_forward_modifies_hidden_states() -> None:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from activerb.hooks import steered_forward

    cfg = AutoConfig.from_pretrained("sshleifer/tiny-gpt2")
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer("hello world foo", return_tensors="pt")
    inputs["input_ids"].shape[1]
    D = cfg.n_embd

    # Collect baseline logits
    with torch.inference_mode():
        baseline = model(**inputs).logits.clone()

    # Inject a random steering vector at position 0
    vec = torch.randn(1, D)
    vectors = [vec]
    positions = [[0]]

    with torch.inference_mode(), steered_forward(model, layer=0, vectors=vectors, positions=positions):
        steered = model(**inputs).logits.clone()

    # Steering should change the output
    assert not torch.allclose(baseline, steered, atol=1e-4), "Steering had no effect"
