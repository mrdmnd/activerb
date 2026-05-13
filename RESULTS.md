# Activation Oracle Replication — Results Summary

## What We Did

We replicated the core experiment from [**"Activation Oracles"** (arXiv 2512.15674)](https://arxiv.org/abs/2512.15674) using `Qwen/Qwen3-4B` as both the subject model and oracle base model.

### The Core Idea

An **activation oracle** is an LLM trained to interpret the hidden states of another LLM (the "subject" model). The oracle receives subject activations injected directly into its own residual stream at placeholder token positions, then is trained to answer questions about what those activations encode — without ever seeing the subject model's input text.

The injection rule:
```
oracle_resid[pos] = normalize(subject_vec) * ‖oracle_resid[pos]‖ * coefficient
```

This preserves the oracle's internal scale while replacing direction with the subject model's signal.

---

## What We Built

### Core Pipeline (`activerb/`)

| File | Purpose |
|---|---|
| `hooks.py` | Activation extraction (early-exit hooks) and residual-stream steering |
| `data.py` | Training example construction; placeholder token ` ?` injection prefix |
| `datasets/past_lens.py` | Self-supervised token prediction task (PastLens) |
| `datasets/classification.py` | SST-2 sentiment classification task |
| `train.py` | LoRA fine-tuning loop (r=64, α=128, lr=1e-5) |
| `eval.py` | Four-baseline evaluation suite |

### Experiment Scripts

| File | Purpose |
|---|---|
| `modal_experiment.py` | Full training run on Modal H100 |
| `modal_eval_only.py` | Enhanced eval against a saved checkpoint |
| `run_experiment.py` | Local training runner (MPS/CPU, smaller scale) |

---

## Experiments

### Experiment 1 — PastLens Only (20k examples)

**Task**: Oracle predicts K tokens that preceded/followed a window of subject activations.

**Result**: Delta collapsed to ~zero after training.

| | oracle_loss | baseline_loss | delta |
|---|---|---|---|
| Pre-training | 11.54 | 11.55 | +0.012 |
| Post-training | 3.94 | 3.94 | +0.001 |

**Why it failed**: PastLens can be solved by learning token n-gram statistics from the training corpus. The oracle doesn't need to read the activations at all — it just learns the prior distribution of preceding/following tokens. No genuine activation reading occurs.

---

### Experiment 2 — PastLens + Classification (100k examples, H100)

**Fix**: Added SST-2 sentiment classification as 50% of training data. The oracle is asked "positive or negative?" but the label appears **nowhere** in its prompt — the only signal is the injected activations. This forces genuine activation reading.

**Training**: 70,726 examples (after deduplication/filtering), 1 epoch, H100, ~47 minutes.

| | oracle_loss | baseline_loss | delta |
|---|---|---|---|
| Pre-training | 11.60 | 11.60 | −0.002 |
| Post-training | **2.94** | **4.13** | **+1.19** |

The delta went from noise to **+1.19** — a large, meaningful signal.

---

### Experiment 3 — Enhanced Evaluation (Confound Controls)

**Question**: Is the delta real, or is the oracle just detecting "real transformer activations vs. random noise" (a distributional confound)?

We ran four baselines on the trained oracle using the SST-2 **validation** split (unseen during training):

| Baseline | Loss | Δ vs oracle | What it rules out |
|---|---|---|---|
| Oracle (real, correct label) | **0.125** | — | — |
| Random (noise, same norm) | 0.239 | +0.114 | — |
| Shuffled (real acts, different sentence) | 0.390 | +0.265 | "real vs. fake" distributional confound |
| **Label-swapped** (real acts, opposite label) | **0.598** | **+0.472** | All geometry confounds |

**Key findings**:
- Shuffled > Random: real activations from the *wrong* sentence are actually *worse* than random noise — the oracle has learned to read specific content, and wrong content actively misleads it.
- Label-swapped delta of **+0.47**: even with real SST-2 activations with identical distributional properties, injecting the *wrong label's* activations causes a large loss increase. The oracle is extracting **label-specific semantic content** from Qwen3-4B's hidden states.

---

## Conclusion

We successfully replicated the paper's central claim: a language model can be trained to read another language model's internal representations. The key lessons:

1. **Task design matters enormously.** PastLens alone produces a delta of ~0 because statistical shortcuts exist. Classification forces genuine activation reading by withholding the label from the prompt entirely.

2. **The result is robust to confound controls.** The label-swapped baseline — real transformer activations, same distribution, wrong label — still shows a +0.47 delta. The oracle is reading semantically meaningful content, not just detecting activation geometry.

3. **The oracle generalizes to unseen data.** All enhanced eval was run on the SST-2 validation split, not the training split.

The trained LoRA adapter is saved in the `activerb-outputs` Modal volume at `20260513_021015/checkpoints/final`.

---

## Reproduce

```bash
# Install dependencies
uv sync

# Run locally (small scale, wikitext-103, MPS/CPU)
uv run python run_experiment.py

# Run on Modal H100 (full scale, FineWeb + SST-2, detached)
uv run modal run --detach modal_experiment.py

# Run enhanced eval against saved checkpoint
uv run modal run --detach modal_eval_only.py

# Fetch results
modal volume get activerb-outputs /. ./modal_results
```
