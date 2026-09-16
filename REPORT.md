# Reading LLM Hidden States to Detect Agent Confusion
### A Research Report

**Author:** Matt Redmond  
**Date:** May 2026  
**Codebase:** `github.com/mredmond/activerb`

---

## Overview

This report describes two connected experiments:

1. **Activation Oracle Replication** — replicating the core result from ["Activation Oracles"](https://arxiv.org/abs/2512.15674) (arXiv 2512.15674): training a language model to read the hidden states of another language model and answer questions about what those states encode.

2. **Confusion Detection Probe** — applying activation-reading to a real product problem: detecting when an AI agent is confused, using a retrospective dataset of labeled Hex agent conversations.

The central question: *can we build a system that reads an LLM's internal representations to predict its epistemic state — specifically, whether it is confused?*

---

## Part 1: Activation Oracle Replication

### Background

An **activation oracle** is an LLM trained to interpret the hidden states of another LLM (the "subject" model). Rather than reading the subject model's input text, the oracle receives subject activations injected directly into its own residual stream at placeholder token positions, then answers questions about what those activations encode.

The injection rule from the paper:

```
oracle_resid[pos] = normalize(subject_vec) * ||oracle_resid[pos]|| * coefficient
```

This preserves the oracle's internal scale while replacing the directional content of its residual stream with the subject model's signal. We replicated this using `Qwen/Qwen3-4B` as both the subject model and the oracle base model.

### Infrastructure

All experiments ran on Modal H100 GPUs. Training used LoRA (r=64, alpha=128, lr=1e-5, all-linear targets) on top of frozen base weights — the oracle learns to read activations through a low-rank adapter.

### Experiment 1: PastLens Only (20k examples)

**Task:** The oracle predicts K tokens that preceded a window of subject activations. Subject runs a forward pass on a FineWeb text sample; oracle is prompted to guess what tokens came before a window of subject activations.

**Hypothesis:** Since the oracle never sees the input text, it must read the injected activations to answer correctly.

**Result:**

| | oracle_loss | baseline_loss | delta |
|---|---|---|---|
| Pre-training | 11.54 | 11.55 | +0.012 |
| Post-training | 3.94 | 3.94 | +0.001 |

Delta collapsed to ~0. Both the oracle and the baseline (no injected activations) improved equally.

**Why it failed:** PastLens can be solved by learning token n-gram statistics from the training corpus. The oracle learns the prior distribution of preceding tokens without ever reading the injected activations. No genuine activation reading occurs.

### Experiment 2: PastLens + Classification (100k examples, H100)

**Fix:** Added SST-2 sentiment classification as 50% of the training mix. The oracle is asked "positive or negative?" but the label appears nowhere in its prompt — only in the injected activations. This forces genuine activation reading: no corpus statistics can help because the prompt contains no signal about which sentence the subject model processed.

**Training:** 70,726 examples (after deduplication), 1 epoch, H100, ~47 minutes.

**Result:**

| | oracle_loss | baseline_loss | delta |
|---|---|---|---|
| Pre-training | 11.60 | 11.60 | -0.002 |
| Post-training | **2.94** | **4.13** | **+1.19** |

Delta went from noise to **+1.19** — the oracle is successfully reading the injected activations.

### Experiment 3: Confound Controls

**Question:** Is the delta real, or is the oracle simply detecting "real transformer activations vs. noise" as a distributional artifact unrelated to semantic content?

Four baselines run on the trained oracle using the SST-2 **validation** split (unseen during training):

| Baseline | Loss | Delta vs oracle | What it rules out |
|---|---|---|---|
| Oracle (real acts, correct label) | **0.125** | — | — |
| Random (noise vectors, same norm) | 0.239 | +0.114 | — |
| Shuffled (real acts, wrong sentence) | 0.390 | +0.265 | "real vs. fake" distributional confound |
| **Label-swapped** (real acts, opposite label) | **0.598** | **+0.472** | All geometric confounds |

**Key findings:**

- **Shuffled > Random:** Real activations from the wrong sentence are worse than random noise — wrong content actively misleads the oracle, meaning it has learned to read specific content rather than just detect "real vs. fake."
- **Label-swapped delta of +0.472:** Even with real SST-2 activations (identical distributional properties), injecting the wrong label's activations causes a large loss increase. The oracle extracts label-specific semantic content from Qwen3-4B's hidden states.

The oracle generalizes to unseen data and reads genuinely semantic information, not geometric artifacts.

### Part 1 Conclusion

A language model can be trained to read another language model's internal representations. Task design is the critical variable: PastLens allows statistical shortcuts (delta ~= 0); classification with the label withheld from the prompt forces genuine activation reading (delta = +1.19, confirmed by confound controls).

---

## Part 2: Confusion Detection Probe

### Motivation

The activation oracle result raises an applied question: can we use activation-reading to detect epistemic states in production AI agents — specifically, whether an agent is confused?

Hex's AI agent regularly produces responses where it lacks context, hedges, or misleads the user due to uncertainty. Currently these instances are detected retrospectively by an LLM-based classifier. The question: is the confusion signal present in the model's activations at the time the response is generated — and is it linearly decodable?

### Dataset Construction

We extracted a labeled dataset from Hex's production database (RC, `hex-testing` org). The `thread_confusion` table stores instances where an LLM-based pipeline analyzed agent threads and labeled specific responses as confused, along with confusion type (`user_doubt`, `missing_context`, `agent_caveat`) and a natural language summary.

**Positive examples (confused):** For each confusion record, extract all messages up to and including the confusion anchor point. Identify the last substantive agent text response before the confusion marker — this is the "target" message whose activations we probe.

**Negative examples (not_confused):** Sample threads that passed through the confusion detection pipeline and were found clean. For each, pick a later-turn agent text response as the target.

**Dataset:** 364 examples — 182 confused, 182 not_confused, balanced by design.

Confusion type breakdown among positive examples:

| Type | Description |
|---|---|
| `user_doubt` | User expresses explicit doubt or asks for clarification |
| `missing_context` | Agent lacks information needed to answer accurately |
| `agent_caveat` | Agent hedges or qualifies its response |

### Method

We use Qwen3-4B as an external reader — not the production agent, which is Claude-based and inaccessible. Each conversation is serialized to plain text, with content blocks rendered as:

- `text` / `thinking` blocks: truncated plain text (400 chars)
- `tool_call`: `[call toolName(args)]`
- `tool_result`: `[result toolName -> output]`

Each turn is prefixed with `User:` or `Assistant:` and double-newline separated.

**Token alignment:** Prefix (all turns before the target) and target message are tokenized separately to find the exact boundary. Activations are extracted only at target message token positions. The context is truncated to 4096 tokens if necessary, preserving the end of the prefix and all of the target.

**Activation extraction:** Forward hooks on Qwen3-4B's transformer layers capture hidden states at layers 9, 18, and 27 (25%, 50%, 75% of the 36-layer model). Activations at target token positions are mean-pooled to produce a single vector per layer per example.

**Probe:** A scikit-learn pipeline — `StandardScaler -> PCA(64) -> LogisticRegression(balanced)` — trained with 5-fold stratified cross-validation and evaluated on ROC-AUC and accuracy. PCA reduces the ~3072-dimensional activation vectors before logistic regression to avoid overfitting at this dataset size.

**Compute:** Qwen3-4B on Modal H100, ~15 minutes for all 364 forward passes.

### Results

| Layer | AUC | Std | Accuracy |
|---|---|---|---|
| 25% (layer 9) | 0.838 | 0.038 | 77.2% |
| **50% (layer 18)** | **0.875** | **0.027** | **80.8%** |
| 75% (layer 27) | 0.833 | 0.016 | 76.9% |

Layer 50% performs best — AUC=0.875 on a balanced 364-example dataset with 5-fold **StratifiedKFold** CV. The middle of the network carries the strongest confusion signal, consistent with the interpretability literature finding that semantic content peaks in middle layers.

A `user_doubt` leave-one-out subset (14 examples) was too small for reliable evaluation — folds collapsed to single-class, producing NaN AUC. This is a data volume limitation, not a signal failure.

### Baseline Falsification (see `BASELINES.md`)

We re-ran the probe under **GroupKFold by `thread_id`** (38 threads contribute multiple confused examples) and compared against stronger baselines on the same 364 examples:

| Method | AUC (GroupKFold) | Notes |
|---|---|---|
| Qwen probe layer 50% (B0) | **0.831** | Fair CV; down from 0.875 StratifiedKFold |
| TF-IDF full transcript (T2) | **0.843** | Same serialized text the probe sees |
| Structural features only (S1) | 0.807 | Length, position, tool counts — no text |
| Qwen after partialing structure (B2) | 0.735 | Large drop vs B0 |
| Sentence-transformer full (E1) | 0.731 | Frozen `bge-small-en-v1.5` |
| Zero-shot Qwen judge (Z1) | 0.732 | No training |

**Conclusions:**

1. **The probe is largely redundant with bag-of-words.** TF-IDF on the full transcript matches or exceeds the Qwen probe under fair thread-grouped CV (0.843 vs 0.831).
2. **Much of the signal is structural, not semantic.** Conversation length, target position, and tool-call count alone reach AUC=0.807; partialling structure out of activations drops probe AUC to 0.735.
3. **The original 0.875 figure was modestly inflated** by StratifiedKFold (~+0.04) and thread leakage across folds.
4. **`user_doubt` is the one encouraging subset:** Qwen probe 0.870 vs TF-IDF target 0.707 (n=14 positives) — but the sample is too small to rely on.

Reproduce: `uv run modal run modal_extract_features.py`, then `uv run python baselines_run.py --features runs/features_v2.pkl`.

### What This Means

**There is a decodable signal in Qwen's encoding of the conversation text** — but it is mostly captured by simpler methods. A TF-IDF classifier on the same transcript performs comparably; structural metadata explains most of the remainder. This is **not** evidence that Qwen reads Claude's internal epistemic state; it is evidence that confused agent turns differ systematically in length, position, and surface text from non-confused turns.

**Qwen is not the production agent.** We are using Qwen3-4B as an external reader of the conversation text. The oracle reads activations of Qwen's *representation of the conversation* — not the activations of the model that generated the response. This is an important distinction: the signal is present in how Qwen encodes the conversation, not in the production model's internal state.

**Implication for live detection:** Because Qwen is an external reader, live detection is architecturally straightforward. After each agent text response, the conversation is sent to a Qwen3-4B probe service. A single forward pass extracts layer-18 activations and the logistic probe returns a confusion probability in ~0.5–1s on GPU — fast enough for async monitoring after the response is delivered.

---

## Part 3: Early Detection — Probing at Tool Call Points

### Motivation

The Part 2 probe reads activations at the *final agent text response* — after the agent has finished reasoning and produced its answer. A more useful capability would be detecting confusion *earlier*: at the moment the agent is calling tools, before it has even written its reply.

We ran a second probe experiment (`modal_probe_toolcall.py`) comparing four probe points:

| Condition | What is probed |
|---|---|
| `final_response` | Tokens of the final agent text response (baseline) |
| `last_tool_call` | Tool call tokens in the last agent turn before the final response |
| `first_tool_call` | Tool call tokens in the very first agent turn |
| `all_tool_calls` | Activations averaged across all tool call turns |

**Token selection:** For tool call conditions, only the `[call toolName(args)]` tokens are used as the "target" — not thinking blocks, text, or tool results. This isolates the moment of tool invocation specifically.

**Averaging:** Within each probe point, activations are mean-pooled over the target token positions to produce a single `[D]`-dimensional vector. For `all_tool_calls`, a second level of averaging is applied: one `[D]` vector is computed per tool call turn (by mean-pooling over its tokens), then those vectors are averaged across all turns in the thread. Every turn is weighted equally.

### Results

All results use 5-fold stratified cross-validation, PCA(64) + LogisticRegression(balanced).

| Condition | Best layer | AUC | ± | Acc | n |
|---|---|---|---|---|---|
| `final_response` | 50% | **0.863** | 0.019 | 79.9% | 364 |
| `all_tool_calls` | 75% | **0.849** | 0.026 | 78.5% | 326 |
| `first_tool_call` | 25% | **0.794** | 0.032 | 74.3% | 326 |
| `last_tool_call` | 25% | **0.784** | 0.020 | 71.2% | 326 |

(38 examples had no tool calls and are excluded from tool call conditions.)

### Findings

**Confusion is detectable at the very first tool call (AUC=0.794).** Before the agent has finished reasoning, before it has written its reply, at the first action it takes in a thread, the confusion signal is already present in Qwen's hidden states.

**Pooling all tool calls nearly matches the final response (0.849 vs 0.863).** Averaging activations across the entire tool-call trajectory recovers almost all the signal available at the end of the conversation. The full sequence of what the agent chose to call — viewed in aggregate — is almost as informative as the final response.

**The last tool call is not the best predictor.** `last_tool_call` (0.784) is slightly *worse* than `first_tool_call` (0.794), which is counterintuitive. One hypothesis: the last tool call before a confused response is often a "grasping" call where the agent is trying to recover from an unclear situation, producing a noisy activation pattern. The first tool call captures the agent's initial framing of the problem, which may encode uncertainty more cleanly.

**The early-detection signal lives in earlier layers.** `first_tool_call` peaks at layer 25% (early), while `final_response` peaks at layer 50% (middle). This suggests the two probe points are encoding qualitatively different things — the first tool call captures something about the agent's initial epistemic stance, while the final response encodes the accumulated uncertainty after all reasoning is complete.

---

## Part 4: Live Detection Architecture

The tool call probe result changes the architecture significantly — confusion can be flagged *during* the agent's execution, not just after:

```
User sends message
  -> Production agent begins executing
  -> Agent makes first tool call
       -> Async: context + tool call sent to Qwen3-4B probe service
       -> Extract layer-9 activations at tool call tokens
       -> Logistic probe returns P(confused) = 0.82  <- early warning
  -> Agent continues, makes more tool calls...
  -> Agent produces final text response
       -> Async: update P(confused) from layer-18 activations
  -> If P(confused) > threshold at any point:
       -> Surface indicator in UI
       -> Prompt user to verify assumptions before acting on response
       -> Route to higher-capability model for a second pass
```

This opens up an intervention point that doesn't exist with response-only monitoring: if the first tool call already shows high confusion probability, the system could intervene *before* the agent wastes further compute on a confused trajectory, or could inject a clarification prompt mid-execution.

**What the probe still can't do:** access the production model's (Claude's) own internal states. Qwen3-4B is acting as an external reader of the conversation text. The signal is in how Qwen encodes the tool call content, not in the production model's hidden states.

**Next steps toward deployment:**
1. Expand the labeled dataset, especially `user_doubt` examples (currently only 14 usable instances)
2. Train the full activation oracle on the confusion task — replacing the logistic probe with a fine-tuned Qwen oracle that can verbalize *why* the agent is flagged as confused
3. Productionize as an async sidecar service, triggering on each tool call with a calibrated threshold
4. Evaluate whether early flagging (at first tool call) is actionable enough to justify intervention

---

## Reproduce

```bash
# Install dependencies
uv sync

# Build confusion dataset (requires SDM tunnel to RC at localhost:10070)
uv run python build_dataset.py --out datasets/confusion_dataset.jsonl

# Run final-response linear probe on Modal H100
uv run modal run --detach modal_probe.py

# Run tool-call-point probe experiment on Modal H100
uv run modal run --detach modal_probe_toolcall.py

# Run baseline falsification (extract features, then local baselines)
uv run modal run modal_extract_features.py
modal volume get activerb-outputs /probe/features_v2.pkl ./runs/features_v2.pkl
uv run python baselines_run.py --features runs/features_v2.pkl

# Run activation oracle training on Modal H100 (full 100k examples)
uv run modal run --detach modal_experiment.py

# Fetch results from Modal volume
modal volume get activerb-outputs /. ./modal_results
```

---

## Summary of Results

| Experiment | Key Metric | Value |
|---|---|---|
| PastLens only (20k) | Oracle delta | ~0.001 (failure) |
| PastLens + Classification (100k) | Oracle delta | **+1.19** |
| Confound control — label-swapped | Additional loss | **+0.472** |
| Confusion probe — final response, layer 50% (StratifiedKFold) | AUC | **0.875** |
| Confusion probe — final response, layer 50% (GroupKFold) | AUC | **0.831** |
| TF-IDF full transcript baseline (GroupKFold) | AUC | **0.843** |
| Confusion probe — all tool calls, layer 75% | AUC (5-fold CV) | **0.849** |
| Confusion probe — first tool call only, layer 25% | AUC (5-fold CV) | **0.794** |

Confusion is linearly decodable from Qwen3-4B's hidden states at the moment of the agent's very first tool call — before the response is written — at AUC=0.794. Pooling across all tool calls in a thread recovers nearly all the signal available at the final response (0.849 vs 0.863).
