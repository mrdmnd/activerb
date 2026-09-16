"""Feature extraction for confusion-probe baseline experiments."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

MAX_BLOCK_LEN = 400

# Hand-coded hedge / uncertainty phrases (lowercased matching).
HEDGE_PHRASES: tuple[str, ...] = (
    "not sure",
    "unsure",
    "i don't know",
    "i do not know",
    "unable to",
    "cannot",
    "can't",
    "cannot find",
    "no access",
    "don't have",
    "do not have",
    "limitation",
    "caveat",
    "note that",
    "however",
    "unfortunately",
    "i'm not",
    "i am not",
    "might be",
    "maybe",
    "perhaps",
    "could be",
    "i'm sorry",
    "i am sorry",
    "sorry",
    "apolog",
    "unclear",
    "not certain",
    "not confident",
    "hard to say",
    "difficult to",
    "without more",
    "need more context",
    "don't see",
    "do not see",
    "not available",
    "not found",
    "hedge",
    "qualify",
    "to be clear",
    "for what it's worth",
    "as far as i know",
    "i believe",
    "i think",
    "it seems",
    "appears to",
)

LAYER_PERCENTS = (25, 50, 75)
LAYERS = tuple(int(36 * p / 100) for p in LAYER_PERCENTS)  # 9, 18, 27


def serialize_block(block: dict[str, Any]) -> str:
    btype = block.get("type", "")
    if btype == "text":
        return block.get("text", "").strip()[:MAX_BLOCK_LEN]
    if btype == "thinking":
        return f"<thinking>{block.get('text', '').strip()[:MAX_BLOCK_LEN]}</thinking>"
    if btype == "tool_call":
        args = str(block.get("arguments", ""))[:200]
        return f"[call {block.get('toolName', '')}({args})]"
    if btype == "tool_result":
        result = json.dumps(block.get("result", {}))[:MAX_BLOCK_LEN]
        return f"[result {block.get('toolName', '')} → {result}]"
    return ""


def serialize_message(msg: dict[str, Any]) -> str:
    role_label = "User" if msg["role"] == "user" else "Assistant"
    parts = [serialize_block(b) for b in msg.get("content", []) if isinstance(b, dict)]
    body = "\n".join(p for p in parts if p)
    return f"{role_label}: {body}\n\n"


def build_context(example: dict[str, Any]) -> tuple[str, str, dict[str, Any] | None]:
    """Return (prefix_text, target_text, target_message)."""
    messages = example["messages"]
    target_id = example["target_agent_message_id"]
    prefix_msgs: list[dict[str, Any]] = []
    target_msg: dict[str, Any] | None = None
    for msg in messages:
        if msg["id"] == target_id:
            target_msg = msg
            break
        prefix_msgs.append(msg)
    if target_msg is None:
        for msg in reversed(messages):
            if msg["role"] == "agent":
                target_msg = msg
                break
        prefix_msgs = [m for m in messages if m is not target_msg]
    prefix_text = "".join(serialize_message(m) for m in prefix_msgs)
    target_text = serialize_message(target_msg) if target_msg else ""
    return prefix_text, target_text, target_msg


def count_hedge_phrases(text: str) -> dict[str, int]:
    lower = text.lower()
    return {phrase: lower.count(phrase) for phrase in HEDGE_PHRASES}


def structural_vector(meta: dict[str, Any]) -> list[float]:
    return [
        float(meta["n_messages"]),
        float(meta["n_tool_calls"]),
        float(meta["n_thinking_blocks"]),
        float(meta["has_thinking_in_target"]),
        float(meta["target_position_idx"]),
        float(meta["target_position_pct"]),
        float(meta["target_text_chars"]),
        float(meta["target_text_tokens"]),
        float(meta["prefix_chars"]),
        float(meta["prefix_tokens"]),
    ]


STRUCTURAL_FEATURE_NAMES: tuple[str, ...] = (
    "n_messages",
    "n_tool_calls",
    "n_thinking_blocks",
    "has_thinking_in_target",
    "target_position_idx",
    "target_position_pct",
    "target_text_chars",
    "target_text_tokens",
    "prefix_chars",
    "prefix_tokens",
)


def metadata_from_example(
    example: dict[str, Any],
    *,
    target_token_count: int | None = None,
) -> dict[str, Any]:
    prefix_text, target_text, target_msg = build_context(example)
    messages = example["messages"]
    target_id = example["target_agent_message_id"]

    n_tool_calls = 0
    n_thinking_blocks = 0
    target_position_idx = 0
    for i, msg in enumerate(messages):
        for block in msg.get("content", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_call":
                n_tool_calls += 1
            if block.get("type") == "thinking":
                n_thinking_blocks += 1
        if msg["id"] == target_id:
            target_position_idx = i

    has_thinking_in_target = 0
    if target_msg is not None:
        has_thinking_in_target = int(
            any(isinstance(b, dict) and b.get("type") == "thinking" for b in target_msg.get("content", []))
        )

    n_messages = len(messages)
    target_position_pct = target_position_idx / max(n_messages - 1, 1)
    full_text = prefix_text + target_text

    return {
        "thread_id": example["thread_id"],
        "label": 1 if example["label"] == "confused" else 0,
        "confusion_type": example.get("confusion_type"),
        "target_text": target_text,
        "prefix_text": prefix_text,
        "full_text": full_text,
        "n_messages": n_messages,
        "n_tool_calls": n_tool_calls,
        "n_thinking_blocks": n_thinking_blocks,
        "has_thinking_in_target": has_thinking_in_target,
        "target_position_idx": target_position_idx,
        "target_position_pct": target_position_pct,
        "target_text_chars": len(target_text),
        "target_text_tokens": target_token_count if target_token_count is not None else 0,
        "prefix_chars": len(prefix_text),
        "prefix_tokens": 0,
        "hedge_counts": count_hedge_phrases(target_text),
    }


@dataclass
class FeatureRecord:
    thread_id: str
    label: int
    confusion_type: str | None
    meta: dict[str, Any]
    activations: dict[int, Any] = field(default_factory=dict)  # layer_idx -> np.ndarray

    @property
    def hedge_vector(self) -> list[float]:
        counts = self.meta["hedge_counts"]
        return [float(counts[p]) for p in HEDGE_PHRASES]


def load_dataset_jsonl(path: str) -> list[dict[str, Any]]:
    with pathlib.Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_metadata_records(examples: list[dict[str, Any]]) -> list[FeatureRecord]:
    return [
        FeatureRecord(
            thread_id=ex["thread_id"],
            label=1 if ex["label"] == "confused" else 0,
            confusion_type=ex.get("confusion_type"),
            meta=metadata_from_example(ex),
        )
        for ex in examples
    ]
