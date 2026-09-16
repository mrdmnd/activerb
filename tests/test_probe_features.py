"""Tests for confusion-probe feature extraction."""

from activerb.probe_features import (
    HEDGE_PHRASES,
    build_context,
    count_hedge_phrases,
    metadata_from_example,
    structural_vector,
)


def test_hedge_counting() -> None:
    counts = count_hedge_phrases("I don't have access due to a limitation.")
    assert counts["don't have"] >= 1
    assert counts["limitation"] >= 1


def test_metadata_from_minimal_example() -> None:
    example = {
        "thread_id": "t1",
        "label": "confused",
        "confusion_type": "agent_caveat",
        "target_agent_message_id": "m2",
        "messages": [
            {"id": "m1", "role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "id": "m2",
                "role": "agent",
                "content": [
                    {"type": "thinking", "text": "hmm"},
                    {"type": "text", "text": "maybe this works"},
                ],
            },
        ],
    }
    meta = metadata_from_example(example, target_token_count=10)
    assert meta["label"] == 1
    assert meta["has_thinking_in_target"] == 1
    assert meta["n_messages"] == len(example["messages"])
    prefix, target, _ = build_context(example)
    assert "User:" in prefix
    assert "Assistant:" in target
    vec = structural_vector(meta)
    assert len(vec) == len(structural_vector(meta))
    assert HEDGE_PHRASES  # non-empty lexicon
