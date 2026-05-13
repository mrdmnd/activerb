"""Build the confusion detection dataset from the RC dogfood database.

For each example we extract:
  - All messages in the thread up to (and including) the agent response
    that preceded the confusion marker
  - A label: "confused" or "not_confused"
  - Metadata: confusion_type, severity, summary

The resulting JSONL is the raw material for training the activation verbalizer.

Usage:
    python build_dataset.py --out datasets/confusion_dataset.jsonl

Requirements:
    pip install psycopg2-binary
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass

import psycopg2
import psycopg2.extras

# ── DB connection (RC, read-only replica via SDM tunnel) ──────────────────────

RC_DSN = "host=127.0.0.1 port=10070 dbname=hexinc user=postgres password="
ORG_ID = "hex-testing"


def connect() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(RC_DSN)
    conn.autocommit = True
    return conn


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class Message:
    id: str
    role: str
    content: list  # raw JSONB list of content blocks
    created_date: str


@dataclass
class Example:
    thread_id: str
    label: str  # "confused" | "not_confused"
    confusion_type: str | None
    severity: str | None
    summary: str | None
    # All messages up to (and including) the target agent response
    messages: list[dict]
    # ID of the agent message whose activations are most relevant
    target_agent_message_id: str | None


# ── Helpers ───────────────────────────────────────────────────────────────────


def clean_content(content: list) -> list:
    """Strip image binary data, keep text/tool_call/tool_result structure."""
    cleaned = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            cleaned.append({"type": "text", "text": block.get("text", "")})
        elif btype == "thinking":
            # Include thinking blocks — they're the most interpretability-relevant
            cleaned.append({"type": "thinking", "text": block.get("text", "")})
        elif btype == "tool_call":
            cleaned.append({
                "type": "tool_call",
                "toolName": block.get("toolName", ""),
                "arguments": block.get("arguments", ""),
            })
        elif btype == "tool_result":
            result = block.get("result", {})
            # Strip image content, keep text result
            if isinstance(result, dict) and result.get("type") == "image-file":
                result = {"type": "image-file", "label": result.get("label", "[image]")}
            cleaned.append({
                "type": "tool_result",
                "toolName": block.get("toolName", ""),
                "result": result,
            })
    return cleaned


def get_thread_messages(cursor, thread_id: str, before_date: str | None = None) -> list[Message]:
    """Fetch all messages in a thread, optionally up to a cutoff date."""
    if before_date:
        cursor.execute(
            """
            SELECT id, role, content, "createdDate"::text
            FROM agent_chat_message
            WHERE "agentChatThreadId" = %s AND "createdDate" <= %s
            ORDER BY "createdDate"
            """,
            (thread_id, before_date),
        )
    else:
        cursor.execute(
            """
            SELECT id, role, content, "createdDate"::text
            FROM agent_chat_message
            WHERE "agentChatThreadId" = %s
            ORDER BY "createdDate"
            """,
            (thread_id,),
        )
    rows = cursor.fetchall()
    return [Message(id=r[0], role=r[1], content=r[2] or [], created_date=r[3]) for r in rows]


# ── Positive examples (confused) ──────────────────────────────────────────────


def build_confused_examples(conn) -> list[Example]:
    """
    For each ThreadConfusion in hex-testing, build a training example.

    The "target" is the last agent message before the confusion-flagged message.
    We include all messages up to and including that agent response.

    Priority order: user_doubt > missing_context > agent_caveat > other
    """
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cursor.execute(
        """
        SELECT DISTINCT ON (te."agentChatThreadId", tc."agentChatMessageId")
          tc.id as confusion_id,
          te."agentChatThreadId" as thread_id,
          tc."confusionType" as confusion_type,
          tc.severity,
          tc.summary,
          tc."agentChatMessageId" as confusion_msg_id,
          cm."createdDate"::text as confusion_msg_date,
          cm.role as confusion_msg_role
        FROM thread_confusion tc
        JOIN thread_extraction te ON tc."threadExtractionId" = te.id
        LEFT JOIN agent_chat_message cm ON cm.id = tc."agentChatMessageId"
        WHERE tc."orgId" = %s
          AND tc."agentChatMessageId" IS NOT NULL
        ORDER BY te."agentChatThreadId", tc."agentChatMessageId",
          CASE tc."confusionType"
            WHEN 'user_doubt' THEN 1
            WHEN 'missing_context' THEN 2
            WHEN 'agent_caveat' THEN 3
            ELSE 4
          END
        """,
        (ORG_ID,),
    )
    rows = cursor.fetchall()
    print(f"Found {len(rows)} confusion anchor points")

    examples = []
    for row in rows:
        thread_id = row["thread_id"]
        confusion_msg_date = row["confusion_msg_date"]

        if not confusion_msg_date:
            continue

        # Get all messages up to and including the confusion message
        messages = get_thread_messages(cursor, thread_id, before_date=confusion_msg_date)
        if len(messages) < 2:
            continue

        # Find the last agent message before the confusion user message
        target_agent_msg_id = None
        for msg in reversed(messages):
            if msg.role == "agent":
                # Skip pure tool-call/tool-result agent messages, find a text response
                has_text = any(b.get("type") in ("text", "thinking") for b in msg.content if isinstance(b, dict))
                if has_text:
                    target_agent_msg_id = msg.id
                    break

        if target_agent_msg_id is None:
            continue

        examples.append(Example(
            thread_id=thread_id,
            label="confused",
            confusion_type=row["confusion_type"],
            severity=row["severity"],
            summary=row["summary"],
            messages=[
                {"id": m.id, "role": m.role, "content": clean_content(m.content)}
                for m in messages
            ],
            target_agent_message_id=target_agent_msg_id,
        ))

    print(f"Built {len(examples)} confused examples")
    return examples


# ── Negative examples (not confused) ─────────────────────────────────────────


def build_not_confused_examples(conn, n: int, seed: int = 42) -> list[Example]:
    """
    Sample threads from hex-testing that have NO ThreadConfusion records at all.
    For each, pick a random agent response as the target.
    """
    rng = random.Random(seed)
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Find threads with no confusion records, that are ASK type and have enough messages
    cursor.execute(
        """
        SELECT act.id as thread_id
        FROM agent_chat_thread act
        WHERE act."orgId" = %s
          AND act.type = 'ASK'
          AND NOT EXISTS (
            SELECT 1 FROM thread_extraction te
            JOIN thread_confusion tc ON tc."threadExtractionId" = te.id
            WHERE te."agentChatThreadId" = act.id
          )
          AND EXISTS (
            SELECT 1 FROM thread_extraction te2
            WHERE te2."agentChatThreadId" = act.id
              AND te2."extractionType" = 'CONFUSION_DETECTION'
          )
        ORDER BY act."createdDate" DESC
        LIMIT 1000
        """,
        (ORG_ID,),
    )
    candidate_threads = [r[0] for r in cursor.fetchall()]
    print(f"Found {len(candidate_threads)} non-confused candidate threads")

    rng.shuffle(candidate_threads)
    examples = []

    for thread_id in candidate_threads:
        if len(examples) >= n:
            break

        messages = get_thread_messages(cursor, thread_id)
        if len(messages) < 4:
            continue

        # Find agent messages with text content
        agent_msgs = [
            m for m in messages
            if m.role == "agent"
            and any(b.get("type") in ("text", "thinking") for b in m.content if isinstance(b, dict))
        ]
        if not agent_msgs:
            continue

        # Pick one of the later agent messages (more context = more interesting)
        target = rng.choice(agent_msgs[len(agent_msgs) // 2:])

        # Include all messages up to and including target
        cutoff_date = target.created_date
        msgs_up_to_target = [m for m in messages if m.created_date <= cutoff_date]

        examples.append(Example(
            thread_id=thread_id,
            label="not_confused",
            confusion_type=None,
            severity=None,
            summary=None,
            messages=[
                {"id": m.id, "role": m.role, "content": clean_content(m.content)}
                for m in msgs_up_to_target
            ],
            target_agent_message_id=target.id,
        ))

    print(f"Built {len(examples)} not_confused examples")
    return examples


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="datasets/confusion_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Connecting to RC database…")
    conn = connect()

    print("Building confused examples…")
    confused = build_confused_examples(conn)

    print(f"Building {len(confused)} not_confused examples (balanced)…")
    not_confused = build_not_confused_examples(conn, n=len(confused), seed=args.seed)

    all_examples = confused + not_confused
    random.Random(args.seed).shuffle(all_examples)

    with open(args.out, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(asdict(ex)) + "\n")

    conn.close()

    # Print summary
    label_counts = {}
    type_counts = {}
    for ex in all_examples:
        label_counts[ex.label] = label_counts.get(ex.label, 0) + 1
        if ex.confusion_type:
            type_counts[ex.confusion_type] = type_counts.get(ex.confusion_type, 0) + 1

    print(f"\nDataset written to {args.out}")
    print(f"Total examples: {len(all_examples)}")
    print(f"Label distribution: {label_counts}")
    print(f"Confusion type distribution: {type_counts}")


if __name__ == "__main__":
    main()
