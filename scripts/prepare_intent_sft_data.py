"""
prepare_intent_sft_data.py - Prepare SFT data for MiniMind intent generation.

Converts session text -> structured intent JSON pairs, formats as
chat conversations for MiniMind SFT training, and outputs JSONL
with train/val/test splits.
"""

import json
import os
import argparse
import random
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.session_serializer import serialize_session
from src.intent_taxonomy import get_system_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_intent_prompt(
    titles: list,
    categories: list,
    timestamps: list = None,
    max_items: int = 10,
) -> str:
    """
    Build the user prompt for intent generation from a session.

    Returns a chat-style user message asking to analyze user intent.
    """
    session_text = serialize_session(
        titles=titles,
        categories=categories,
        timestamps=timestamps,
        max_items=max_items,
    )
    return (
        f"{session_text}\n\n"
        "Based on the above browsing/purchase history, what is the user's "
        "primary shopping intent and any secondary intents? "
        "Respond with a JSON object including primary_intent, "
        "secondary_intents, confidence, and evidence_items."
    )


def build_conversation(
    session: dict,
    label: dict,
    system_prompt: str = None,
) -> dict:
    """
    Build a chat conversation for MiniMind SFTDataset.

    Args:
        session: Session dict with item_titles, categories, timestamps.
        label: Intent label dict with primary_intent, secondary_intents, etc.
        system_prompt: Optional system prompt override.

    Returns:
        Dict with "conversations" key for SFTDataset consumption.
    """
    if system_prompt is None:
        system_prompt = get_system_prompt()

    titles = session.get("item_titles", [])
    categories = session.get("categories", [])
    timestamps = session.get("timestamps", None)

    user_content = build_intent_prompt(titles, categories, timestamps)
    assistant_content = json.dumps(label, ensure_ascii=False)

    return {
        "conversations": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def convert(
    sessions_path: str,
    labels_path: str,
    output_dir: str,
    system_prompt: str = None,
    seed: int = 42,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    cache_path: str = None,
):
    """
    Convert sessions + labels into SFT format data.

    Args:
        sessions_path: Path to session JSONL.
        labels_path: Path to teacher labels JSONL (session_id -> intent).
        output_dir: Output directory for SFT data.
        system_prompt: System prompt override.
        seed: Random seed for splitting.
        val_ratio: Ratio of validation data.
        test_ratio: Ratio of test data.
        cache_path: Optional path to cached labels to supplement labels_path.
    """
    # Load sessions
    sessions = []
    with open(sessions_path, "r", encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(sessions)} sessions")

    # Load labels
    labels = {}
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                labels[entry.get("session_id", "")] = entry.get("intent", {})
        logger.info(f"Loaded {len(labels)} labels from {labels_path}")

    # Also try loading from cache
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        for sid, entry in cache_data.get("labels", {}).items():
            if sid not in labels:
                intent_json = entry.get("intent_json", {})
                if intent_json:
                    labels[sid] = intent_json
        logger.info(f"After cache supplement: {len(labels)} labels")

    # Build session_id for each session
    for s in sessions:
        uid = s.get("user_id", "")
        target = s.get("target_item", "")
        last_ts = s.get("timestamps", [0])[-1] if s.get("timestamps") else 0
        s["_session_id"] = f"{uid}_{target}_{last_ts}"

    # Match sessions with labels
    matched = []
    for s in sessions:
        sid = s.get("_session_id")
        label = labels.get(sid) or labels.get(s.get("user_id", ""))
        # Also try matching by user_id + target_item
        alt_key = f"{s.get('user_id', '')}_{s.get('target_item', '')}"
        if label is None and alt_key in labels:
            label = labels[alt_key]

        if label and isinstance(label, dict) and len(label) > 0:
            matched.append((s, label))

    logger.info(f"Matched {len(matched)} sessions with labels out of {len(sessions)}")

    if len(matched) == 0:
        logger.error("No matched sessions with labels. Cannot create SFT data.")
        return

    # Build conversations
    conversations = []
    for s, label in matched:
        conv = build_conversation(s, label, system_prompt)
        conversations.append(conv)

    # Shuffle and split
    random.seed(seed)
    random.shuffle(conversations)

    n_total = len(conversations)
    n_test = max(1, int(n_total * test_ratio))
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val - n_test

    train_data = conversations[:n_train]
    val_data = conversations[n_train:n_train + n_val]
    test_data = conversations[n_train + n_val:]

    logger.info(
        f"Split: {len(train_data)} train, {len(val_data)} val, "
        f"{len(test_data)} test"
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    for name, data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for conv in data:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(data)} samples -> {path}")

    # Save combined also for easy access
    combined_path = os.path.join(output_dir, "all.jsonl")
    with open(combined_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(conversations)} total -> {combined_path}")

    # Summary
    primary_intents = {}
    for _, label in matched:
        pi = label.get("primary_intent", "unknown")
        primary_intents[pi] = primary_intents.get(pi, 0) + 1
    top_intents = sorted(primary_intents.items(), key=lambda x: -x[1])[:10]
    logger.info(f"Top primary intents: {top_intents}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SFT data for MiniMind intent generation."
    )
    parser.add_argument(
        "--sessions", required=True, help="Path to session JSONL"
    )
    parser.add_argument(
        "--labels", required=True, help="Path to teacher labels JSONL or cache JSON"
    )
    parser.add_argument(
        "--output-dir", default="./data/intent_sft",
        help="Output directory for SFT JSONL files"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.05, help="Validation ratio"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.05, help="Test ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--cache", default=None, help="Path to teacher label cache JSON (supplement)"
    )
    args = parser.parse_args()

    convert(
        sessions_path=args.sessions,
        labels_path=args.labels,
        output_dir=args.output_dir,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        cache_path=args.cache,
    )


if __name__ == "__main__":
    main()
