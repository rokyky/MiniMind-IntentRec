"""
prepare_intent_sft_data.py - 为 MiniMind 意图生成准备 SFT 数据。

将会话文本转换为结构化意图 JSON 对，格式化为
MiniMind SFT 训练的聊天对话，并输出包含
训练/验证/测试划分的 JSONL。
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
    从会话构建用于意图生成的用户提示。

    返回要求分析用户意图的聊天风格用户消息。
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
    为 MiniMind SFTDataset 构建聊天对话。

    Args:
        session: 包含 item_titles、categories、timestamps 的会话字典。
        label: 包含 primary_intent、secondary_intents 等的意图标签字典。
        system_prompt: 可选的系统提示覆盖。

    Returns:
        带有 SFTDataset 使用的 "conversations" 键的字典。
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
    将会话+标签转换为 SFT 格式数据。

    Args:
        sessions_path: 会话 JSONL 的路径。
        labels_path: 教师标签 JSONL 的路径（session_id -> intent）。
        output_dir: SFT 数据的输出目录。
        system_prompt: 系统提示覆盖。
        seed: 用于划分的随机种子。
        val_ratio: 验证数据比例。
        test_ratio: 测试数据比例。
        cache_path: 可选的缓存标签路径，用于补充 labels_path。
    """
    # 加载会话
    sessions = []
    with open(sessions_path, "r", encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(sessions)} sessions")

    # 加载标签
    labels = {}
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                labels[entry.get("session_id", "")] = entry.get("intent", {})
        logger.info(f"Loaded {len(labels)} labels from {labels_path}")

    # 也尝试从缓存加载
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        for sid, entry in cache_data.get("labels", {}).items():
            if sid not in labels:
                intent_json = entry.get("intent_json", {})
                if intent_json:
                    labels[sid] = intent_json
        logger.info(f"After cache supplement: {len(labels)} labels")

    # 为每个会话构建 session_id
    for s in sessions:
        uid = s.get("user_id", "")
        target = s.get("target_item", "")
        last_ts = s.get("timestamps", [0])[-1] if s.get("timestamps") else 0
        s["_session_id"] = f"{uid}_{target}_{last_ts}"

    # 将会话与标签匹配
    matched = []
    for s in sessions:
        sid = s.get("_session_id")
        label = labels.get(sid) or labels.get(s.get("user_id", ""))
        # 也尝试按 user_id + target_item 匹配
        alt_key = f"{s.get('user_id', '')}_{s.get('target_item', '')}"
        if label is None and alt_key in labels:
            label = labels[alt_key]

        if label and isinstance(label, dict) and len(label) > 0:
            matched.append((s, label))

    logger.info(f"Matched {len(matched)} sessions with labels out of {len(sessions)}")

    if len(matched) == 0:
        logger.error("No matched sessions with labels. Cannot create SFT data.")
        return

    # 构建对话
    conversations = []
    for s, label in matched:
        conv = build_conversation(s, label, system_prompt)
        conversations.append(conv)

    # 打乱并划分
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

    # 保存
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

    # 保存组合文件以便于访问
    combined_path = os.path.join(output_dir, "all.jsonl")
    with open(combined_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(conversations)} total -> {combined_path}")

    # 汇总
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
