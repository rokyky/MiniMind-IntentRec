"""
build_session_data.py - 从用户物品序列构建会话样本。

读取用户行为数据（物品 ID、标题、类别、时间戳），
按用户分组，按时间戳排序，并从每个用户最近的 N 个物品创建会话。
每个会话输出包括：user_id, item_ids, item_titles, categories,
timestamps, target_item, 和 split_id。
"""

import json
import os
import argparse
import random
import logging
from collections import defaultdict
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_metadata(metadata_path: str) -> Dict[str, Dict]:
    """从 JSONL 加载物品元数据（标题、类别）。"""
    metadata = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            item_id = item.get("asin") or item.get("item_id") or item.get("id")
            metadata[item_id] = {
                "title": item.get("title", ""),
                "category": item.get("category", ""),
                "description": item.get("description", ""),
            }
    logger.info(f"Loaded {len(metadata)} items from metadata.")
    return metadata


def load_interactions(interactions_path: str, metadata: Dict[str, Dict]) -> List[Dict]:
    """加载用户-物品交互数据，并用元数据丰富。"""
    sessions = []
    with open(interactions_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            user_id = rec.get("user_id") or rec.get("reviewerID") or rec.get("uid")
            item_id = rec.get("item_id") or rec.get("asin") or rec.get("iid")
            timestamp = rec.get("timestamp") or rec.get("unixReviewTime") or 0
            meta = metadata.get(item_id, {})
            sessions.append({
                "user_id": str(user_id),
                "item_id": str(item_id),
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "timestamp": int(timestamp) if timestamp else 0,
            })
    logger.info(f"Loaded {len(sessions)} interactions.")
    return sessions


def build_sessions(
    interactions: List[Dict],
    max_session_len: int = 10,
    min_session_len: int = 2,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> List[Dict]:
    """
    按用户将交互分组，按时间戳排序，创建滑动窗口会话。

    每个会话：
      - user_id
      - item_ids：会话中的物品 ID 列表（历史）
      - item_titles：对应的标题
      - categories：对应的类别
      - timestamps：对应的时间戳
      - target_item：会话窗口后的下一个物品
      - split_id：'train' | 'val' | 'test'
    """
    # 按用户分组
    user_items = defaultdict(list)
    for rec in interactions:
        user_items[rec["user_id"]].append(rec)

    # 将每个用户的物品按时间戳排序，去重连续相同物品
    for user_id in user_items:
        user_items[user_id].sort(key=lambda x: x["timestamp"])
        # 移除连续相同的 item_id
        deduped = []
        for rec in user_items[user_id]:
            if not deduped or rec["item_id"] != deduped[-1]["item_id"]:
                deduped.append(rec)
        user_items[user_id] = deduped

    # 构建会话：在排序后的物品上滑动窗口
    sessions = []
    for user_id, items in user_items.items():
        if len(items) < min_session_len + 1:
            continue  # 至少需要 min_session_len 个历史物品 + 1 个目标
        for i in range(len(items) - 1):
            history_end = min(i + max_session_len, len(items) - 1)
            history = items[i:history_end]
            if len(history) < min_session_len:
                continue
            target = items[history_end]
            sessions.append({
                "user_id": user_id,
                "item_ids": [h["item_id"] for h in history],
                "item_titles": [h["title"] for h in history],
                "categories": [h["category"] for h in history],
                "timestamps": [h["timestamp"] for h in history],
                "target_item": target["item_id"],
                "target_title": target["title"],
                "target_category": target["category"],
            })

    logger.info(f"Built {len(sessions)} raw sessions.")
    return sessions


def assign_splits(
    sessions: List[Dict],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> List[Dict]:
    """按用户分配训练/验证/测试划分以防止数据泄露。"""
    random.seed(seed)
    # 按用户将会话分组
    user_sessions = defaultdict(list)
    for s in sessions:
        user_sessions[s["user_id"]].append(s)

    train_sessions, val_sessions, test_sessions = [], [], []
    for uid, us in user_sessions.items():
        us.sort(key=lambda x: x["timestamps"][-1] if x["timestamps"] else 0)
        total = len(us)
        n_test = max(1, int(total * test_ratio)) if test_ratio > 0 else 0
        n_val = max(1, int(total * val_ratio)) if val_ratio > 0 else 0
        n_train = total - n_test - n_val
        if n_train <= 0:
            # 如果太少，最早的放训练集，最晚的放测试集
            n_train = total // 2
            n_test = total - n_train
            n_val = 0

        test_slice = us[-n_test:] if n_test > 0 else []
        val_slice = us[-(n_test + n_val):-n_test] if n_val > 0 else []
        train_slice = us[:-(n_test + n_val)] if (n_test + n_val) > 0 else us

        for s in train_slice:
            s["split_id"] = "train"
        train_sessions.extend(train_slice)
        for s in val_slice:
            s["split_id"] = "val"
        val_sessions.extend(val_slice)
        for s in test_slice:
            s["split_id"] = "test"
        test_sessions.extend(test_slice)

    all_sessions = train_sessions + val_sessions + test_sessions
    logger.info(
        f"Split: {len(train_sessions)} train, {len(val_sessions)} val, "
        f"{len(test_sessions)} test"
    )
    return all_sessions


def save_sessions(sessions: List[Dict], output_path: str):
    """将会话保存为 JSONL。"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(sessions)} sessions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build session samples from user behavior data."
    )
    parser.add_argument(
        "--metadata", required=True, help="Item metadata JSONL (asin, title, category)"
    )
    parser.add_argument(
        "--interactions", required=True, help="User-item interactions JSONL"
    )
    parser.add_argument(
        "--output", default="./data/sessions.jsonl", help="Output session JSONL path"
    )
    parser.add_argument(
        "--max-session-len", type=int, default=10, help="Max items per session"
    )
    parser.add_argument(
        "--min-session-len", type=int, default=2, help="Min items per session"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logger.info("Loading metadata...")
    metadata = load_metadata(args.metadata)

    logger.info("Loading interactions...")
    interactions = load_interactions(args.interactions, metadata)

    logger.info("Building sessions...")
    raw_sessions = build_sessions(
        interactions,
        max_session_len=args.max_session_len,
        min_session_len=args.min_session_len,
        seed=args.seed,
    )

    logger.info("Assigning splits...")
    sessions = assign_splits(
        raw_sessions,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    save_sessions(sessions, args.output)

    # Summary stats
    lens = [len(s["item_ids"]) for s in sessions]
    logger.info(f"Session length stats: min={min(lens)}, max={max(lens)}, "
                f"avg={sum(lens)/len(lens):.1f}")
    cat_set = set()
    for s in sessions:
        cat_set.update(s["categories"])
    logger.info(f"Unique categories: {len(cat_set)}")
    user_set = set(s["user_id"] for s in sessions)
    logger.info(f"Unique users: {len(user_set)}")


if __name__ == "__main__":
    main()
