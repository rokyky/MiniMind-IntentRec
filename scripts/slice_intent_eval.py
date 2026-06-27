"""
slice_intent_eval.py - 意图变体的切片评估。

在特定用户切片上进行评估：
  - short_history：历史长度排在末尾 33% 的用户
  - cold_start：交互少于 5 次的用户
  - session_drift：最后 3 个物品中类别发生切换的用户

报告每个意图变体在每个切片上的 HR/NDCG。
"""

import json
import os
import sys
import argparse
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.shared_split_protocol import SplitProtocol
from src.intent_taxonomy import ALL_INTENTS
from scripts.compare_intent_variants import (
    no_intent,
    category_majority_intent,
    cluster_intent,
    teacher_intent,
    minimind_intent,
    mlp_intent,
    evaluate_variant,
    load_results,
    _get_session_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def identify_slices(sessions: List[Dict]) -> Dict[str, List[int]]:
    """
    基于会话属性识别用户切片。

    返回切片名称 -> 会话索引列表的字典。
    """
    # 计算每个用户的历史长度
    user_history_len = defaultdict(list)
    for i, s in enumerate(sessions):
        uid = s.get("user_id", "")
        user_history_len[uid].append({
            "idx": i,
            "session_len": len(s.get("item_ids", [])),
        })

    # Short-history（短历史）：按平均历史长度排在末尾 33% 的用户
    user_avg_len = {}
    for uid, entries in user_history_len.items():
        user_avg_len[uid] = np.mean([e["session_len"] for e in entries])

    sorted_users = sorted(user_avg_len.items(), key=lambda x: x[1])
    threshold_idx = max(1, int(len(sorted_users) * 0.33))
    short_history_users = set(u[0] for u in sorted_users[:threshold_idx])

    # Cold-start（冷启动）：总交互少于 5 次的用户
    cold_start_users = set()
    for uid, entries in user_history_len.items():
        total_interactions = sum(e["session_len"] for e in entries)
        if total_interactions < 5:
            cold_start_users.add(uid)

    # Session-drift（会话漂移）：最后 3 个物品中类别发生切换
    session_drift_indices = set()
    for i, s in enumerate(sessions):
        categories = s.get("categories", [])
        if len(categories) >= 3:
            last_3 = categories[-3:]
            # 如果最后 3 个类别不完全相同则视为漂移
            if len(set(last_3)) > 1:
                session_drift_indices.add(i)

    slices = {
        "all": list(range(len(sessions))),
        "short_history": [
            i for i, s in enumerate(sessions)
            if s.get("user_id", "") in short_history_users
        ],
        "cold_start": [
            i for i, s in enumerate(sessions)
            if s.get("user_id", "") in cold_start_users
        ],
        "session_drift": list(session_drift_indices),
    }

    logger.info(f"Slice sizes:")
    for name, indices in slices.items():
        logger.info(f"  {name}: {len(indices)}")

    return slices


def main():
    parser = argparse.ArgumentParser(
        description="Slice-based evaluation of intent variants."
    )
    parser.add_argument(
        "--sessions", required=True, help="Path to session JSONL"
    )
    parser.add_argument(
        "--labels", default=None, help="Path to teacher labels"
    )
    parser.add_argument(
        "--minimind-results", default=None, help="Path to MiniMind inference results"
    )
    parser.add_argument(
        "--mlp-results", default=None, help="Path to MLP head inference results"
    )
    parser.add_argument(
        "--split-file", default=None, help="Path to split protocol JSON"
    )
    parser.add_argument(
        "--output", default="./eval_results/slice_eval_results.json",
        help="Output path"
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[5, 10],
        help="K values for HR/NDCG"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()

    # 加载会话
    sessions = []
    with open(args.sessions, "r", encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(sessions)} sessions")

    # 生成会话 ID
    for s in sessions:
        s["_session_id"] = _get_session_id(s)

    # 加载划分
    if args.split_file and os.path.exists(args.split_file):
        protocol = SplitProtocol.load(args.split_file)
    else:
        protocol = SplitProtocol(seed=args.seed)
        protocol.assign_splits(sessions)

    # 获取测试会话
    test_session_indices = [
        i for i, s in enumerate(sessions)
        if protocol.get_split(s) == "test"
    ]
    test_sessions = [sessions[i] for i in test_session_indices]
    logger.info(f"Test sessions: {len(test_sessions)}")

    # 识别切片
    slices = identify_slices(sessions)

    # 将切片过滤为仅测试会话
    test_idx_set = set(test_session_indices)
    test_slices = {}
    for slice_name, indices in slices.items():
        test_slice_indices = [
            i for i in indices if i in test_idx_set
        ]
        if test_slice_indices:
            test_slices[slice_name] = test_slice_indices
            logger.info(f"  {slice_name} in test: {len(test_slice_indices)}")

    # 加载标签和结果
    label_map = {}
    if args.labels and os.path.exists(args.labels):
        with open(args.labels, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                label_map[entry.get("session_id", "")] = entry.get("intent", {})

    minimind_map = load_results(args.minimind_results) if args.minimind_results else {}
    mlp_map = load_results(args.mlp_results) if args.mlp_results else {}

    # 按切片和变体进行评估
    all_results = {}
    variant_generators = {
        "no_intent": (no_intent, {}),
        "category_majority": (category_majority_intent, {}),
        "cluster": (cluster_intent, {}),
    }
    if label_map:
        variant_generators["teacher"] = (teacher_intent, {"label_map": label_map})
    if minimind_map:
        variant_generators["minimind"] = (minimind_intent, {"minimind_results": minimind_map})
    if mlp_map:
        variant_generators["mlp"] = (mlp_intent, {"mlp_results": mlp_map})

    for slice_name, slice_indices in test_slices.items():
        slice_sessions = [sessions[i] for i in slice_indices]
        all_results[slice_name] = {}

        for variant_name, (gen_fn, extra_args) in variant_generators.items():
            if variant_name == "teacher":
                preds = [gen_fn(s, extra_args["label_map"]) for s in slice_sessions]
            elif variant_name == "minimind":
                preds = [gen_fn(s, extra_args["minimind_results"]) for s in slice_sessions]
            elif variant_name == "mlp":
                preds = [gen_fn(s, extra_args["mlp_results"]) for s in slice_sessions]
            else:
                preds = [gen_fn(s) for s in slice_sessions]

            results = evaluate_variant(
                variant_name, preds, slice_sessions, args.ks
            )
            all_results[slice_name][variant_name] = results

    # 打印结果
    print("\n" + "=" * 100)
    print("  切片评估结果")
    print("=" * 100)

    for slice_name in ["all", "short_history", "cold_start", "session_drift"]:
        if slice_name not in all_results:
            continue
        print(f"\n--- {slice_name.upper()} ({sum(1 for i in (test_slices.get(slice_name, []))):d} sessions) ---")

        header = f"{'Variant':<22}"
        for k in args.ks:
            header += f"  {'HR@' + str(k):>8}  {'NDCG@' + str(k):>8}"
        print(header)
        print("-" * 60)

        for variant_name in ["no_intent", "category_majority", "cluster", "teacher", "minimind", "mlp"]:
            if variant_name not in all_results.get(slice_name, {}):
                continue
            res = all_results[slice_name][variant_name]
            row = f"{variant_name:<22}"
            for k in args.ks:
                row += f"  {res[f'hr@{k}']*100:>6.2f}%  {res[f'ndcg@{k}']*100:>6.2f}%"
            print(row)

    print("\n" + "=" * 100)

    # 保存结果
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
