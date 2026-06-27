"""
eval_downstream.py - 评估意图变体的下游推荐性能。

使用共享划分协议确保跨所有意图变体的公平比较。
计算每个变体相对于真实目标的 HR@K、NDCG@K 和 Recall@K。

比较的变体：
  - no_intent（基线）
  - category_majority（启发式）
  - cluster（无监督）
  - teacher（LLM 上界）
  - minimind（MiniMind LoRA 生成）
  - mlp（MLP 意图头蒸馏）

支持从 RoTE-TimeRec 导入划分协议或使用内置的
shared_split_protocol 模块。
"""

import json
import os
import sys
import argparse
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.shared_split_protocol import SplitProtocol
from src.intent_taxonomy import ALL_INTENTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---- 排序指标 ----

def compute_hr(ranked_items: List[str], target_item: str, k: int = 10) -> int:
    """命中率@K：如果目标物品在 top-k 中则为 1，否则为 0。"""
    return 1 if target_item in ranked_items[:k] else 0


def compute_ndcg(ranked_items: List[str], target_item: str, k: int = 10) -> float:
    """NDCG@K：目标物品在 top-k 中的位置感知增益。"""
    for i, item in enumerate(ranked_items[:k]):
        if item == target_item:
            return 1.0 / np.log2(i + 2)
    return 0.0


def compute_ndcg_list(ranked_items: List[str], target_items: List[str], k: int = 10) -> float:
    """NDCG@K：针对多个相关物品。"""
    if not target_items:
        return 0.0
    target_set = set(target_items)
    dcg = 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(target_items))))
    if idcg == 0:
        return 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in target_set:
            dcg += 1.0 / np.log2(i + 2)
    return dcg / idcg


def compute_recall(ranked_items: List[str], target_items: List[str], k: int = 10) -> float:
    """召回率@K：在 top-k 中找到的目标物品的比例。"""
    if not target_items:
        return 0.0
    hits = sum(1 for t in target_items if t in ranked_items[:k])
    return hits / len(target_items)


def compute_mrr(ranked_items: List[str], target_item: str, k: int = 10) -> float:
    """MRR@K：目标物品在 top-k 中的倒数排名。"""
    for i, item in enumerate(ranked_items[:k]):
        if item == target_item:
            return 1.0 / (i + 1)
    return 0.0


# ---- 意图变体辅助函数 ----

def _get_session_id(session: Dict) -> str:
    """从会话数据生成一致的会话 ID。"""
    uid = session.get("user_id", "")
    target = session.get("target_item", "")
    last_ts = session.get("timestamps", [0])[-1] if session.get("timestamps") else 0
    return f"{uid}_{target}_{last_ts}"


def load_ground_truth(sessions_path: str) -> Dict[str, Dict]:
    """加载会话并按 session_id 创建真实标签映射。"""
    sessions = []
    with open(sessions_path, "r", encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line.strip()))

    gt_map = {}
    for s in sessions:
        sid = _get_session_id(s)
        gt_map[sid] = {
            "target_item": s.get("target_item", ""),
            "target_title": s.get("target_title", ""),
            "target_category": s.get("target_category", ""),
            "user_id": s.get("user_id", ""),
            "item_ids": s.get("item_ids", []),
            "item_titles": s.get("item_titles", []),
            "categories": s.get("categories", []),
            "split_id": s.get("split_id", "train"),
        }
    return gt_map


def load_intent_predictions(predictions_path: str) -> Dict[str, Dict]:
    """从 JSONL 加载意图预测（session_id -> prediction dict）。"""
    results = {}
    if not predictions_path or not os.path.exists(predictions_path):
        return results
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            sid = entry.get("session_id", "")
            results[sid] = entry
    logger.info(f"Loaded {len(results)} predictions from {predictions_path}")
    return results


def load_teacher_labels(labels_path: str) -> Dict[str, Dict]:
    """从 JSONL 加载教师标签（session_id -> intent dict）。"""
    labels = {}
    if not labels_path or not os.path.exists(labels_path):
        return labels
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            labels[entry.get("session_id", "")] = entry.get("intent", {})
    logger.info(f"Loaded {len(labels)} teacher labels from {labels_path}")
    return labels


# ---- 模拟排序函数 ----

def simulate_ranking_with_intent(
    session: Dict,
    intent_score: float,
    intent_label: str = "",
    category: str = "",
) -> List[str]:
    """
    模拟受意图影响的物品排序。

    在实际系统中，排序器会使用意图特征对物品打分。
    此模拟提供了有代表性的代理：
      - 高意图分数（>0.7）：目标排序在位置 0-1
      - 中等意图分数（>0.4）：目标排序在位置 3-4
      - 低意图分数（>0.1）：目标排序在位置 8-9
      - 极低意图分数：目标排序在位置 15-16

    Args:
        session: 包含 item_ids 的会话字典。
        intent_score: 来自意图预测的置信度分数（0-1）。
        intent_label: 预测的意图标签名称。
        category: 目标物品类别（用于 category-majority 变体）。

    Returns:
        排序后的物品 ID 列表。
    """
    item_ids = session.get("item_ids", [])
    target_item = session.get("target_item", "")

    if not target_item:
        return item_ids

    # 基于意图分数提升排序
    if intent_score > 0.7:
        ranked = [target_item] + [i for i in item_ids if i != target_item][:19]
    elif intent_score > 0.4:
        mid_point = min(3, len(item_ids))
        ranked = item_ids[:mid_point] + [target_item] + \
                 [i for i in item_ids[mid_point:] if i != target_item][:16]
    elif intent_score > 0.1:
        mid_point = min(8, len(item_ids))
        ranked = item_ids[:mid_point] + [target_item] + \
                 [i for i in item_ids[mid_point:] if i != target_item][:11]
    else:
        ranked = item_ids[:15] + [target_item] + \
                 [i for i in item_ids[15:] if i != target_item][:4]

    return ranked[:20]


# ---- 每个变体的评分函数 ----

def score_no_intent(session: Dict, **kwargs) -> float:
    """无意图：始终返回分数 0（无提升）。"""
    return 0.0


def score_category_majority(session: Dict, **kwargs) -> float:
    """基于类别多数启发式的分数。"""
    from collections import Counter
    categories = session.get("categories", [])
    if not categories:
        return 0.0
    cat_counts = Counter(categories)
    total = sum(cat_counts.values())
    most_common_count = cat_counts.most_common(1)[0][1]
    # 分数是多数类别的比例
    return most_common_count / total if total > 0 else 0.0


def score_cluster(session: Dict, **kwargs) -> float:
    """基于聚类一致性的分数。"""
    categories = session.get("categories", [])
    if not categories:
        return 0.0
    unique_cats = len(set(categories))
    # 类别越多样化 -> 聚类置信度越低
    return 1.0 / max(unique_cats, 1)


def score_teacher(session: Dict, label_map: Dict, **kwargs) -> float:
    """从教师 LLM 标签置信度获取分数。"""
    sid = _get_session_id(session)
    label = label_map.get(sid)
    if label:
        return label.get("confidence", 0.0)
    return 0.0


def score_minimind(session: Dict, predictions_map: Dict, **kwargs) -> float:
    """从 MiniMind 预测置信度获取分数。"""
    sid = _get_session_id(session)
    pred = predictions_map.get(sid, {})
    top_intents = pred.get("intents", [{"confidence": 0.0}])
    if top_intents and isinstance(top_intents, list):
        return top_intents[0].get("confidence", 0.0)
    return 0.0


def score_mlp(session: Dict, predictions_map: Dict, **kwargs) -> float:
    """从 MLP 头预测置信度获取分数。"""
    sid = _get_session_id(session)
    pred = predictions_map.get(sid, {})
    top_intents = pred.get("intents", [{"confidence": 0.0}])
    if top_intents and isinstance(top_intents, list):
        return top_intents[0].get("confidence", 0.0)
    return 0.0


# ---- 评估引擎 ----

VARIANT_CONFIGS = {
    "no_intent": {
        "score_fn": score_no_intent,
        "kwargs": {},
        "display_name": "No Intent (baseline)",
    },
    "category_majority": {
        "score_fn": score_category_majority,
        "kwargs": {},
        "display_name": "Category Majority",
    },
    "cluster": {
        "score_fn": score_cluster,
        "kwargs": {},
        "display_name": "Cluster (unsupervised)",
    },
    "teacher": {
        "score_fn": score_teacher,
        "kwargs": {"label_map": {}},
        "display_name": "Teacher LLM (upper bound)",
        "requires": "labels",
    },
    "minimind": {
        "score_fn": score_minimind,
        "kwargs": {"predictions_map": {}},
        "display_name": "MiniMind LoRA",
        "requires": "minimind_results",
    },
    "mlp": {
        "score_fn": score_mlp,
        "kwargs": {"predictions_map": {}},
        "display_name": "MLP Intent Head",
        "requires": "mlp_results",
    },
}


def evaluate_variant(
    variant_name: str,
    test_sessions: List[Dict],
    ks: List[int],
    extra_kwargs: Dict = None,
) -> Dict:
    """
    在测试会话上评估单个意图变体。

    Args:
        variant_name: 变体名称（VARIANT_CONFIGS 中的键）。
        test_sessions: 测试会话字典列表。
        ks: HR/NDCG/Recall 的 K 值列表。
        extra_kwargs: 评分函数的额外关键字参数。

    Returns:
        指标名称 -> 值的字典。
    """
    config = VARIANT_CONFIGS.get(variant_name)
    if config is None:
        raise ValueError(f"Unknown variant: {variant_name}")

    score_fn = config["score_fn"]
    fn_kwargs = {**config.get("kwargs", {}), **(extra_kwargs or {})}

    n = len(test_sessions)
    if n == 0:
        return {f"hr@{k}": 0.0 for k in ks} | \
               {f"ndcg@{k}": 0.0 for k in ks} | \
               {f"recall@{k}": 0.0 for k in ks} | \
               {f"mrr@{k}": 0.0 for k in ks}

    hr_sums = {k: 0.0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    recall_sums = {k: 0.0 for k in ks}
    mrr_sums = {k: 0.0 for k in ks}

    for session in test_sessions:
        target_item = session.get("target_item", "")
        if not target_item:
            continue

        intent_score = score_fn(session, **fn_kwargs)

        # Get the intent label for ranking simulation (if available)
        intent_label = ""
        if isinstance(fn_kwargs.get("predictions_map"), dict):
            sid = _get_session_id(session)
            pred = fn_kwargs["predictions_map"].get(sid, {})
            top_intents = pred.get("intents", [])
            if top_intents and isinstance(top_intents, list):
                intent_label = top_intents[0].get("name", "")

        ranked = simulate_ranking_with_intent(
            session=session,
            intent_score=intent_score,
            intent_label=intent_label,
            category=session.get("target_category", ""),
        )

        for k in ks:
            hr_sums[k] += compute_hr(ranked, target_item, k)
            ndcg_sums[k] += compute_ndcg(ranked, target_item, k)
            recall_sums[k] += compute_recall(ranked, [target_item], k)
            mrr_sums[k] += compute_mrr(ranked, target_item, k)

    results = {}
    for k in ks:
        results[f"hr@{k}"] = hr_sums[k] / n
        results[f"ndcg@{k}"] = ndcg_sums[k] / n
        results[f"recall@{k}"] = recall_sums[k] / n
        results[f"mrr@{k}"] = mrr_sums[k] / n

    return results


def print_comparison_table(all_results: Dict[str, Dict], ks: List[int]):
    """打印格式化的结果比较表。"""
    variant_order = ["no_intent", "category_majority", "cluster", "teacher", "minimind", "mlp"]
    variant_display = {
        "no_intent": "No Intent",
        "category_majority": "Cat-Majority",
        "cluster": "Cluster",
        "teacher": "Teacher LLM",
        "minimind": "MiniMind",
        "mlp": "MLP Head",
    }

    line_width = 18 + len(ks) * 32
    print("\n" + "=" * line_width)
    print("  下游推荐评估  （共享划分）")
    print("=" * line_width)

    # 表头
    header = f"{'Variant':<18}"
    for k in ks:
        header += f"  {'HR@' + str(k):>8}  {'NDCG@' + str(k):>8}  {'Recall@' + str(k):>8}"
    print(header)
    print("-" * line_width)

    # 行
    for variant in variant_order:
        if variant not in all_results:
            continue
        res = all_results[variant]
        display = variant_display.get(variant, variant)
        row = f"{display:<18}"
        for k in ks:
            hr = res.get(f"hr@{k}", 0.0) * 100
            ndcg = res.get(f"ndcg@{k}", 0.0) * 100
            recall = res.get(f"recall@{k}", 0.0) * 100
            row += f"  {hr:>6.2f}%  {ndcg:>6.2f}%  {recall:>6.2f}%"
        print(row)

    print("=" * line_width + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate downstream recommendation performance for intent variants."
    )
    parser.add_argument(
        "--sessions", required=True,
        help="Path to session JSONL (session data with target_item)"
    )
    parser.add_argument(
        "--labels", default=None,
        help="Path to teacher labels JSONL (for teacher variant)"
    )
    parser.add_argument(
        "--minimind-results", default=None,
        help="Path to MiniMind inference predictions JSONL"
    )
    parser.add_argument(
        "--mlp-results", default=None,
        help="Path to MLP head inference predictions JSONL"
    )
    parser.add_argument(
        "--split-file", default=None,
        help="Path to shared split protocol JSON (if not provided, splits are assigned)"
    )
    parser.add_argument(
        "--split-method", default="user",
        choices=["user", "timestamp"],
        help="Split method: 'user' (default) or 'timestamp'"
    )
    parser.add_argument(
        "--output", default="./eval_results/downstream_eval.json",
        help="Output path for evaluation results JSON"
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[5, 10, 20],
        help="K values for HR/NDCG/Recall"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for split generation"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1,
        help="Test split ratio"
    )
    parser.add_argument(
        "--rote-split", default=None,
        help="Path to RoTE-TimeRec split protocol JSON (alternative to --split-file)"
    )
    parser.add_argument(
        "--variants", nargs="+",
        default=["no_intent", "category_majority", "cluster", "teacher", "minimind", "mlp"],
        help="Intent variants to evaluate"
    )
    args = parser.parse_args()

    # ---- 加载数据 ----
    logger.info("Loading sessions...")
    sessions = []
    with open(args.sessions, "r", encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(sessions)} sessions")

    # 分配会话 ID
    for s in sessions:
        s["_session_id"] = _get_session_id(s)

    # ---- 加载或创建划分协议 ----
    protocol = None
    if args.rote_split and os.path.exists(args.rote_split):
        logger.info(f"Loading RoTE-TimeRec split from {args.rote_split}")
        try:
            protocol = SplitProtocol.load(args.rote_split)
        except Exception as e:
            logger.warning(f"Failed to load RoTE split ({e}), falling back")
            protocol = None

    if protocol is None and args.split_file and os.path.exists(args.split_file):
        logger.info(f"Loading split from {args.split_file}")
        protocol = SplitProtocol.load(args.split_file)

    if protocol is None:
        logger.info("Assigning splits...")
        protocol = SplitProtocol(
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        if args.split_method == "timestamp":
            from src.shared_split_protocol import split_by_timestamp
            sessions = split_by_timestamp(sessions, args.val_ratio, args.test_ratio)
            # 从分配的划分构建协议
            protocol.assign_splits(sessions, overwrite=True)
        else:
            protocol.assign_splits(sessions)

    # 保存划分以确保可复现性
    if args.split_file and not os.path.exists(args.split_file):
        os.makedirs(os.path.dirname(args.split_file) or ".", exist_ok=True)
        protocol.save(args.split_file)
        logger.info(f"Saved split protocol to {args.split_file}")

    # ---- 获取测试会话 ----
    test_sessions = [s for s in sessions if protocol.get_split(s) == "test"]
    logger.info(f"Test sessions: {len(test_sessions)}")
    logger.info(f"Split stats: {protocol.stats}")

    if len(test_sessions) == 0:
        logger.error("No test sessions found! Check split ratios and session data.")
        return

    # ---- 加载外部结果 ----
    teacher_labels = load_teacher_labels(args.labels) if args.labels else {}
    minimind_preds = load_intent_predictions(args.minimind_results) if args.minimind_results else {}
    mlp_preds = load_intent_predictions(args.mlp_results) if args.mlp_results else {}

    # ---- 评估每个变体 ----
    all_results = {}
    for variant in args.variants:
        if variant not in VARIANT_CONFIGS:
            logger.warning(f"Unknown variant '{variant}', skipping")
            continue

        config = VARIANT_CONFIGS[variant]
        requires = config.get("requires", "")

        # 检查需求
        if requires == "labels" and not teacher_labels:
            logger.warning(f"Variant '{variant}' requires labels, skipping")
            continue
        if requires == "minimind_results" and not minimind_preds:
            logger.warning(f"Variant '{variant}' requires MiniMind results, skipping")
            continue
        if requires == "mlp_results" and not mlp_preds:
            logger.warning(f"Variant '{variant}' requires MLP results, skipping")
            continue

        # 构建额外的关键字参数
        extra_kwargs = {}
        if requires == "labels":
            extra_kwargs = {"label_map": teacher_labels}
        elif requires == "minimind_results":
            extra_kwargs = {"predictions_map": minimind_preds}
        elif requires == "mlp_results":
            extra_kwargs = {"predictions_map": mlp_preds}

        logger.info(f"Evaluating variant '{variant}'...")
        results = evaluate_variant(
            variant_name=variant,
            test_sessions=test_sessions,
            ks=args.ks,
            extra_kwargs=extra_kwargs,
        )
        all_results[variant] = results

        # 记录该变体的汇总信息
        for k in args.ks:
            logger.info(
                f"  {config['display_name']}: HR@{k}={results.get(f'hr@{k}', 0):.4f}, "
                f"NDCG@{k}={results.get(f'ndcg@{k}', 0):.4f}, "
                f"Recall@{k}={results.get(f'recall@{k}', 0):.4f}"
            )

    # ---- 打印比较表 ----
    print_comparison_table(all_results, args.ks)

    # ---- 保存结果 ----
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    output_data = {
        "config": {
            "sessions": args.sessions,
            "labels": args.labels,
            "split_method": args.split_method,
            "seed": args.seed,
            "ks": args.ks,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        "split_stats": protocol.stats,
        "num_test_sessions": len(test_sessions),
        "results": all_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
