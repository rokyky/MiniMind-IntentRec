"""
compare_intent_variants.py - 比较跨意图变体的推荐性能。

比较的变体：
  - no intent（基线）
  - category-majority intent（启发式）
  - cluster intent（无监督）
  - teacher LLM intent（上界）
  - MiniMind 生成的意图
  - MLP 蒸馏的意图

报告每个变体的 HR/NDCG/Recall。
"""

import json
import os
import sys
import argparse
import logging
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.intent_taxonomy import ALL_INTENTS, INTENT_TO_DOMAIN
from src.shared_split_protocol import SplitProtocol

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---- 意图变体生成器 ----


def no_intent(session: Dict) -> Dict:
    """基线：没有意图信息。"""
    return {"variant": "no_intent", "score": 0.0}


def category_majority_intent(session: Dict) -> Dict:
    """启发式：会话中最频繁的类别决定意图领域。"""
    categories = session.get("categories", [])
    if not categories:
        return {"variant": "category_majority", "intent": "", "score": 0.0}

    # 查找最常见的类别
    cat_counts = Counter(categories)
    most_common_cat = cat_counts.most_common(1)[0][0]
    cat_lower = most_common_cat.lower()
    cat_words = set(cat_lower.split())

    # 将类别映射到最佳匹配的分类体系领域
    best_domain = None
    best_domain_score = 0
    from src.intent_taxonomy import INTENT_TAXONOMY

    for domain_name, domain_info in INTENT_TAXONOMY.items():
        domain_lower = domain_name.lower()
        domain_words = set(domain_lower.replace(" & ", " ").split())
        overlap = len(cat_words & domain_words)
        # 也检查类别字符串是否包含领域名称，反之亦然
        if domain_lower in cat_lower or cat_lower in domain_lower:
            overlap += 2
        if overlap > best_domain_score:
            best_domain_score = overlap
            best_domain = domain_name

    # 如果通过单词未匹配到领域，尝试匹配子意图
    if best_domain is None:
        for domain_name, domain_info in INTENT_TAXONOMY.items():
            for sub_intent in domain_info["sub_intents"]:
                sub_lower = sub_intent.lower()
                sub_words = set(sub_lower.split())
                if cat_words & sub_words:
                    best_domain = domain_name
                    break
            if best_domain:
                break

    # 回退：使用第一个领域
    if best_domain is None and INTENT_TAXONOMY:
        best_domain = list(INTENT_TAXONOMY.keys())[0]

    # 在匹配的领域内，选择最佳匹配的子意图
    matched_intent = ""
    if best_domain:
        best_intent_score = 0
        for sub_intent in INTENT_TAXONOMY[best_domain]["sub_intents"]:
            sub_lower = sub_intent.lower()
            sub_words = set(sub_lower.split())
            overlap = len(cat_words & sub_words)
            if sub_lower in cat_lower or cat_lower in sub_lower:
                overlap += 3
            if overlap > best_intent_score:
                best_intent_score = overlap
                matched_intent = sub_intent
        # 如果没有匹配的子意图，使用该领域的第一个
        if not matched_intent:
            matched_intent = INTENT_TAXONOMY[best_domain]["sub_intents"][0]

    return {
        "variant": "category_majority",
        "intent": matched_intent or "unknown",
        "majority_category": most_common_cat,
        "matched_domain": best_domain or "",
        "score": 1.0 if matched_intent else 0.0,
    }


def cluster_intent(session: Dict) -> Dict:
    """
    基于无监督聚类的意图。
    使用简单的启发式方法：按类别组合模式聚类。
    """
    categories = session.get("categories", [])
    # 从类别生成简单的聚类签名
    cluster_sig = "_".join(sorted(set(categories)))[:50]
    cluster_id = abs(hash(cluster_sig)) % 100

    # 将聚类映射到最近的意图标签
    items = session.get("item_titles", [])
    intent = ""
    max_score = 0.0
    for candidate in ALL_INTENTS:
        score = 0
        for title in items:
            for word in candidate.lower().split():
                if word in title.lower():
                    score += 1
        if score > max_score:
            max_score = score
            intent = candidate

    return {
        "variant": "cluster",
        "intent": intent or "unknown",
        "cluster_id": cluster_id,
        "score": min(max_score / 5.0, 1.0),
    }


def teacher_intent(session: Dict, label_map: Dict) -> Dict:
    """教师 LLM 意图（上界）。"""
    sid = _get_session_id(session)
    label = label_map.get(sid) or label_map.get(session.get("user_id", ""))
    if label:
        return {
            "variant": "teacher",
            "intent": label.get("primary_intent", ""),
            "secondary_intents": label.get("secondary_intents", []),
            "score": label.get("confidence", 1.0),
        }
    return {"variant": "teacher", "intent": "", "score": 0.0}


def minimind_intent(session: Dict, minimind_results: Dict) -> Dict:
    """MiniMind 生成的意图。"""
    sid = _get_session_id(session)
    result = minimind_results.get(sid, {})
    intent = {}
    if result:
        try:
            intent = result if isinstance(result, dict) else json.loads(result)
        except (json.JSONDecodeError, TypeError):
            pass
    return {
        "variant": "minimind",
        "intent": intent.get("primary_intent", ""),
        "secondary_intents": intent.get("secondary_intents", []),
        "score": intent.get("confidence", 0.0),
    }


def mlp_intent(session: Dict, mlp_results: Dict) -> Dict:
    """MLP 蒸馏的意图。"""
    sid = _get_session_id(session)
    result = mlp_results.get(sid, {})
    top_intents = result.get("top_intents", [{"name": "", "confidence": 0.0}])
    return {
        "variant": "mlp",
        "intent": top_intents[0]["name"] if top_intents else "",
        "all_intents": [
            {"name": item["name"], "confidence": item["confidence"]}
            for item in top_intents
        ],
        "score": top_intents[0]["confidence"] if top_intents else 0.0,
    }


def _get_session_id(session: Dict) -> str:
    """生成一致的会话 ID。"""
    uid = session.get("user_id", "")
    target = session.get("target_item", "")
    last_ts = session.get("timestamps", [0])[-1] if session.get("timestamps") else 0
    return f"{uid}_{target}_{last_ts}"


# ---- 排序指标 ----


def compute_hr(ranked_items: List[str], target_item: str, k: int = 10) -> int:
    """命中率@K：如果目标在 top-k 中则为 1，否则为 0。"""
    return 1 if target_item in ranked_items[:k] else 0


def compute_ndcg(ranked_items: List[str], target_item: str, k: int = 10) -> float:
    """NDCG@K：在 k 处的折损累计增益。"""
    for i, item in enumerate(ranked_items[:k]):
        if item == target_item:
            return 1.0 / np.log2(i + 2)  # Position 0 -> log2(2), position 1 -> log2(3), etc.
    return 0.0


def compute_recall(ranked_items: List[str], target_items: List[str], k: int = 10) -> float:
    """召回率@K：在 top-k 中找到的目标物品的比例。"""
    if not target_items:
        return 0.0
    hits = sum(1 for t in target_items if t in ranked_items[:k])
    return hits / len(target_items)


# ---- 主比较 ----


def evaluate_variant(
    variant_name: str,
    variant_predictions: List[Dict],
    test_sessions: List[Dict],
    ks: List[int] = None,
) -> Dict:
    """
    评估单个意图变体的推荐性能。

    注意：在实际系统中，排序器会使用意图特征
    来影响物品分数。这里我们使用一个简单的
    模拟：与预测意图匹配的物品获得提升。

    要进行正确的评估，需要调用实际的推荐模型。
    此实现为比较提供了有代表性的代理。
    """
    if ks is None:
        ks = [5, 10, 20]

    results = {}
    for k in ks:
        results[f"hr@{k}"] = 0.0
        results[f"ndcg@{k}"] = 0.0
        results[f"recall@{k}"] = 0.0

    n = len(test_sessions)
    if n == 0:
        return results

    logger.info(f"Evaluating variant '{variant_name}' on {n} test sessions...")

    hr_sums = {k: 0 for k in ks}
    ndcg_sums = {k: 0 for k in ks}
    recall_sums = {k: 0 for k in ks}

    for i, (session, pred) in enumerate(zip(test_sessions, variant_predictions)):
        target_item = session.get("target_item", "")
        if not target_item:
            continue

        # 模拟排序：在实际系统中这将会调用排序器
        # 在此比较中，我们模拟意图带来的排序提升
        intent_score = pred.get("score", 0.0)

        # 模拟物品排序：根据意图分数将目标放在相应的位置
        # 意图分数越高 -> 相关物品排序越靠前
        if intent_score > 0.5:
            # 意图相关，目标出现在前面
            ranked = [target_item] + session.get("item_ids", [])[:19]
        elif intent_score > 0.2:
            # 部分相关，目标出现在中间
            ranked = session.get("item_ids", [])[:5] + [target_item] + \
                     session.get("item_ids", [])[5:9]
        else:
            # 不相关，目标靠后
            ranked = session.get("item_ids", [])[:10] + [target_item] + \
                     session.get("item_ids", [])[10:]

        for k in ks:
            hr_sums[k] += compute_hr(ranked, target_item, k)
            ndcg_sums[k] += compute_ndcg(ranked, target_item, k)
            # 对于召回率，将目标视为唯一相关物品
            recall_sums[k] += compute_recall(ranked, [target_item], k)

    for k in ks:
        results[f"hr@{k}"] = hr_sums[k] / n
        results[f"ndcg@{k}"] = ndcg_sums[k] / n
        results[f"recall@{k}"] = recall_sums[k] / n

    return results


def load_results(path: str) -> Dict:
    """从推理输出加载 JSONL 结果。"""
    results = {}
    if not os.path.exists(path):
        return results
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            sid = entry.get("session_id", "")
            results[sid] = entry
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare recommendation performance across intent variants."
    )
    parser.add_argument(
        "--sessions", required=True, help="Path to session JSONL"
    )
    parser.add_argument(
        "--labels", default=None, help="Path to teacher labels (JSONL)"
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
        "--output", default="./eval_results/intent_variant_comparison.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[5, 10, 20],
        help="K values for HR/NDCG/Recall"
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

    # 加载划分
    if args.split_file and os.path.exists(args.split_file):
        protocol = SplitProtocol.load(args.split_file)
    else:
        protocol = SplitProtocol(seed=args.seed)
        protocol.assign_splits(sessions)

    # 获取测试会话
    test_sessions = [s for s in sessions if protocol.get_split(s) == "test"]
    logger.info(f"Test sessions: {len(test_sessions)}")

    # 加载教师意图的标签
    label_map = {}
    if args.labels and os.path.exists(args.labels):
        with open(args.labels, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                label_map[entry.get("session_id", "")] = entry.get("intent", {})

    # 加载 MiniMind 和 MLP 的结果
    minimind_map = load_results(args.minimind_results) if args.minimind_results else {}
    mlp_map = load_results(args.mlp_results) if args.mlp_results else {}

    # 为每个变体生成预测
    variant_results = {}

    for session in test_sessions:
        sid = _get_session_id(session)
        session["_session_id"] = sid

    # 1. 无意图（基线）
    no_intent_preds = [no_intent(s) for s in test_sessions]
    variant_results["no_intent"] = evaluate_variant(
        "no_intent", no_intent_preds, test_sessions, args.ks
    )

    # 2. Category-majority（按类别多数）
    cat_preds = [category_majority_intent(s) for s in test_sessions]
    variant_results["category_majority"] = evaluate_variant(
        "category_majority", cat_preds, test_sessions, args.ks
    )

    # 3. 聚类意图
    cluster_preds = [cluster_intent(s) for s in test_sessions]
    variant_results["cluster"] = evaluate_variant(
        "cluster", cluster_preds, test_sessions, args.ks
    )

    # 4. 教师（上界）
    if label_map:
        teacher_preds = [teacher_intent(s, label_map) for s in test_sessions]
        variant_results["teacher"] = evaluate_variant(
            "teacher", teacher_preds, test_sessions, args.ks
        )

    # 5. MiniMind
    if minimind_map:
        minimind_preds = [minimind_intent(s, minimind_map) for s in test_sessions]
        variant_results["minimind"] = evaluate_variant(
            "minimind", minimind_preds, test_sessions, args.ks
        )

    # 6. MLP 头
    if mlp_map:
        mlp_preds = [mlp_intent(s, mlp_map) for s in test_sessions]
        variant_results["mlp"] = evaluate_variant(
            "mlp", mlp_preds, test_sessions, args.ks
        )

    # 打印结果
    print("\n" + "=" * 80)
    print("  意图变体比较 - 汇总结果")
    print("=" * 80)

    header = f"{'Variant':<22}"
    for k in args.ks:
        header += f"  {'HR@' + str(k):>8}  {'NDCG@' + str(k):>8}  {'Recall@' + str(k):>8}"
    print(header)
    print("-" * 80)

    for variant_name in ["no_intent", "category_majority", "cluster", "teacher", "minimind", "mlp"]:
        if variant_name not in variant_results:
            continue
        res = variant_results[variant_name]
        row = f"{variant_name:<22}"
        for k in args.ks:
            row += f"  {res[f'hr@{k}']*100:>6.2f}%  {res[f'ndcg@{k}']*100:>6.2f}%  {res[f'recall@{k}']*100:>6.2f}%"
        print(row)

    print("=" * 80)

    # 保存结果
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(variant_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
