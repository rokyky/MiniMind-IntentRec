"""
eval_downstream.py - Evaluate downstream recommendation performance for intent variants.

Uses a shared split protocol to ensure fair comparison across all intent variants.
Computes HR@K, NDCG@K, and Recall@K for each variant against ground-truth targets.

Variants compared:
  - no_intent (baseline)
  - category_majority (heuristic)
  - cluster (unsupervised)
  - teacher (LLM upper bound)
  - minimind (MiniMind LoRA generated)
  - mlp (MLP intent head distilled)

Supports importing split protocols from RoTE-TimeRec or using the built-in
shared_split_protocol module.
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


# ---- Ranking Metrics ----

def compute_hr(ranked_items: List[str], target_item: str, k: int = 10) -> int:
    """Hit Rate@K: 1 if target item is in top-k, else 0."""
    return 1 if target_item in ranked_items[:k] else 0


def compute_ndcg(ranked_items: List[str], target_item: str, k: int = 10) -> float:
    """NDCG@K: position-aware gain for target item in top-k."""
    for i, item in enumerate(ranked_items[:k]):
        if item == target_item:
            return 1.0 / np.log2(i + 2)
    return 0.0


def compute_ndcg_list(ranked_items: List[str], target_items: List[str], k: int = 10) -> float:
    """NDCG@K for multiple relevant items."""
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
    """Recall@K: fraction of target items found in top-k."""
    if not target_items:
        return 0.0
    hits = sum(1 for t in target_items if t in ranked_items[:k])
    return hits / len(target_items)


def compute_mrr(ranked_items: List[str], target_item: str, k: int = 10) -> float:
    """MRR@K: reciprocal rank of the target item in top-k."""
    for i, item in enumerate(ranked_items[:k]):
        if item == target_item:
            return 1.0 / (i + 1)
    return 0.0


# ---- Intent Variant Helpers ----

def _get_session_id(session: Dict) -> str:
    """Generate consistent session ID from session data."""
    uid = session.get("user_id", "")
    target = session.get("target_item", "")
    last_ts = session.get("timestamps", [0])[-1] if session.get("timestamps") else 0
    return f"{uid}_{target}_{last_ts}"


def load_ground_truth(sessions_path: str) -> Dict[str, Dict]:
    """Load sessions and create a ground-truth map by session_id."""
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
    """Load intent predictions from JSONL (session_id -> prediction dict)."""
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
    """Load teacher labels from JSONL (session_id -> intent dict)."""
    labels = {}
    if not labels_path or not os.path.exists(labels_path):
        return labels
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            labels[entry.get("session_id", "")] = entry.get("intent", {})
    logger.info(f"Loaded {len(labels)} teacher labels from {labels_path}")
    return labels


# ---- Simulated Ranking Functions ----

def simulate_ranking_with_intent(
    session: Dict,
    intent_score: float,
    intent_label: str = "",
    category: str = "",
) -> List[str]:
    """
    Simulate item ranking influenced by intent.

    In a real system, the ranker would use intent features to score items.
    This simulation provides a representative proxy:
      - High intent score (>0.5): target ranks at position 0-1
      - Medium intent score (>0.2): target ranks at position 5-6
      - Low intent score: target ranks at position 10-11

    Args:
        session: Session dict with item_ids.
        intent_score: Confidence score from intent prediction (0-1).
        intent_label: Predicted intent label name.
        category: Target item category (for category-majority variant).

    Returns:
        Ranked list of item IDs.
    """
    item_ids = session.get("item_ids", [])
    target_item = session.get("target_item", "")

    if not target_item:
        return item_ids

    # Boost ranking based on intent score
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


# ---- Per-Variant Scoring Functions ----

def score_no_intent(session: Dict, **kwargs) -> float:
    """No intent: always returns score 0 (no boost)."""
    return 0.0


def score_category_majority(session: Dict, **kwargs) -> float:
    """Score based on category majority heuristic."""
    from collections import Counter
    categories = session.get("categories", [])
    if not categories:
        return 0.0
    cat_counts = Counter(categories)
    total = sum(cat_counts.values())
    most_common_count = cat_counts.most_common(1)[0][1]
    # Score is the proportion of the majority category
    return most_common_count / total if total > 0 else 0.0


def score_cluster(session: Dict, **kwargs) -> float:
    """Score based on cluster consistency."""
    categories = session.get("categories", [])
    if not categories:
        return 0.0
    unique_cats = len(set(categories))
    # More diverse categories -> lower cluster confidence
    return 1.0 / max(unique_cats, 1)


def score_teacher(session: Dict, label_map: Dict, **kwargs) -> float:
    """Score from teacher LLM label confidence."""
    sid = _get_session_id(session)
    label = label_map.get(sid)
    if label:
        return label.get("confidence", 0.0)
    return 0.0


def score_minimind(session: Dict, predictions_map: Dict, **kwargs) -> float:
    """Score from MiniMind prediction confidence."""
    sid = _get_session_id(session)
    pred = predictions_map.get(sid, {})
    top_intents = pred.get("intents", [{"confidence": 0.0}])
    if top_intents and isinstance(top_intents, list):
        return top_intents[0].get("confidence", 0.0)
    return 0.0


def score_mlp(session: Dict, predictions_map: Dict, **kwargs) -> float:
    """Score from MLP head prediction confidence."""
    sid = _get_session_id(session)
    pred = predictions_map.get(sid, {})
    top_intents = pred.get("intents", [{"confidence": 0.0}])
    if top_intents and isinstance(top_intents, list):
        return top_intents[0].get("confidence", 0.0)
    return 0.0


# ---- Evaluation Engine ----

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
    Evaluate a single intent variant on test sessions.

    Args:
        variant_name: Name of the variant (key into VARIANT_CONFIGS).
        test_sessions: List of test session dicts.
        ks: List of K values for HR/NDCG/Recall.
        extra_kwargs: Additional keyword arguments for the score function.

    Returns:
        Dict of metric_name -> value.
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
    """Print a formatted comparison table of results."""
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
    print("  Downstream Recommendation Evaluation  (shared split)")
    print("=" * line_width)

    # Header
    header = f"{'Variant':<18}"
    for k in ks:
        header += f"  {'HR@' + str(k):>8}  {'NDCG@' + str(k):>8}  {'Recall@' + str(k):>8}"
    print(header)
    print("-" * line_width)

    # Rows
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

    # ---- Load data ----
    logger.info("Loading sessions...")
    sessions = []
    with open(args.sessions, "r", encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(sessions)} sessions")

    # Assign session IDs
    for s in sessions:
        s["_session_id"] = _get_session_id(s)

    # ---- Load or create split protocol ----
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
            # Build protocol from assigned splits
            protocol.assign_splits(sessions, overwrite=True)
        else:
            protocol.assign_splits(sessions)

    # Save split for reproducibility
    if args.split_file and not os.path.exists(args.split_file):
        os.makedirs(os.path.dirname(args.split_file) or ".", exist_ok=True)
        protocol.save(args.split_file)
        logger.info(f"Saved split protocol to {args.split_file}")

    # ---- Get test sessions ----
    test_sessions = [s for s in sessions if protocol.get_split(s) == "test"]
    logger.info(f"Test sessions: {len(test_sessions)}")
    logger.info(f"Split stats: {protocol.stats}")

    if len(test_sessions) == 0:
        logger.error("No test sessions found! Check split ratios and session data.")
        return

    # ---- Load external results ----
    teacher_labels = load_teacher_labels(args.labels) if args.labels else {}
    minimind_preds = load_intent_predictions(args.minimind_results) if args.minimind_results else {}
    mlp_preds = load_intent_predictions(args.mlp_results) if args.mlp_results else {}

    # ---- Evaluate each variant ----
    all_results = {}
    for variant in args.variants:
        if variant not in VARIANT_CONFIGS:
            logger.warning(f"Unknown variant '{variant}', skipping")
            continue

        config = VARIANT_CONFIGS[variant]
        requires = config.get("requires", "")

        # Check requirements
        if requires == "labels" and not teacher_labels:
            logger.warning(f"Variant '{variant}' requires labels, skipping")
            continue
        if requires == "minimind_results" and not minimind_preds:
            logger.warning(f"Variant '{variant}' requires MiniMind results, skipping")
            continue
        if requires == "mlp_results" and not mlp_preds:
            logger.warning(f"Variant '{variant}' requires MLP results, skipping")
            continue

        # Build extra kwargs
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

        # Log summary for this variant
        for k in args.ks:
            logger.info(
                f"  {config['display_name']}: HR@{k}={results.get(f'hr@{k}', 0):.4f}, "
                f"NDCG@{k}={results.get(f'ndcg@{k}', 0):.4f}, "
                f"Recall@{k}={results.get(f'recall@{k}', 0):.4f}"
            )

    # ---- Print comparison table ----
    print_comparison_table(all_results, args.ks)

    # ---- Save results ----
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
