"""
check_taxonomy_stability.py - Taxonomy stability checker.

Detects duplicate intent labels, overlapping intent definitions,
reports coverage (% of sessions with valid intents), and
validates all cached labels conform to the taxonomy schema.
"""

import json
import os
import argparse
import sys
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.intent_taxonomy import (
    INTENT_TAXONOMY,
    ALL_INTENTS,
    INTENT_TO_DOMAIN,
    validate_intent_label,
    check_taxonomy_stability as taxonomy_check,
    get_taxonomy_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def check_duplicate_intents() -> list:
    """Check for intent labels that appear in multiple domains."""
    issues = []
    intent_domains = {}
    for domain, info in INTENT_TAXONOMY.items():
        for intent in info["sub_intents"]:
            if intent in intent_domains:
                intent_domains[intent].append(domain)
            else:
                intent_domains[intent] = [domain]
    for intent, domains in intent_domains.items():
        if len(domains) > 1:
            issues.append(
                f"Intent '{intent}' appears in multiple domains: {domains}"
            )
    return issues


def check_overlapping_definitions() -> list:
    """Check for potentially overlapping intent definitions."""
    issues = []
    # Group intents by domain and check for semantic similarity indicators
    for domain, info in INTENT_TAXONOMY.items():
        intents = info["sub_intents"]
        for i in range(len(intents)):
            for j in range(i + 1, len(intents)):
                a, b = intents[i], intents[j]
                # Check for shared words that might indicate overlap
                words_a = set(a.lower().replace(" & ", " ").split())
                words_b = set(b.lower().replace(" & ", " ").split())
                overlap = words_a & words_b
                if len(overlap) >= 2:
                    issues.append(
                        f"Potential overlap in domain '{domain}': "
                        f"'{a}' and '{b}' share words {overlap}"
                    )
    return issues


def report_coverage(
    sessions_path: str,
    labels_path: str = None,
    cache_path: str = None,
) -> dict:
    """
    Report what % of sessions have valid intents.

    Args:
        sessions_path: Path to session JSONL file.
        labels_path: Path to teacher labels JSONL file.
        cache_path: Path to cached labels JSON file.

    Returns:
        Dict with coverage statistics.
    """
    # Load sessions
    sessions = []
    with open(sessions_path, "r", encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(sessions)} sessions from {sessions_path}")

    session_ids = set()
    for s in sessions:
        # Generate a session key
        uid = s.get("user_id", "")
        target = s.get("target_item", "")
        last_ts = s.get("timestamps", [0])[-1] if s.get("timestamps") else 0
        # Simple session id: user_id + target_item + last_timestamp prefix
        sid = f"{uid}_{target}_{last_ts}"
        s["_session_id"] = sid
        session_ids.add(sid)

    loaded_labels = {}
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                loaded_labels[entry.get("session_id", "")] = entry.get("intent", {})

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        for sid, entry in cache_data.get("labels", {}).items():
            if sid not in loaded_labels:
                loaded_labels[sid] = entry.get("intent_json", {})

    # Check coverage
    sessions_with_labels = 0
    valid_labels = 0
    for s in sessions:
        sid = s.get("_session_id")
        label = loaded_labels.get(sid)
        if label and isinstance(label, dict) and len(label) > 0:
            sessions_with_labels += 1
            is_valid, errors = validate_intent_label(label)
            if is_valid:
                valid_labels += 1
            else:
                logger.debug(f"Invalid label for session {sid}: {errors}")

    coverage = {
        "total_sessions": len(sessions),
        "sessions_with_labels": sessions_with_labels,
        "label_coverage_pct": round(
            sessions_with_labels / len(sessions) * 100, 2
        ) if sessions else 0,
        "valid_labels": valid_labels,
        "valid_label_pct": round(
            valid_labels / sessions_with_labels * 100, 2
        ) if sessions_with_labels else 0,
    }
    logger.info(
        f"Coverage: {coverage['sessions_with_labels']}/{coverage['total_sessions']} "
        f"sessions have labels ({coverage['label_coverage_pct']}%)"
    )
    logger.info(
        f"Valid: {coverage['valid_labels']}/{coverage['sessions_with_labels']} "
        f"({coverage['valid_label_pct']}%)"
    )
    return coverage


def validate_cached_labels(cache_path: str) -> dict:
    """
    Validate all cached labels conform to schema.

    Args:
        cache_path: Path to the teacher label cache JSON.

    Returns:
        Dict with validation results.
    """
    if not os.path.exists(cache_path):
        logger.warning(f"Cache file not found: {cache_path}")
        return {"total": 0, "valid": 0, "invalid": 0, "errors": []}

    with open(cache_path, "r", encoding="utf-8") as f:
        cache_data = json.load(f)

    labels = cache_data.get("labels", {})
    all_errors = []
    valid_count = 0
    invalid_count = 0

    for sid, entry in labels.items():
        intent_json = entry.get("intent_json", {})
        is_valid, errors = validate_intent_label(intent_json)
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            all_errors.append({"session_id": sid, "errors": errors})

    result = {
        "total": len(labels),
        "valid": valid_count,
        "invalid": invalid_count,
        "errors": all_errors[:20],  # Limit error output
        "error_sample_count": len(all_errors),
    }
    logger.info(
        f"Cached labels: {result['valid']} valid, "
        f"{result['invalid']} invalid (out of {result['total']})"
    )
    if all_errors:
        logger.warning(f"First errors: {all_errors[:3]}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check taxonomy stability and label coverage."
    )
    parser.add_argument(
        "--sessions", default=None, help="Path to session JSONL for coverage check"
    )
    parser.add_argument(
        "--labels", default=None, help="Path to teacher labels JSONL"
    )
    parser.add_argument(
        "--cache", default=None, help="Path to cached labels JSON"
    )
    parser.add_argument(
        "--validate-cache", action="store_true",
        help="Validate all cached labels against schema"
    )
    parser.add_argument(
        "--check-taxonomy", action="store_true",
        help="Run taxonomy stability checks"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    all_passed = True

    # Category summary
    summary = get_taxonomy_summary()
    print("\n=== Taxonomy Summary ===")
    for domain, info in summary.items():
        print(f"  {domain}: {info['intent_count']} intents")
    print(f"  Total: {len(ALL_INTENTS)} intent labels across "
          f"{len(INTENT_TAXONOMY)} domains\n")

    # Stability checks
    if args.check_taxonomy:
        print("=== Taxonomy Stability Checks ===")

        # Built-in check from intent_taxonomy
        issues = taxonomy_check()
        if issues:
            print(f"  Found {len(issues)} taxonomy issues:")
            for issue in issues:
                print(f"    - {issue}")
            all_passed = False
        else:
            print("  No taxonomy structure issues found.")

        # Duplicate intent check
        dup_issues = check_duplicate_intents()
        if dup_issues:
            print(f"  Found {len(dup_issues)} duplicate intent(s) across domains:")
            for issue in dup_issues:
                print(f"    - {issue}")
            all_passed = False
        else:
            print("  No duplicate intents across domains.")

        # Overlap check
        overlap_issues = check_overlapping_definitions()
        if overlap_issues:
            print(f"  Found {len(overlap_issues)} potential overlaps:")
            for issue in overlap_issues:
                print(f"    - {issue}")
        else:
            print("  No overlapping intent definitions detected.")
        print()

    # Coverage check
    if args.sessions:
        print("=== Label Coverage ===")
        coverage = report_coverage(
            sessions_path=args.sessions,
            labels_path=args.labels,
            cache_path=args.cache,
        )
        if coverage["label_coverage_pct"] < 50:
            print(f"  WARNING: Low coverage ({coverage['label_coverage_pct']}%)")
        print()

    # Cache validation
    if args.validate_cache and args.cache:
        print("=== Cache Validation ===")
        cache_result = validate_cached_labels(args.cache)
        if cache_result["invalid"] > 0:
            print(
                f"  WARNING: {cache_result['invalid']} invalid labels found "
                f"(out of {cache_result['total']})"
            )
            all_passed = False
        else:
            print(f"  All {cache_result['valid']} cached labels are valid.")
        print()

    if all_passed:
        print("All checks passed.")
    else:
        print("Some checks failed. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
