"""
intent_taxonomy.py - Controlled intent taxonomy schema and validation.

Defines the intent schema: {primary_intent, secondary_intents, confidence, evidence_items}
Provides taxonomy categories, schema validation, and stability checks.
"""

import json
from typing import Dict, List, Optional, Tuple


# ---- Intent Taxonomy Categories ----
# Hierarchical taxonomy: top-level domain -> specific intents
INTENT_TAXONOMY = {
    "Sports & Fitness": {
        "description": "Athletic activities, workout gear, outdoor sports",
        "sub_intents": [
            "running injury prevention",
            "sports recovery",
            "footwear accessory",
            "gym equipment",
            "yoga & pilates",
            "cycling accessories",
            "swimming gear",
            "team sports equipment",
        ],
    },
    "Electronics": {
        "description": "Consumer electronics, gadgets, accessories",
        "sub_intents": [
            "smartphone accessories",
            "laptop & tablet",
            "audio equipment",
            "wearable technology",
            "gaming peripherals",
            "smart home devices",
            "camera & photography",
            "charging & cables",
        ],
    },
    "Fashion": {
        "description": "Clothing, footwear, accessories, personal style",
        "sub_intents": [
            "casual wear",
            "formal attire",
            "seasonal fashion",
            "footwear selection",
            "accessories & jewelry",
            "athleisure",
            "sustainable fashion",
            "plus size fashion",
        ],
    },
    "Home & Kitchen": {
        "description": "Home improvement, kitchenware, furniture, decor",
        "sub_intents": [
            "kitchen appliances",
            "home decor",
            "furniture shopping",
            "cleaning supplies",
            "storage solutions",
            "bedding & bath",
            "gardening tools",
            "home improvement",
        ],
    },
    "Health & Beauty": {
        "description": "Personal care, cosmetics, health products",
        "sub_intents": [
            "skincare routine",
            "hair care",
            "makeup & cosmetics",
            "vitamins & supplements",
            "personal hygiene",
            "fragrance",
            "men's grooming",
            "oral care",
        ],
    },
    "Books & Media": {
        "description": "Books, music, movies, digital content",
        "sub_intents": [
            "fiction reading",
            "non-fiction research",
            "educational materials",
            "music & audio",
            "movie & tv series",
            "comics & graphic novels",
            "self-development",
            "children's books",
        ],
    },
    "Food & Grocery": {
        "description": "Food items, beverages, gourmet products",
        "sub_intents": [
            "snack selection",
            "beverage choice",
            "cooking ingredients",
            "organic food",
            "international cuisine",
            "meal planning",
            "special diet",
            "coffee & tea",
        ],
    },
    "Toys & Games": {
        "description": "Children's toys, board games, puzzles",
        "sub_intents": [
            "educational toys",
            "board games",
            "outdoor play",
            "building & construction",
            "dolls & action figures",
            "puzzles & brain teasers",
            "video games",
            "arts & crafts",
        ],
    },
    "Automotive": {
        "description": "Car parts, accessories, maintenance",
        "sub_intents": [
            "car maintenance",
            "auto parts",
            "car accessories",
            "interior upgrades",
            "exterior care",
            "motorcycle gear",
            "tools & equipment",
            "emergency supplies",
        ],
    },
    "Office & Stationery": {
        "description": "Office supplies, stationery, workspace",
        "sub_intents": [
            "office supplies",
            "writing instruments",
            "paper products",
            "desk organization",
            "printing & scanning",
            "shipping supplies",
            "presentation tools",
            "work from home setup",
        ],
    },
}

# Flat list of all intent labels for quick lookup
ALL_INTENTS = []
for domain, info in INTENT_TAXONOMY.items():
    for intent in info["sub_intents"]:
        ALL_INTENTS.append(intent)

# Intent to domain mapping
INTENT_TO_DOMAIN = {}
for domain, info in INTENT_TAXONOMY.items():
    for intent in info["sub_intents"]:
        INTENT_TO_DOMAIN[intent] = domain


# ---- Intent Schema ----
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_intent": {
            "type": "string",
            "description": "The single most likely intent of the user session",
            "enum": ALL_INTENTS,
        },
        "secondary_intents": {
            "type": "array",
            "description": "Additional plausible intents beyond the primary",
            "items": {"type": "string", "enum": ALL_INTENTS},
            "maxItems": 5,
        },
        "confidence": {
            "type": "number",
            "description": "Model confidence in the primary intent (0.0 to 1.0)",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "evidence_items": {
            "type": "array",
            "description": "Indices (0-based) of items in the session that support this intent",
            "items": {"type": "integer", "minimum": 0},
            "uniqueItems": True,
        },
    },
    "required": ["primary_intent", "confidence"],
}


def validate_intent_label(label: Dict) -> Tuple[bool, List[str]]:
    """
    Validate that an intent label conforms to the taxonomy schema.

    Args:
        label: Dict with keys primary_intent, secondary_intents (optional),
               confidence, evidence_items (optional).

    Returns:
        (is_valid, errors) tuple.
    """
    errors = []

    # Check required fields
    if "primary_intent" not in label:
        errors.append("Missing required field: primary_intent")
    if "confidence" not in label:
        errors.append("Missing required field: confidence")

    if errors:
        return False, errors

    # Validate primary_intent
    primary = label.get("primary_intent", "")
    if not isinstance(primary, str):
        errors.append("primary_intent must be a string")
    elif primary not in ALL_INTENTS:
        errors.append(
            f"primary_intent '{primary}' is not in taxonomy. "
            f"Valid intents: {ALL_INTENTS[:5]}... ({len(ALL_INTENTS)} total)"
        )

    # Validate confidence
    confidence = label.get("confidence", -1)
    if not isinstance(confidence, (int, float)):
        errors.append("confidence must be a number")
    elif confidence < 0.0 or confidence > 1.0:
        errors.append(f"confidence must be between 0.0 and 1.0, got {confidence}")

    # Validate secondary_intents
    secondary = label.get("secondary_intents", [])
    if not isinstance(secondary, list):
        errors.append("secondary_intents must be a list")
    else:
        if len(secondary) > 5:
            errors.append(f"secondary_intents too many ({len(secondary)} > 5)")
        for i, intent in enumerate(secondary):
            if not isinstance(intent, str):
                errors.append(f"secondary_intents[{i}] must be a string")
            elif intent not in ALL_INTENTS:
                errors.append(
                    f"secondary_intents[{i}] '{intent}' is not in taxonomy"
                )
            if intent == primary:
                errors.append(
                    f"secondary_intents[{i}] '{intent}' duplicates primary_intent"
                )
        if len(secondary) != len(set(secondary)):
            errors.append("secondary_intents contains duplicates")

    # Validate evidence_items
    evidence = label.get("evidence_items", [])
    if not isinstance(evidence, list):
        errors.append("evidence_items must be a list")
    elif len(evidence) != len(set(evidence)):
        errors.append("evidence_items contains duplicates")
    else:
        for i, idx in enumerate(evidence):
            if not isinstance(idx, int) or idx < 0:
                errors.append(f"evidence_items[{i}] must be a non-negative integer")

    return len(errors) == 0, errors


def get_intent_domain(intent_name: str) -> Optional[str]:
    """
    Get the top-level domain for a given intent label.
    Returns None if the intent is not found.
    """
    return INTENT_TO_DOMAIN.get(intent_name)


def get_taxonomy_summary() -> Dict:
    """Return a summary of the taxonomy: domains and intent counts."""
    return {
        domain: {
            "description": info["description"],
            "intent_count": len(info["sub_intents"]),
            "intents": info["sub_intents"],
        }
        for domain, info in INTENT_TAXONOMY.items()
    }


def check_taxonomy_stability() -> List[str]:
    """
    Check for stability issues in the taxonomy:
    - No duplicate intent labels across domains
    - Proper hierarchy (all intents have a domain)
    - No overlapping or ambiguous definitions
    """
    issues = []

    # Check for duplicate intent labels across domains
    seen_intents = {}
    for domain, info in INTENT_TAXONOMY.items():
        for intent in info["sub_intents"]:
            if intent in seen_intents:
                issues.append(
                    f"Duplicate intent '{intent}' in domains: "
                    f"'{seen_intents[intent]}' and '{domain}'"
                )
            seen_intents[intent] = domain

    # Check that ALL_INTENTS and INTENT_TO_DOMAIN are consistent
    if len(ALL_INTENTS) != len(seen_intents):
        issues.append(
            f"ALL_INTENTS count ({len(ALL_INTENTS)}) differs from "
            f"unique intents ({len(seen_intents)})"
        )

    # Check that all intents in INTENT_TO_DOMAIN are valid
    for intent, domain in INTENT_TO_DOMAIN.items():
        if domain not in INTENT_TAXONOMY:
            issues.append(f"Intent '{intent}' maps to unknown domain '{domain}'")
        elif intent not in INTENT_TAXONOMY[domain]["sub_intents"]:
            issues.append(
                f"Intent '{intent}' maps to domain '{domain}' but is not "
                f"listed in that domain's sub_intents"
            )

    return issues


def get_system_prompt() -> str:
    """
    Get the system prompt for teacher LLM describing the intent taxonomy.
    """
    domain_lines = []
    for domain, info in INTENT_TAXONOMY.items():
        intents_str = ", ".join(info["sub_intents"])
        domain_lines.append(f"- {domain}: {intents_str}")

    return (
        "You are a user intent analyst for e-commerce recommendations. "
        "Given a user's recent browsing/purchase history, determine their "
        "primary shopping intent and secondary intents.\n\n"
        "Valid intent categories:\n"
        + "\n".join(domain_lines)
        + "\n\n"
        "Respond with a JSON object containing:\n"
        '  - "primary_intent": the single most likely intent (must be from taxonomy)\n'
        '  - "secondary_intents": list of additional plausible intents (0-5 items)\n'
        '  - "confidence": your confidence in the primary intent (0.0 to 1.0)\n'
        '  - "evidence_items": list of item indices (0-based) supporting this intent\n\n'
        "Output ONLY the JSON object, no explanation."
    )
