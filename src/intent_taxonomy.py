"""
intent_taxonomy.py - 受控意图分类体系（taxonomy）模式与验证。

定义意图模式：{primary_intent, secondary_intents, confidence, evidence_items}
提供分类类别、模式验证和稳定性检查。
"""

import json
from typing import Dict, List, Optional, Tuple


# ---- 意图分类类别 ----
# 层次化分类：顶层领域 -> 具体意图
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

# 所有意图标签的扁平列表，用于快速查找
ALL_INTENTS = []
for domain, info in INTENT_TAXONOMY.items():
    for intent in info["sub_intents"]:
        ALL_INTENTS.append(intent)

# 意图到领域的映射
INTENT_TO_DOMAIN = {}
for domain, info in INTENT_TAXONOMY.items():
    for intent in info["sub_intents"]:
        INTENT_TO_DOMAIN[intent] = domain


# ---- 意图模式 ----
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_intent": {
            "type": "string",
            "description": "用户会话最可能的单个意图",
            "enum": ALL_INTENTS,
        },
        "secondary_intents": {
            "type": "array",
            "description": "除主要意图外的其他可能意图",
            "items": {"type": "string", "enum": ALL_INTENTS},
            "maxItems": 5,
        },
        "confidence": {
            "type": "number",
            "description": "模型对主要意图的置信度（0.0 到 1.0）",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "evidence_items": {
            "type": "array",
            "description": "会话中支持该意图的物品索引（从0开始）",
            "items": {"type": "integer", "minimum": 0},
            "uniqueItems": True,
        },
    },
    "required": ["primary_intent", "confidence"],
}


def validate_intent_label(label: Dict) -> Tuple[bool, List[str]]:
    """
    验证意图标签是否符合分类体系（taxonomy）模式。

    Args:
        label: 包含键 primary_intent, secondary_intents (可选),
               confidence, evidence_items (可选) 的字典。

    Returns:
        (is_valid, errors) 元组。
    """
    errors = []

    # 检查必填字段
    if "primary_intent" not in label:
        errors.append("Missing required field: primary_intent")
    if "confidence" not in label:
        errors.append("Missing required field: confidence")

    if errors:
        return False, errors

    # 验证 primary_intent
    primary = label.get("primary_intent", "")
    if not isinstance(primary, str):
        errors.append("primary_intent must be a string")
    elif primary not in ALL_INTENTS:
        errors.append(
            f"primary_intent '{primary}' is not in taxonomy. "
            f"Valid intents: {ALL_INTENTS[:5]}... ({len(ALL_INTENTS)} total)"
        )

    # 验证 confidence
    confidence = label.get("confidence", -1)
    if not isinstance(confidence, (int, float)):
        errors.append("confidence must be a number")
    elif confidence < 0.0 or confidence > 1.0:
        errors.append(f"confidence must be between 0.0 and 1.0, got {confidence}")

    # 验证 secondary_intents
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

    # 验证 evidence_items
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
    获取给定意图标签的顶层领域。
    如果未找到该意图则返回 None。
    """
    return INTENT_TO_DOMAIN.get(intent_name)


def get_taxonomy_summary() -> Dict:
    """返回分类体系摘要：领域和意图数量。"""
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
    检查分类体系的稳定性问题：
    - 不同领域之间没有重复的意图标签
    - 层次结构正确（所有意图都有所属领域）
    - 没有重叠或模糊的定义
    """
    issues = []

    # 检查不同领域之间的重复意图标签
    seen_intents = {}
    for domain, info in INTENT_TAXONOMY.items():
        for intent in info["sub_intents"]:
            if intent in seen_intents:
                issues.append(
                    f"Duplicate intent '{intent}' in domains: "
                    f"'{seen_intents[intent]}' and '{domain}'"
                )
            seen_intents[intent] = domain

    # 检查 ALL_INTENTS 和 INTENT_TO_DOMAIN 的一致性
    if len(ALL_INTENTS) != len(seen_intents):
        issues.append(
            f"ALL_INTENTS count ({len(ALL_INTENTS)}) differs from "
            f"unique intents ({len(seen_intents)})"
        )

    # 检查 INTENT_TO_DOMAIN 中的所有意图是否有效
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
    获取描述意图分类体系的教师 LLM 系统提示词。
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
