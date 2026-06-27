"""
eval_intent_student.py - 评估 MiniMind 意图生成器。

报告：
  - JSON 有效比例（输出是否为可解析的 JSON？）
  - 模式合规性（是否匹配意图分类体系模式？）
  - 与教师标签的意图匹配准确率（精确 + 语义）
  - 推理延迟（平均、p95）
"""

import json
import os
import sys
import time
import argparse
import logging
import torch
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.intent_taxonomy import validate_intent_label, ALL_INTENTS
from src.session_serializer import serialize_session
from src.intent_taxonomy import get_system_prompt
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_prompt(session: dict) -> str:
    """为会话构建生成提示。"""
    titles = session.get("item_titles", [])
    categories = session.get("categories", [])
    timestamps = session.get("timestamps", None)
    session_text = serialize_session(titles, categories, timestamps)
    return (
        f"{session_text}\n\n"
        "Based on the above browsing/purchase history, what is the user's "
        "primary shopping intent and any secondary intents? "
        "Respond with a JSON object including primary_intent, "
        "secondary_intents, confidence, and evidence_items."
    )


def load_model(
    model_path: str,
    lora_path: str = None,
    hidden_size: int = 768,
    num_hidden_layers: int = 8,
    use_moe: bool = False,
    device: str = "cpu",
):
    """加载 MiniMind 模型及可选的 LoRA 权重。"""
    config = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        use_moe=use_moe,
    )
    model = MiniMindForCausalLM(config)
    # 加载基础权重
    moe_suffix = "_moe" if use_moe else ""
    weight_path = os.path.join(model_path, f"../out/full_sft_{hidden_size}{moe_suffix}.pth")
    if os.path.exists(weight_path):
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)
        logger.info(f"Loaded base weights from {weight_path}")
    else:
        logger.warning(f"Base weights not found at {weight_path}, using random init")

    # 应用并加载 LoRA
    if lora_path and os.path.exists(lora_path):
        apply_lora(model)
        load_lora(model, lora_path)
        logger.info(f"Loaded LoRA weights from {lora_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model.half().eval().to(device), tokenizer


def extract_intent_from_response(response: str) -> Tuple[bool, dict, str]:
    """
    尝试从模型响应中解析 JSON。

    返回 (parse_success, intent_dict, raw_response)。
    """
    # 尝试从响应中提取 JSON
    response = response.strip()
    # 如果存在 Markdown 代码围栏则移除
    if "```json" in response:
        response = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()

    try:
        intent = json.loads(response)
        return True, intent, response
    except json.JSONDecodeError:
        # 尝试查找类似 JSON 的子字符串
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                intent = json.loads(response[start:end + 1])
                return True, intent, response
            except json.JSONDecodeError:
                pass
        return False, {}, response


def compute_exact_match(pred_intent: dict, label_intent: dict) -> bool:
    """检查预测的 primary_intent 是否与标签完全匹配。"""
    return pred_intent.get("primary_intent") == label_intent.get("primary_intent")


def compute_semantic_match(pred_intent: dict, label_intent: dict) -> bool:
    """
    检查预测的 primary_intent 是否语义有效：
    精确匹配，或者标签的主要意图出现在
    预测的次要意图中，反之亦然。
    """
    pred_primary = pred_intent.get("primary_intent", "")
    label_primary = label_intent.get("primary_intent", "")
    pred_secondary = set(pred_intent.get("secondary_intents", []))
    label_secondary = set(label_intent.get("secondary_intents", []))

    if pred_primary == label_primary:
        return True
    if pred_primary in label_secondary:
        return True
    if label_primary in pred_secondary:
        return True
    # 检查领域级别匹配
    from src.intent_taxonomy import INTENT_TO_DOMAIN
    pred_domain = INTENT_TO_DOMAIN.get(pred_primary, "")
    label_domain = INTENT_TO_DOMAIN.get(label_primary, "")
    if pred_domain and pred_domain == label_domain:
        return True

    return False


def evaluate(
    model,
    tokenizer,
    sessions: List[Dict],
    labels: Dict[str, dict],
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    device: str = "cpu",
    system_prompt: str = None,
) -> Dict:
    """
    在一组带有真实标签的会话上运行评估。
    """
    if system_prompt is None:
        system_prompt = get_system_prompt()

    results = {
        "total": 0,
        "json_valid": 0,
        "schema_compliant": 0,
        "exact_match": 0,
        "semantic_match": 0,
        "latencies": [],
        "errors": [],
    }

    # 构建用于查找的会话 ID
    for s in sessions:
        uid = s.get("user_id", "")
        target = s.get("target_item", "")
        last_ts = s.get("timestamps", [0])[-1] if s.get("timestamps") else 0
        s["_session_id"] = f"{uid}_{target}_{last_ts}"

    for i, session in enumerate(sessions):
        sid = session.get("_session_id", "")
        label = labels.get(sid) or labels.get(session.get("user_id", ""))
        if not label:
            continue

        prompt = build_prompt(session)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)

        # 生成
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 0.01),
                pad_token_id=tokenizer.pad_token_id or 0,
                eos_token_id=tokenizer.eos_token_id or 2,
            )
        elapsed = time.time() - start_time

        response = tokenizer.decode(
            generated_ids[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True
        )

        results["total"] += 1
        results["latencies"].append(elapsed)

        # 解析 JSON
        parse_ok, intent, raw_response = extract_intent_from_response(response)
        if parse_ok:
            results["json_valid"] += 1

            # 模式合规性
            is_valid, _ = validate_intent_label(intent)
            if is_valid:
                results["schema_compliant"] += 1

            # 意图匹配
            if compute_exact_match(intent, label):
                results["exact_match"] += 1
            if compute_semantic_match(intent, label):
                results["semantic_match"] += 1
        else:
            results["errors"].append({
                "session_id": sid,
                "response_preview": response[:200],
            })

        if (i + 1) % 50 == 0:
            logger.info(f"  Evaluated {i+1}/{len(sessions)} samples")

    return results


def print_results(results: Dict):
    """美观地打印评估结果。"""
    total = results["total"]
    print("\n" + "=" * 60)
    print("  MiniMind Intent Student Evaluation Report")
    print("=" * 60)

    if total == 0:
        print("  No samples evaluated!")
        return

    json_rate = results["json_valid"] / total * 100
    schema_rate = results["schema_compliant"] / total * 100
    exact_rate = results["exact_match"] / total * 100
    semantic_rate = results["semantic_match"] / total * 100

    print(f"\n  Total Samples:       {total}")
    print(f"  JSON Valid Rate:     {json_rate:.2f}% ({results['json_valid']}/{total})")
    print(f"  Schema Compliance:   {schema_rate:.2f}% ({results['schema_compliant']}/{total})")
    print(f"  Exact Match (P):     {exact_rate:.2f}% ({results['exact_match']}/{total})")
    print(f"  Semantic Match:      {semantic_rate:.2f}% ({results['semantic_match']}/{total})")

    # 延迟
    latencies = results["latencies"]
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        p95_lat = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]
        print(f"\n  Latency (avg):        {avg_lat:.4f}s")
        print(f"  Latency (p95):        {p95_lat:.4f}s")

    # 错误
    if results["errors"]:
        print(f"\n  Parse Errors:         {len(results['errors'])}")
        for e in results["errors"][:5]:
            print(f"    - {e['session_id']}: {e['response_preview'][:80]}...")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MiniMind intent generator."
    )
    parser.add_argument(
        "--sessions", required=True, help="Path to session JSONL"
    )
    parser.add_argument(
        "--labels", required=True, help="Path to teacher labels JSONL"
    )
    parser.add_argument(
        "--model-path", default="./model", help="MiniMind model directory"
    )
    parser.add_argument(
        "--lora-path", default=None, help="Trained LoRA checkpoint path"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=768, help="Model hidden size"
    )
    parser.add_argument(
        "--num-hidden-layers", type=int, default=8, help="Number of hidden layers"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256, help="Max generation tokens"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Generation temperature"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device"
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Max samples to evaluate (0=all)"
    )
    args = parser.parse_args()

    logger.info("Loading sessions and labels...")
    sessions = []
    with open(args.sessions, "r", encoding="utf-8") as f:
        for line in f:
            sessions.append(json.loads(line.strip()))
    if args.max_samples > 0:
        sessions = sessions[:args.max_samples]
    logger.info(f"Loaded {len(sessions)} sessions")

    labels = {}
    with open(args.labels, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            labels[entry.get("session_id", "")] = entry.get("intent", {})
    logger.info(f"Loaded {len(labels)} labels")

    logger.info("Loading model...")
    model, tokenizer = load_model(
        model_path=args.model_path,
        lora_path=args.lora_path,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        device=args.device,
    )

    logger.info("Starting evaluation...")
    results = evaluate(
        model=model,
        tokenizer=tokenizer,
        sessions=sessions,
        labels=labels,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )

    print_results(results)

    # Save results
    output_path = "./eval_results/intent_student_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert latencies to summary stats for saving
    save_results = {k: v for k, v in results.items() if k != "latencies"}
    save_results["latency_summary"] = {
        "avg": sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0,
        "p95": sorted(results["latencies"])[
            min(int(len(results["latencies"]) * 0.95), len(results["latencies"]) - 1)
        ] if results["latencies"] else 0,
        "count": len(results["latencies"]),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
