import os, json, argparse, time, requests
from typing import Dict, List, Optional

SYSTEM_PROMPT = """You are a product semantic tagger for e-commerce recommendation.
Given a product title and category, output a JSON object with these fields:
- function: list of product functionalities (3-5 items)
- attributes: list of product attributes (2-4 items)
- scenario: list of usage scenarios (1-3 items)
- target_user: list of target user types (1-3 items)
- purchase_intent: list of purchase reasons (1-3 items)
Reply with JSON only, no explanation."""

def build_user_prompt(item: Dict) -> str:
    text = f"Title: {item.get('title', '')}\nCategory: {item.get('category', '')}"
    desc = item.get('description', '')
    if desc:
        text += f"\nDescription: {desc[:300]}"
    return text

def call_api(messages: List, api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com") -> Optional[str]:
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 256},
            timeout=30
        )
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"API error: {e}")
        return None

def process_item(item: Dict, api_key: str, model: str, base_url: str) -> Dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(item)}
    ]
    result = call_api(messages, api_key, model, base_url)
    if not result:
        return {**item, "tags": {}, "tag_status": "failed"}
    try:
        tags = json.loads(result)
        return {**item, "tags": tags, "tag_status": "ok"}
    except json.JSONDecodeError:
        return {**item, "tags": {"raw": result}, "tag_status": "parse_failed"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="metadata.jsonl path")
    parser.add_argument("--output", default="./data/tagged.jsonl")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--max-items", type=int, default=0, help="0 = all")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between calls")
    args = parser.parse_args()

    items = []
    with open(args.input, encoding='utf-8') as f:
        for line in f:
            items.append(json.loads(line))
    if args.max_items > 0:
        items = items[:args.max_items]

    results = []
    for i, item in enumerate(items):
        print(f"[{i+1}/{len(items)}] {item.get('asin', '')[:20]}...", end=" ")
        result = process_item(item, args.api_key, args.model, args.base_url)
        results.append(result)
        print(result.get("tag_status", "unknown"))
        if i < len(items) - 1:
            time.sleep(args.delay)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Done: {len(results)} items -> {args.output}")
    ok = sum(1 for r in results if r.get("tag_status") == "ok")
    print(f"OK: {ok}, Failed: {len(results)-ok}")

if __name__ == "__main__":
    main()
