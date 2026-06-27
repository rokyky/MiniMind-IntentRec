import json, os, argparse, random, copy

def build_conversation(item: dict) -> dict:
    """Convert tagged item into chat format for MiniMind SFTDataset."""
    title = item.get("title", "")
    category = item.get("category", "")
    desc = item.get("description", "")
    tags = item.get("tags", {})

    user_content = f"Generate semantic tags for this product.\nTitle: {title}\nCategory: {category}"
    if desc:
        user_content += f"\nDescription: {desc[:300]}"
    assistant_content = json.dumps(tags, ensure_ascii=False) if tags else "{}"

    return {
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def convert(input_path: str, output_dir: str, val_ratio: float = 0.05):
    items = []
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item.get("tag_status") == "ok" and item.get("tags"):
                items.append(item)

    print(f"Valid tagged items: {len(items)}")
    random.shuffle(items)
    split = int(len(items) * (1 - val_ratio))
    os.makedirs(output_dir, exist_ok=True)

    for name, data in [("train", items[:split]), ("val", items[split:])]:
        path = os.path.join(output_dir, f"{name}.jsonl")
        with open(path, "w", encoding='utf-8') as f:
            for item in data:
                conv = build_conversation(item)
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)} samples -> {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./data/tag_sft")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()
    convert(args.input, args.output_dir, args.val_ratio)
