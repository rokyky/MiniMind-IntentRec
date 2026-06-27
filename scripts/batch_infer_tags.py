import json, os, argparse, torch, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.model_minimind import MiniMind, MiniMindConfig
from model.model_lora import apply_lora

def load_model(model_dir: str, lora_path: str = None, device: str = "cpu"):
    config = MiniMindConfig()
    config.dim = 512
    config.n_layers = 8
    config.n_heads = 8
    config.vocab_size = 6400
    config.max_seq_len = 256

    model = MiniMind(config)
    state = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    if lora_path and os.path.exists(lora_path):
        apply_lora(model, rank=8)
        lora_state = torch.load(lora_path, map_location=device)
        model.load_state_dict(lora_state, strict=False)
        print(f"LoRA loaded: {lora_path}")

    return model

def generate_tags(model, tokenizer, title: str, category: str, desc: str = "", device: str = "cpu") -> str:
    prompt = f"Generate semantic tags for this product.\nTitle: {title}\nCategory: {category}"
    if desc:
        prompt += f"\nDescription: {desc[:200]}"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id or 2,
        )
    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return result.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="metadata.jsonl")
    parser.add_argument("--output", default="./data/tags_output.jsonl")
    parser.add_argument("--model-dir", required=True, help="MiniMind model dir")
    parser.add_argument("--lora-path", default=None, help="LoRA checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    except:
        print("Warning: no HF tokenizer, using simple fallback")
        class SimpleTok:
            def __call__(self, text, **kw):
                class T:
                    input_ids = torch.tensor([[0]])
                    shape = (1,1)
                return T()
            def decode(self, ids, **kw): return ""
            @property
            def eos_token_id(self): return 2
        tokenizer = SimpleTok()

    model = load_model(args.model_dir, args.lora_path, args.device)
    items = [json.loads(line) for line in open(args.input, encoding='utf-8')]

    results = []
    for i, item in enumerate(items):
        tags = generate_tags(model, tokenizer, item.get("title",""), item.get("category",""), item.get("description",""), args.device)
        result = {**item, "generated_tags": tags, "infer_status": "ok"}
        results.append(result)
        if (i+1) % 100 == 0:
            print(f"[{i+1}/{len(items)}]")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Done: {len(results)} items -> {args.output}")

if __name__ == "__main__":
    main()
