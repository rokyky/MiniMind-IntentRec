## Context

Mini-DeepLLM has ~4400 lines of model code: multiple architectures (llama3, deepseekv3, qwen3, deepseekv4),
pretrain/SFT/DPO/GRPO training pipelines, tokenizer, data processing.

MiniMind-Rec reuses the SFT + LoRA training pipeline for recommendation-specific semantic tasks.

## Goals / Non-Goals

Goals:
- Create text-based item semantic tagging SFT dataset
- LoRA fine-tune small LLM (~100M params) for tag generation
- Run batch inference on all items
- Evaluate cold-start recall improvement

Non-Goals:
- Training a chat LLM
- Serving infrastructure

## Design Decisions

D1: Use smallest Mini-DeepLLM model (~100M params) for inference speed
D2: LoRA rank=8, target attention layers only
D3: SFT data: (title+category) -> (tags). Tags from LLM API as pseudo-labels.
D4: Evaluated by: tag quality (BLEU) + downstream recall (cold-start items with text-only features)

## Risks
- Pseudo-label quality depends on LLM API -> manual check 100 samples
- Small model may not generate useful tags -> can scale up if needed