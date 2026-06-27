# MiniMind-IntentRec: LLM-Distilled Session Intent for Recommendations

MiniMind-IntentRec is a session intent modeling module for recommendation systems.
It uses an offline LLM teacher to build a controlled intent taxonomy and generate
soft labels, then distills that signal into:

1. **MiniMind LoRA** -- a local, interpretable intent generator (session text -> intent JSON)
2. **MLP Intent Head** -- a low-latency classifier (session embedding -> intent distribution)

The LLM **never runs at serving time**. Only MiniMind (lightweight) and the MLP head
(near-zero latency) serve predictions.

## Project Role

MiniMind-IntentRec is one of three projects in a recommendation research matrix:

| Project | Role |
|---|---|
| [RoTE-TimeRec](https://github.com/) | Temporal modeling, full-ranking eval, trustworthy metrics |
| MiniMind-IntentRec | LLM/MiniMind distillation of user session intent |
| Gryphon-lite | Semantic ID generative recommendation & item-level calibration |

## Intent Schema

Each session is labeled with a structured intent JSON conforming to the taxonomy:

```json
{
  "primary_intent": "running injury prevention",
  "secondary_intents": ["sports recovery", "footwear accessory"],
  "confidence": 0.82,
  "evidence_items": [3, 4, 5]
}
```

## Pipeline Overview

```
User item sequence (IDs, titles, categories, timestamps)
       |
       v
build_session_data.py  -->  session samples (JSONL)
       |
       v
prepare_intent_sft_data.py  +  teacher_label_cache.py
       |
       v
train_intent_lora.py  -->  MiniMind LoRA: session text -> intent JSON
       |
       v
export_session_embeddings.py  +  MLPIntentHead
       |
       v
train_intent_head.py  -->  MLP head: embedding -> intent distribution
       |
       v
intent_feature_exporter.py  -->  downstream ranker features
```

## Quick Start

### 0. Install dependencies

```bash
pip install -r requirements.txt
```

### 1. Build session data

```bash
python scripts/build_session_data.py \
    --metadata ./data/metadata.jsonl \
    --interactions ./data/interactions.jsonl \
    --output ./data/sessions.jsonl \
    --max-session-len 10 --min-session-len 2
```

If no real data is available, the script falls back to generating synthetic demo data.

### 2. Check taxonomy stability

```bash
python scripts/check_taxonomy_stability.py \
    --check-taxonomy \
    --sessions ./data/sessions.jsonl
```

### 3. Prepare SFT data for MiniMind

```bash
python scripts/prepare_intent_sft_data.py \
    --sessions ./data/sessions.jsonl \
    --labels ./data/teacher_labels.jsonl \
    --output-dir ./data/intent_sft \
    --cache ./data/teacher_label_cache.json
```

### 4. Train MiniMind LoRA intent generator

```bash
python scripts/train_intent_lora.py \
    --config ./configs/lora_intent.yaml \
    --lora-name lora_intent \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 1e-4
```

### 5. Evaluate MiniMind student

```bash
python scripts/eval_intent_student.py \
    --sessions ./data/sessions.jsonl \
    --labels ./data/teacher_labels.jsonl \
    --lora-path ./checkpoints/intent_lora/lora_intent.pth \
    --model-path ./model
```

### 6. Export session embeddings

```bash
python scripts/export_session_embeddings.py \
    --sessions ./data/sessions.jsonl \
    --output-dir ./data/embeddings \
    --encoder-type dummy \
    --embedding-dim 64
```

### 7. Train MLP intent head

```bash
python scripts/train_intent_head.py \
    --embeddings ./data/embeddings/session_embeddings.npy \
    --metadata ./data/embeddings/session_embeddings_metadata.jsonl \
    --labels ./data/teacher_labels.jsonl \
    --output-dir ./checkpoints/intent_head \
    --input-dim 64 --hidden-dims 128 64 --epochs 50
```

### 8. Run MLP head inference

```bash
python scripts/infer_intent_head.py \
    --checkpoint ./checkpoints/intent_head/mlp_intent_head.pth \
    --embeddings ./data/embeddings/session_embeddings.npy \
    --mode topk --top-k 5 \
    --output ./output/mlp_predictions.jsonl

# Latency benchmark
python scripts/infer_intent_head.py \
    --checkpoint ./checkpoints/intent_head/mlp_intent_head.pth \
    --benchmark
```

### 9. Export intent features for downstream ranker

```bash
python -c "
from src.intent_feature_exporter import IntentFeatureExporter
import json, numpy as np

# Load predictions
preds = []
with open('./output/mlp_predictions.jsonl') as f:
    for line in f: preds.append(json.loads(line.strip()))

exporter = IntentFeatureExporter()
for pred in preds:
    intents = pred.get('intents', [])
    indices = np.array([[ALL_INTENTS.index(i['name'])] for i in intents])
    values = np.array([[i['confidence']] for i in intents])

features = exporter.from_top_k(indices, values, source_model='mlp_intent_head')
exporter.save_jsonl(features, './output/intent_features.jsonl')
exporter.save_npz(features, './output/intent_features.npz')
"
```

### 10. Compare intent variants

```bash
python scripts/compare_intent_variants.py \
    --sessions ./data/sessions.jsonl \
    --labels ./data/teacher_labels.jsonl \
    --minimind-results ./output/minimind_predictions.jsonl \
    --mlp-results ./output/mlp_predictions.jsonl \
    --output ./eval_results/variant_comparison.json
```

### 11. Downstream evaluation

```bash
python scripts/eval_downstream.py \
    --sessions ./data/sessions.jsonl \
    --labels ./data/teacher_labels.jsonl \
    --minimind-results ./output/minimind_predictions.jsonl \
    --mlp-results ./output/mlp_predictions.jsonl \
    --output ./eval_results/downstream_eval.json \
    --ks 5 10 20 \
    --split-file ./data/split_protocol.json
```

### 12. Slice evaluation

```bash
python scripts/slice_intent_eval.py \
    --sessions ./data/sessions.jsonl \
    --labels ./data/teacher_labels.jsonl \
    --minimind-results ./output/minimind_predictions.jsonl \
    --mlp-results ./output/mlp_predictions.jsonl \
    --output ./eval_results/slice_eval.json \
    --ks 5 10
```

## Evaluation Metrics

### Intent Quality

- **JSON valid rate**: fraction of MiniMind outputs that parse as valid JSON
- **Schema compliance**: fraction that conform to the intent taxonomy schema
- **Exact match**: predicted primary_intent == label primary_intent
- **Semantic match**: primary matches label primary, or appears in secondary list, or same domain
- **Micro/macro F1**: classification metrics for the MLP head
- **Precision@k / Recall@k**: top-k intent retrieval metrics
- **ECE**: Expected Calibration Error (softmax mode)

### Downstream Recommendation

- **HR@K** (Hit Rate): is the target item in top-K?
- **NDCG@K** (Normalized Discounted Cumulative Gain): position-aware
- **Recall@K**: fraction of relevant items retrieved
- **MRR@K** (Mean Reciprocal Rank)

### Slice Evaluation

- **short_history**: users in bottom 33% by session length
- **cold_start**: users with fewer than 5 total interactions
- **session_drift**: category switch in last 3 items

## Results

### Intent Quality (MLP Head)

| Metric | Value |
|--------|-------|
| Micro F1 | _ |
| Macro F1 | _ |
| Precision@5 | _ |
| Recall@5 | _ |
| ECE | _ |

### Downstream Recommendation (shared split)

| Variant | HR@10 | NDCG@10 | Recall@10 |
|---------|-------|---------|-----------|
| No Intent (baseline) | _ | _ | _ |
| Category Majority | _ | _ | _ |
| Cluster | _ | _ | _ |
| Teacher LLM (upper bound) | _ | _ | _ |
| MiniMind LoRA | _ | _ | _ |
| MLP Intent Head | _ | _ | _ |

### Latency

| Model | Avg Latency | P95 Latency |
|-------|-------------|-------------|
| MiniMind LoRA (1 sample) | _ | _ |
| MLP Intent Head (1 sample) | _ | _ |
| MLP Intent Head (batch=64, per sample) | _ | _ |

## Taxonomy

Controlled intent taxonomy with 10 domains and 80 sub-intents:

| Domain | Intents |
|--------|---------|
| Sports & Fitness | 8 |
| Electronics | 8 |
| Fashion | 8 |
| Home & Kitchen | 8 |
| Health & Beauty | 8 |
| Books & Media | 8 |
| Food & Grocery | 8 |
| Toys & Games | 8 |
| Automotive | 8 |
| Office & Stationery | 8 |

## Scope

This project adapts LLM-distilled session intent modeling for Amazon-style
e-commerce behavior data. It does not claim to use production cross-platform
session data from finance, nor does it call an LLM in the online serving path.
