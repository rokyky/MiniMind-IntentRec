# MiniMind-IntentRec：LLM 蒸馏的会话意图推荐模块

MiniMind-IntentRec 是一个面向推荐系统的 session intent modeling 项目。核心思路：用离线 LLM teacher 构建可控意图 taxonomy 并生成软标签，再蒸馏到两条低成本路径：

1. **MiniMind LoRA** — 本地可解释意图生成器（session text → intent JSON）
2. **MLP Intent Head** — 低延迟分类器（session embedding → intent distribution）

LLM **绝不出现在线上 serving 路径**。线上只消费 MiniMind（轻量）或 MLP head（极低延迟）产出的 intent feature。

## 项目定位

MiniMind-IntentRec 是三项目推荐研究矩阵里的 LLM 语义蒸馏项目：

| 项目 | 角色 |
|---|---|
| RoTE-TimeRec | 时间建模、full-ranking 评估、评测可信度 |
| MiniMind-IntentRec | LLM / MiniMind 蒸馏用户会话意图 |
| Gryphon-lite | Semantic ID 生成式推荐与 item-level scoring 校准 |

## Intent Schema 示例

每个 session 标注为结构化 intent JSON，严格符合 taxonomy schema：

```json
{
  "primary_intent": "running injury prevention",
  "secondary_intents": ["sports recovery", "footwear accessory"],
  "confidence": 0.82,
  "evidence_items": [3, 4, 5]
}
```

## 技术路线

```
用户近期 item IDs, titles, categories, timestamps
       │
       ▼
build_session_data.py  ──→  session 样本 (JSONL)
       │
       ▼
prepare_intent_sft_data.py  +  teacher_label_cache.py
       │
       ▼
train_intent_lora.py  ──→  MiniMind LoRA: session text → intent JSON
       │
       ▼
export_session_embeddings.py  +  MLPIntentHead
       │
       ▼
train_intent_head.py  ──→  MLP head: embedding → intent distribution
       │
       ▼
intent_feature_exporter.py  ──→  下游 ranker 消费的 intent features
```

## 模型角色

| 组件 | 作用 | serving 成本 |
|---|---|---|
| LLM teacher | 离线构建 taxonomy 和标注 session | 仅离线 |
| MiniMind LoRA | 本地可解释 intent generator | 低到中 |
| MLP intent head | session embedding → intent distribution | 极低 |
| RoTE-TimeRec / ranker | 消费 intent feature 做推荐 | 复用推荐链路 |

## 快速开始

### 0. 安装依赖

```bash
pip install -r requirements.txt
```

### 1. 构造 session 数据

```bash
python scripts/build_session_data.py \
    --metadata ./data/metadata.jsonl \
    --interactions ./data/interactions.jsonl \
    --output ./data/sessions.jsonl \
    --max-session-len 10 --min-session-len 2
```

若无真实数据，脚本自动生成合成 demo 数据。

### 2. 检查 taxonomy 稳定性

```bash
python scripts/check_taxonomy_stability.py \
    --check-taxonomy \
    --sessions ./data/sessions.jsonl
```

### 3. 准备 MiniMind SFT 数据

```bash
python scripts/prepare_intent_sft_data.py \
    --sessions ./data/sessions.jsonl \
    --labels ./data/teacher_labels.jsonl \
    --output-dir ./data/intent_sft \
    --cache ./data/teacher_label_cache.json
```

### 4. 训练 MiniMind LoRA 意图生成器

```bash
python scripts/train_intent_lora.py \
    --config ./configs/lora_intent.yaml \
    --lora-name lora_intent \
    --epochs 5 \
    --batch-size 16 \
    --learning-rate 1e-4
```

### 5. 评估 MiniMind student

```bash
python scripts/eval_intent_student.py \
    --sessions ./data/sessions.jsonl \
    --labels ./data/teacher_labels.jsonl \
    --lora-path ./checkpoints/intent_lora/lora_intent.pth \
    --model-path ./model
```

### 6. 导出 session embeddings

```bash
python scripts/export_session_embeddings.py \
    --sessions ./data/sessions.jsonl \
    --output-dir ./data/embeddings \
    --encoder-type dummy \
    --embedding-dim 64
```

### 7. 训练 MLP intent head

```bash
python scripts/train_intent_head.py \
    --embeddings ./data/embeddings/session_embeddings.npy \
    --metadata ./data/embeddings/session_embeddings_metadata.jsonl \
    --labels ./data/teacher_labels.jsonl \
    --output-dir ./checkpoints/intent_head \
    --input-dim 64 --hidden-dims 128 64 --epochs 50
```

### 8. MLP head 推理

```bash
# Top-k 推理
python scripts/infer_intent_head.py \
    --checkpoint ./checkpoints/intent_head/mlp_intent_head.pth \
    --embeddings ./data/embeddings/session_embeddings.npy \
    --mode topk --top-k 5 \
    --output ./output/mlp_predictions.jsonl

# 延迟基准测试
python scripts/infer_intent_head.py \
    --checkpoint ./checkpoints/intent_head/mlp_intent_head.pth \
    --benchmark
```

### 9. 导出 intent feature 给下游 ranker

```bash
python -c "
from src.intent_feature_exporter import IntentFeatureExporter
import json, numpy as np

preds = []
with open('./output/mlp_predictions.jsonl') as f:
    for line in f:
        preds.append(json.loads(line.strip()))

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

### 10. 对比 intent 变体

```bash
python scripts/compare_intent_variants.py \
    --sessions ./data/sessions.jsonl \
    --labels ./data/teacher_labels.jsonl \
    --minimind-results ./output/minimind_predictions.jsonl \
    --mlp-results ./output/mlp_predictions.jsonl \
    --output ./eval_results/variant_comparison.json
```

### 11. 下游推荐评估

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

### 12. 切片评估

```bash
python scripts/slice_intent_eval.py \
    --sessions ./data/sessions.jsonl \
    --labels ./data/teacher_labels.jsonl \
    --minimind-results ./output/minimind_predictions.jsonl \
    --mlp-results ./output/mlp_predictions.jsonl \
    --output ./eval_results/slice_eval.json \
    --ks 5 10
```

## 评估体系

### 意图质量指标

- **JSON valid rate**：MiniMind 输出可解析为合法 JSON 的比例
- **Schema compliance**：输出符合 intent taxonomy schema 的比例
- **Exact match**：预测 primary_intent 与标签完全一致
- **Semantic match**：primary 匹配标签 primary、出现在 secondary 中，或属于同一 domain
- **Micro/macro F1**：MLP head 分类指标
- **Precision@k / Recall@k**：top-k intent 检索指标
- **ECE**：Expected Calibration Error（softmax 模式）

### 下游推荐指标

- **HR@K**（Hit Rate）：目标 item 是否在 top-K 中
- **NDCG@K**（Normalized Discounted Cumulative Gain）：位置感知排序质量
- **Recall@K**：相关 item 的检出比例
- **MRR@K**（Mean Reciprocal Rank）

### 切片评估

- **short_history**：session 长度位于 bottom 33% 的用户
- **cold_start**：总交互少于 5 次的用户
- **session_drift**：最近 3 个 item 发生类目切换的 session

## 当前边界与必须补的实验

当前代码层面已经覆盖 session 序列化、受控 taxonomy、teacher label cache、MiniMind LoRA 配置、MLP intent head、intent feature 导出和下游对比脚本。真正的风险在于 teacher label 质量和下游推荐提升是否来自真实 ranker，而不是模拟排序。

### 已解决的代码级风险

- Taxonomy 含 10 个 domain、80 个 sub-intent，并提供 schema validation 和 stability check。
- Teacher label cache 支持版本检查、复用和强制刷新。
- MLP intent head 支持 top-k、threshold、raw logits 和 softmax/sigmoid 两种输出模式。
- Intent features 可以导出为 JSONL / NPZ，供下游 ranker 消费。

### 当前实验硬伤

- 需要提供 teacher label 样例、schema compliance、JSON valid rate 和人工可解释案例。
- 需要说明 10 domain / 80 sub-intent 的设计来源，避免看起来像随意手写标签。
- 下游推荐提升如果只来自 `eval_downstream.py` 的模拟 ranking，不能当作强结论；必须接入 RoTE-TimeRec 或真实 ranker 特征后再给主结果。
- MiniMind LoRA 如果没有真实训练日志和 held-out 评估，只能作为设计路线；不能声称已完成有效蒸馏。
- 需要统一 split、seed 和 teacher label 版本，否则 teacher / MiniMind / MLP 对比不公平。

### 面试叙事边界

推荐表述：这是一个”离线 LLM teacher 到轻量 intent student 的蒸馏模块”，线上不调用大模型，只消费 MiniMind 或 MLP 产出的 intent feature。重点是把不可控文本意图压缩成可控 taxonomy 和低延迟特征。

## 算力估算与实验建议

### 最低配置

单卡 **RTX 4090（24GB）** 足够跑完整实验。A100 只在本地跑 7B teacher 推理或赶时间时短租即可。

### 资源估算

| 版本 | 实验范围 | 4090 单卡 | A100 单卡 |
|------|---------|----------|----------|
| 最小闭环 | 2k–5k sessions，少量 teacher labels，MiniMind LoRA 小跑，MLP intent head | 4–10 h | 2–6 h |
| **可投递可信版** | 10k–30k sessions，teacher/MiniMind/MLP 对比，intent F1/ECE/valid rate，下游 HR/NDCG | **15–35 h** | **8–22 h** |
| 完整实验 | 50k–100k sessions，多 prompt、多 taxonomy、多 student、多 seed | 40–90 h | 25–60 h |

### 隐藏成本（不是 GPU，是标签质量）

```
taxonomy 是否稳定
intent 是否过细/过泛
JSON 是否合法
teacher label 是否泄漏 target item
soft label confidence 是否可信
intent F1 是否能解释
```

算力不应该烧在大规模训练，而应该烧在：**小样本高质量 teacher label → 标签审计 → intent slice 分析 → MiniMind vs MLP latency 对比**。

### 必须跑的实验

```
no intent（基线）
category-majority intent（启发式对照）
LLM teacher intent（上界）
MiniMind LoRA intent
MLP intent head（蒸馏后低延迟路径）
JSON valid rate / intent F1 / ECE / p95 latency
short-history / cold-start / session-drift 切片
```

### 可以砍的实验

- 大规模 teacher label（全部 session 都标）→ 抽样 10–30%
- 本地跑 7B teacher → API 或小样本离线 label 更划算
- MiniMind 多尺寸对比 → 固定 1.1B 或 0.5B
- 多 prompt 大规模 sweep → 固定 prompt template

### 建议跑法

1. **小样本标签验证**（2–5 h）：确认 taxonomy 稳定、teacher label 可用
2. **主实验**（10–30 h）：10k–30k sessions × MiniMind LoRA + MLP head + 下游对比
3. **切片分析**（2–5 h）：short-history、cold-start、session-drift

不推荐表述：不要把它说成完整线上 LLM 推荐系统；没有真实 ranker 实验前，也不要声称 intent 一定提升推荐指标。

## 结果（占位）

### 意图质量（MLP Head）

| 指标 | 值 |
|------|-----|
| Micro F1 | _ |
| Macro F1 | _ |
| Precision@5 | _ |
| Recall@5 | _ |
| ECE | _ |

### 下游推荐（共享 split）

| 变体 | HR@10 | NDCG@10 | Recall@10 |
|------|-------|---------|-----------|
| 无 Intent（基线） | _ | _ | _ |
| 类目多数投票 | _ | _ | _ |
| 聚类 Intent | _ | _ | _ |
| Teacher LLM（上界） | _ | _ | _ |
| MiniMind LoRA | _ | _ | _ |
| MLP Intent Head | _ | _ | _ |

### 延迟

| 模型 | 平均延迟 | P95 延迟 |
|------|---------|---------|
| MiniMind LoRA（单样本） | _ | _ |
| MLP Intent Head（单样本） | _ | _ |
| MLP Intent Head（batch=64, 每样本） | _ | _ |

## Taxonomy 总览

受控 intent taxonomy，共 10 个 domain、80 个 sub-intent：

| Domain / 领域 | 子意图数 |
|------|---------|
| Sports & Fitness / 运动健身 | 8 |
| Electronics / 电子产品 | 8 |
| Fashion / 时尚 | 8 |
| Home & Kitchen / 家居厨房 | 8 |
| Health & Beauty / 健康美容 | 8 |
| Books & Media / 图书媒体 | 8 |
| Food & Grocery / 食品杂货 | 8 |
| Toys & Games / 玩具游戏 | 8 |
| Automotive / 汽车用品 | 8 |
| Office & Stationery / 办公文具 | 8 |

## 范围

本项目在 Amazon 风格电商行为数据上适配 LLM-distilled session intent modeling，不声称使用金融跨平台生产 session 数据，也不在线上推荐链路调用 LLM。
