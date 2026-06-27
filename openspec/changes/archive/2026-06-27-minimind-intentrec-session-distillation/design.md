## 总体设计

MiniMind-IntentRec 包含三个模型角色：

- 离线 LLM teacher：构建 taxonomy，并标注有限 session。
- MiniMind LoRA student：本地生成可解释的结构化 intent。
- MLP intent head：从 sequence encoder 的 session embedding 预测 intent distribution，用于低延迟 serving 风格推理。

## 数据设计

从用户历史构造 session 样本：

```text
recent item titles/categories + optional timestamps
-> session text
-> target next item
```

每条样本保存：

- user id
- item ids
- item titles/categories
- timestamps
- target item
- split id

## Intent Taxonomy 与 Teacher Labels

teacher 阶段输出受控 schema，例如：

```json
{
  "primary_intent": "sports recovery",
  "secondary_intents": ["injury prevention", "running accessories"],
  "confidence": 0.82,
  "evidence_items": [3, 4, 5]
}
```

taxonomy generation 和 labeling 必须缓存，避免重复 API 成本。

## MiniMind LoRA Student

MiniMind 训练目标：

```text
session text -> structured intent JSON
```

评估指标：

- JSON valid rate
- schema compliance
- intent exact / semantic match
- inference latency

## Embedding Intent Head

训练轻量分类器：

```text
session_embedding -> intent distribution
```

session embedding 可以来自 SASRec、RoTE-TimeRec 或简单 Transformer encoder。MLP head 是更接近线上服务的路径。

## 下游集成

intent feature 支持三种形式：

- one-hot / multi-hot intent ids
- intent distribution vector
- top-k intent 文本标签用于分析

下游对比：

- no intent
- category-majority intent
- cluster intent
- teacher LLM intent
- MiniMind-generated intent
- MLP distilled intent

## 风险

- intent label 有主观性，需要控制 taxonomy 并报告 calibration。
- MiniMind 推理可能比 MLP 慢，因此 MiniMind 作为可解释本地 student / 离线增强，MLP head 作为 serving 路径。
- Amazon session 不是金融跨平台 session，项目表述应为对 LLM-distilled intent modeling 的商品推荐适配。
