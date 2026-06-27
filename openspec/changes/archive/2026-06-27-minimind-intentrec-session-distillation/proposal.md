## 为什么需要这个变更

MiniMind-IntentRec 将原本的 item tag 生成叙事升级为 session intent 蒸馏。item tag 回答“这个商品是什么”，而推荐系统还需要回答“用户此刻想做什么”。

本项目使用离线 LLM teacher 构建意图 taxonomy 并标注 session，再将信号蒸馏到 MiniMind LoRA 和低延迟 MLP intent head。这样既保留 MiniMind 的本地小模型价值，也避免在线 LLM 成本。

## 改动内容

- 从用户近期行为序列构造 session 样本。
- 使用离线 LLM teacher 生成 controlled session intent taxonomy 和 soft labels。
- 使用 MiniMind LoRA 学习 `session text -> structured intent JSON`。
- 从 SASRec / RoTE session embedding 训练 MLP intent head。
- 导出 intent feature 给下游 ranker，并评估短历史、冷启动和意图漂移切片。

## 不做什么

- 不在线上调用 LLM。
- 不声称使用金融跨平台生产数据。
- 不做 MiniMind 全量预训练。
- 不做大规模 RLHF / GRPO。
- 不把 Agentic item descriptor 作为主路线。

## 验收标准

- 能生成并缓存 session-to-intent 数据。
- MiniMind LoRA 能输出符合 schema 的结构化 intent JSON。
- MLP intent head 能从 session embedding 预测 intent distribution。
- intent feature 能被 RoTE-TimeRec 或其他 ranker 消费。
- 评估报告包含 intent F1、calibration、下游 HR/NDCG、切片指标和延迟。
