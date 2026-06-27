## 1. Session 数据与 Taxonomy

- [x] 1.1 从用户近期 item 序列构造 session 样本。
- [x] 1.2 序列化 session text，包含 title、category 和 timestamp。
- [x] 1.3 定义 controlled intent taxonomy schema。
- [x] 1.4 增加离线 teacher label 缓存格式。
- [x] 1.5 增加 taxonomy stability 检查。

## 2. MiniMind Intent Student

- [x] 2.1 准备 SFT 数据：session text -> structured intent JSON。
- [x] 2.2 增加 MiniMind LoRA intent generation 配置。
- [x] 2.3 训练和评估 MiniMind intent generator。
- [x] 2.4 报告 JSON valid rate、schema compliance、intent match 和 latency。

## 3. Embedding Intent Head

- [x] 3.1 从 SASRec / RoTE encoder 导出 session embeddings。
- [x] 3.2 训练 MLP intent classifier。
- [x] 3.3 报告 micro/macro F1、precision@k、recall@k 和 calibration/ECE。
- [x] 3.4 增加 threshold 和 top-k inference 模式。

## 4. 下游推荐

- [x] 4.1 导出 intent features 给 ranker 消费。
- [x] 4.2 对比 no intent、category-majority、cluster、teacher、MiniMind、MLP intent。
- [x] 4.3 在共享 split 下评估 HR/NDCG/Recall。
- [x] 4.4 增加 short-history、cold-start、session-drift 切片。
- [x] 4.5 更新 README 和 quickstart。
