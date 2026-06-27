## ADDED Requirements

### Requirement: SHALL 支持 MiniMind LoRA 意图生成
系统 SHALL 使用 LoRA 微调 MiniMind，使其能生成结构化 session intent 输出。

#### Scenario: 生成结构化 intent JSON
- **When** 输入用户 session text prompt
- **Then** MiniMind intent inference 返回符合 schema 的 JSON
- **And** 预测 intent 属于 controlled taxonomy

### Requirement: SHALL 报告意图生成质量指标
系统 SHALL 将 MiniMind student 和 teacher labels 对齐评估。

#### Scenario: 评估 MiniMind intent student
- **When** 输入 held-out session prompts 和 teacher labels
- **Then** 输出 JSON valid rate、schema compliance、intent match 和 inference latency
