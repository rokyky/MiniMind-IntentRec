# embedding-intent-head Specification

## Purpose
TBD - created by archiving change minimind-intentrec-session-distillation. Update Purpose after archive.
## Requirements
### Requirement: SHALL 训练低延迟 intent classifier
系统 SHALL 训练轻量 intent head，将 session embedding 映射为 intent distribution。

#### Scenario: 从 session embedding 预测 intent
- **When** 输入 sequence encoder 产出的 session embedding
- **Then** MLP intent head 输出 intent distribution 和 top-k intent labels

### Requirement: SHALL 报告 intent classifier 质量指标
系统 SHALL 报告分类和校准指标。

#### Scenario: 评估 intent head
- **When** 输入 validation embeddings 和 teacher/student labels
- **Then** 输出 micro/macro F1、precision@k、recall@k
- **And** 在有概率输出时报告 ECE

