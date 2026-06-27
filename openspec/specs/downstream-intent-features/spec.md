# downstream-intent-features Specification

## Purpose
TBD - created by archiving change minimind-intentrec-session-distillation. Update Purpose after archive.
## Requirements
### Requirement: SHALL 导出下游 intent feature
系统 SHALL 将 intent 预测结果导出给推荐模型使用。

#### Scenario: 导出 intent features
- **When** 已有 session/user 的 intent predictions
- **Then** feature export 写出 intent ids、probabilities、source model 和 confidence
- **And** 格式稳定、可机器读取

### Requirement: SHALL 对比 intent feature 推荐效果
系统 SHALL 比较使用和不使用 intent feature 的推荐效果。

#### Scenario: 比较 intent variants
- **When** 下游推荐结果包含多个 intent variants
- **Then** 报告 no-intent、category-majority、cluster、teacher、MiniMind、MLP intent
- **And** 输出 aggregate 和 short-history / cold-start / session-drift 切片

