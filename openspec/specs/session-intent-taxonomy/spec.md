# session-intent-taxonomy Specification

## Purpose
TBD - created by archiving change minimind-intentrec-session-distillation. Update Purpose after archive.
## Requirements
### Requirement: SHALL 定义 Session Intent Taxonomy
系统 SHALL 定义受控 schema，用于 session-level user intent labels。

#### Scenario: 生成 taxonomy labels
- **When** 输入由近期用户行为构造的 session
- **Then** teacher labeling 输出 primary intent、可选 secondary intents、confidence 和 evidence items
- **And** 输出符合配置 schema

### Requirement: SHALL 缓存 Teacher Label
系统 SHALL 缓存 teacher 生成的 taxonomy 和 labels。

#### Scenario: 复用 teacher labels
- **When** session id 已存在 teacher label
- **Then** labeling 默认复用缓存
- **And** 显式指定刷新时重新生成

