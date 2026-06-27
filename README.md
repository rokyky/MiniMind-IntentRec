# MiniMind-Rec — 推荐语义增强引擎

基于 [MiniMind](https://github.com/jingyaogong/minimind)（从零训练 64M 小语言模型）构建的推荐语义增强模块。
使用 LLM API 作为 teacher 生成结构化 item semantic tags，通过 LoRA SFT 蒸馏到小模型，
接入 TimeGenRec（精排 item-side feature）和 MiniOneRec（Semantic ID 构造）。

| | |
|---|---|
| MiniMind | [github.com/jingyaogong/minimind](https://github.com/jingyaogong/minimind) |
| 架构 | 标准 LLaMA 风格 + LoRA 适配器 |
| 参数量 | 64M-100M |

---

## 项目定位

不是独立主项目，而是 TimeGenRec 和 MiniOneRec 的语义特征生产模块。
Teacher LLM API -> MiniMind LoRA SFT -> 本地批量推理 -> TimeGenRec / MiniOneRec 消费

## 技术路线

1. 用 Qwen/DeepSeek API 生成结构化 semantic tags（teacher 标注）
2. LoRA SFT 蒸馏到 MiniMind 小模型（64M params）
3. 批量推理生成所有 item 的标签
4. 接入下游推荐模型验证指标收益

## 标签 Schema

{
  "function": ["moisturizing", "anti-aging"],
  "attributes": ["fragrance-free", "lightweight"],
  "scenario": ["daily use", "night routine"],
  "target_user": ["dry skin", "beginner"],
  "purchase_intent": ["repair skin barrier"]
}

---

## GPU 与训练配置

| 阶段 | 推荐 GPU | 显存 | 时长 | 租用价 |
|------|---------|------|------|--------|
| Teacher API 打标 | CPU / API | - | 10-30m | ~¥15（API 费）|
| LoRA SFT | 1x RTX 4090 24GB | ~4-6GB | 20-40m | ~¥2 |
| LoRA SFT | 1x A100 80GB | ~4GB | 10-20m | ~¥6 |
| 批量推理 12K items | 1x RTX 4090 | ~4GB | 5-10m | ~¥1 |
| 批量推理 12K items | CPU | - | 30-60m | 自带 |

## 文件结构

| 目录/文件 | 说明 |
|----------|------|
| model/ | MiniMind 模型架构 + LoRA 实现 |
| trainer/ | 训练脚本（pretrain / SFT / LoRA / DPO）|
| dataset/ | 数据集加载 |
| scripts/prepare_sft_data.py | Amazon 元数据 -> SFT 格式 |
| configs/lora_tag.yaml | LoRA tag 训练配置 |
| scripts/ | 适配脚本 |

---

## 评估体系

**第一层（标签质量）：** JSON valid rate > 95%, schema compliance > 90%, BERTScore vs teacher
**第二层（语义表示）：** 冷启动 retrieval recall, 类目内标签一致性
**第三层（推荐指标-核心）：** No tags vs API tags vs MiniMind tags 三组对比

| 方案 | Recall@K | NDCG@K | 冷启动 Recall |
|------|----------|--------|--------------|
| No tags（baseline）| baseline | baseline | baseline |
| API tags（teacher）| TBD | TBD | TBD |
| MiniMind tags | TBD | TBD | TBD |

---

## 参考

- MiniMind: github.com/jingyaogong/minimind
- Teacher LLM: Qwen / DeepSeek API

## License

Apache 2.0