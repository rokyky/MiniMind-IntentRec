## 1. Data Preparation

- [ ] 1.1 Extract Amazon item metadata (title, category, description) from raw JSON
- [ ] 1.2 Generate structured semantic tags via LLM API (teacher labels)
- [ ] 1.3 Format SFT data: input=(title+category), output=(structured JSON tags)
- [ ] 1.4 Split train/val (500 held-out for evaluation)

## 2. Model Fine-tuning

- [ ] 2.1 Add LoRA rank=8 adapter to Mini-DeepLLM smallest model
- [ ] 2.2 SFT for 1 epoch on tag generation data
- [ ] 2.3 Save LoRA adapter checkpoint

## 3. Evaluation Layer 1: Tag Quality

- [ ] 3.1 BLEU / ROUGE-L / BERTScore vs teacher API output
- [ ] 3.2 Manual check 100 samples for invalid tag rate

## 4. Evaluation Layer 2: Semantic Representation

- [ ] 4.1 Cold-start retrieval recall (text emb vs no text)
- [ ] 4.2 Intra-category tag consistency check

## 5. Evaluation Layer 3: Downstream Recommendation

- [ ] 5.1 Integrate tags with TimeGenRec TiSASRec-Cat
- [ ] 5.2 Experiment table: No tags vs API tags vs MiniMind tags
- [ ] 5.3 Semantic ID quality comparison for MiniOneRec

## 6. Documentation

- [ ] 6.1 Structured tag schema documentation
- [ ] 6.2 Experiment results and analysis