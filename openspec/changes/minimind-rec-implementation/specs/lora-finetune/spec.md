## ADDED Requirements

### Requirement: LoRA adapter added to Mini-DeepLLM
The system SHALL add LoRA rank=8 adapters to attention layers of the smallest Mini-DeepLLM model.

#### Scenario: LoRA layers created
- **WHEN** model is initialized with LoRA
- **THEN** trainable params are < 5% of base model

### Requirement: SFT on tag generation data
The system SHALL train the LoRA model on the SFT dataset for 1 epoch.

#### Scenario: SFT loss decreases
- **WHEN** training runs for 1 epoch
- **THEN** the training loss decreases over steps
