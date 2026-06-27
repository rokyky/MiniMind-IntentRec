## ADDED Requirements

### Requirement: Extract Amazon item metadata
The system SHALL parse Amazon Beauty/Sports metadata JSON and extract title, category, description for each item.

#### Scenario: Metadata extracted
- **WHEN** prepare_data.py --dataset beauty is run
- **THEN** a JSONL file with (asin, title, category, description) is saved

### Requirement: Generate SFT data via LLM API
The system SHALL call Qwen/DeepSeek API to generate structured semantic tags for each item.

#### Scenario: Tags generated
- **WHEN** API is called for an item
- **THEN** a JSON response with function, attributes, scenario, target_user is stored

### Requirement: Format SFT pairs
The system SHALL format (input, output) pairs as JSONL for LoRA training.

#### Scenario: SFT data ready
- **WHEN** format_sft.py is run
- **THEN** train.jsonl and val.jsonl files are created with input/output fields
