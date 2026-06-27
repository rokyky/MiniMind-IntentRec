## ADDED Requirements

### Requirement: Batch inference for all items
The system SHALL run the trained model on all items and output structured tags.

#### Scenario: All items tagged
- **WHEN** batch_inference.py is run
- **THEN** a tags.json file with per-item structured tags is created
