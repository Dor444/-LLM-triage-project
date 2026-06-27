# Model Results Summary

## Synthetic Data Generation

| Metric | Value |
|---|---:|
| Generation tasks | 3,000 |
| Valid examples | 2,799 |
| Failed examples | 201 |
| Success rate | 93.3% |

## Dataset Split

| Split | Count |
|---|---:|
| Train | 1,958 |
| Validation | 420 |
| Test | 420 |

## Fine-Tuned Models

| Task | Model | Key Metric |
|---|---|---:|
| Urgency classification | DistilBERT | Accuracy 0.8095 |
| Urgency classification | DistilBERT | Macro-F1 0.8104 |
| Risk-factor multi-label classification | DistilBERT | Best threshold 0.30 |
| Risk-factor multi-label classification | DistilBERT | Micro-F1 0.8792 |
| Risk-factor multi-label classification | DistilBERT | Macro-F1 0.7775 |
| Insufficient-information classification | DistilBERT | Accuracy 0.8690 |
| Insufficient-information classification | DistilBERT | Macro-F1 0.7945 |
| Insufficient-information classification | DistilBERT | F1 insufficient 0.6707 |

## Prompting Baselines

| Baseline | Accuracy | Macro-F1 |
|---|---:|---:|
| Zero-shot Qwen urgency | 0.34 | 0.2949 |
| Few-shot Qwen urgency | 0.25 | 0.2080 |

## Unified Pipeline on 50 Test Examples

| Metric | Value |
|---|---:|
| Urgency accuracy | 0.76 |
| Risk exact-match accuracy | 0.92 |
| Insufficient accuracy | 0.86 |
| Human review rate | 0.72 |

The corresponding small output artifacts are included in this folder:

- `pipeline_test_outputs.csv`
- `pipeline_test_outputs.jsonl`
- `risk_factor_threshold_tuning_results.json`
