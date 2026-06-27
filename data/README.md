# Data

This folder is reserved for data files used by the project.

Full generated JSONL datasets are intentionally ignored by Git. Keep large or sensitive files outside version control.

Expected labeled JSONL schema, matching the original notebook:

```json
{
  "task_id": "example-001",
  "text": "Patient portal message text",
  "labels": {
    "urgency_level": "Green",
    "risk_factors": ["None"],
    "insufficient_information": false
  }
}
```

The `sample_data/` folder contains a tiny synthetic sample for command-line smoke tests.
