# Models

This directory is a local destination for trained model artifacts. Expected
output folders include:

- `models/urgency_distilbert/`
- `models/risk_factor_distilbert/`
- `models/insufficient_info_distilbert/`

Large artifacts are intentionally excluded from Git, including model weights,
`.safetensors`, `.bin`, `.pt`, `.pth`, `.ckpt`, and `checkpoint-*` directories.
They can be regenerated with the command-line training scripts under `src/`.

Only this README is committed from the `models/` directory.
