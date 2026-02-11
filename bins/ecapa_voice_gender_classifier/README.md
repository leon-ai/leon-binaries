# ECAPA Voice Gender Classifier Binary

This binary detects speaker gender from an audio file using an ECAPA model.

## Model Resources

Models must be downloaded manually and loaded locally from `--model_path`.

`--model_path` can be a directory containing `model.safetensors` or `pytorch_model.bin` (and `model.py`).

## Usage

```bash
uv run run_ecapa_voice_gender_classifier.py \
  --function detect_gender \
  --input input.wav \
  --model_path /path/to/model \
  --device cpu
```

## JSON Task Format

### detect_gender

This tool uses CLI arguments instead of JSON tasks.

Notes:
- `--device` supports `cpu` and `cuda`.
- The model directory must include the ECAPA weights and `model.py`.
