# Ultimate Vocal Remover ONNX Binary

This binary separates vocals and instrumentals using an ONNX MDX model.

## Model Resources

Models must be downloaded manually and loaded locally from `--resource_path`.

`--resource_path` must point to the ONNX model file you want to use (e.g., an MDX-Net model).

## Usage

```bash
uv run run_ultimate_vocal_remover_onnx.py \
  --function separate_sources \
  --json_file tasks.json \
  --resource_path /path/to/model.onnx
```

```bash
uv run run_ultimate_vocal_remover_onnx.py \
  --function separate_sources \
  --json_file tasks.json \
  --resource_path /path/to/model.onnx \
  --cuda_runtime_path /path/to/cuda
```

## JSON Task Format

### separate_sources

```json
[
  {
    "audio_path": "input.wav",
    "vocal_output_path": "output/vocals.wav",
    "instrumental_output_path": "output/instrumental.wav",
    "aggression": 1.3
  }
]
```

Notes:
- `aggression` controls vocal isolation intensity (default: 1.3).
- `--function` is required but not validated by the script; use `separate_sources` for consistency.
