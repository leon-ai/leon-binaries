# Chatterbox ONNX Binary

This binary provides ONNX-based text-to-speech synthesis with optional CUDA acceleration.

## Model Resources

Models must be downloaded manually and loaded locally from `--resource_path`.

Expected files under `--resource_path` (or in `--resource_path/onnx`):

- `speech_encoder.onnx`
- `embed_tokens.onnx`
- `conditional_decoder.onnx`
- `language_model_q4.onnx`
- `tokenizer.json`
- `default_voices/` (default speaker wavs)

## Usage

```bash
uv run run_chatterbox_onnx.py \
  --function synthesize_speech \
  --json_file tasks.json \
  --resource_path /path/to/chatterbox_onnx_models
```

```bash
uv run run_chatterbox_onnx.py \
  --function synthesize_speech \
  --json_file tasks.json \
  --resource_path /path/to/chatterbox_onnx_models \
  --cuda_runtime_path /path/to/cuda
```

## JSON Task Format

### synthesize_speech

```json
[
  {
    "text": "Hello from Chatterbox.",
    "target_language": "en",
    "audio_path": "output/hello.wav",
    "voice_name": "en_female",
    "cfg_strength": 0.5,
    "exaggeration": 0.5,
    "temperature": 0.5
  }
]
```

Notes:
- Use `voice_name` to select a file from `default_voices`, or set `speaker_reference_path` to a custom wav.
- `speaker_reference_path` takes precedence over `voice_name`.
- `--cuda_runtime_path` should contain `cudnn` and `cublas` subdirectories for GPU usage.
