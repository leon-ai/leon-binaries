# Qwen3 ASR Binary

This binary provides speech recognition using Qwen3-ASR with batch and long-audio support.

## Model Resources

Models must be downloaded manually and loaded locally from `--model_path`.

Download the model resources:

```bash
huggingface-cli download Qwen/Qwen3-ASR-1.7B \
  --local-dir "/home/louis/Downloads/qwen3_asr_resources/Qwen3-ASR-1.7B" \
  --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B \
  --local-dir "/home/louis/Downloads/qwen3_asr_resources/Qwen3-ForcedAligner-0.6B" \
  --local-dir-use-symlinks False
```

## Usage

```bash
uv run run_qwen3_asr.py \
  --function transcribe_audio \
  --json_file tasks.json \
  --model_path /home/louis/Downloads/qwen3_asr_resources/Qwen3-ASR-1.7B \
  --device auto \
  --batch_size 4 \
  --language auto \
  --return_timestamps true
```

```bash
uv run run_qwen3_asr.py \
  --function transcribe_audio \
  --json_file tasks.json \
  --model_path /home/louis/Downloads/qwen3_asr_resources/Qwen3-ASR-1.7B \
  --device cuda \
  --cuda_runtime_path /path/to/cuda \
  --torch_path /path/to/torch \
  --forced_aligner_model_path /home/louis/Downloads/qwen3_asr_resources/Qwen3-ForcedAligner-0.6B
```

## JSON Task Format

### transcribe_audio

```json
[
  {
    "audio_path": "input.wav",
    "output_path": "output/transcript.txt",
    "language": "auto",
    "context": "Optional context hint"
  }
]
```

Notes:
- `audio_path` must exist; `output_path` is optional.
- `forced_aligner_model_path` is required for timestamp alignment.
- `--model_path` must point to a local Qwen3-ASR model directory containing `config.json`.
