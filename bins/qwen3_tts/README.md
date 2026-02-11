# Qwen3 TTS Binary

This binary provides text-to-speech (voice cloning) and voice design using the official Qwen3-TTS models.

## Model Resources

Models must be downloaded manually and loaded locally from `--resource_path`.

Download the model resources:

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir "/home/louis/Downloads/qwen3_tts_resources/Qwen3-TTS-12Hz-1.7B-Base" \
  --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --local-dir "/home/louis/Downloads/qwen3_tts_resources/Qwen3-TTS-12Hz-1.7B-VoiceDesign" \
  --local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir "/home/louis/Downloads/qwen3_tts_resources/Qwen3-TTS-12Hz-1.7B-CustomVoice" \
  --local-dir-use-symlinks False
```

## Usage

```bash
uv run run_qwen3_tts.py \
  --function synthesize_speech \
  --json_file mock/tasks_synthesize.json \
  --resource_path /home/louis/Downloads/qwen3_tts_resources \
  --device auto \
  --torch_path /path/to/torch \
  --nvidia_libs_path /path/to/nvidia
```

```bash
uv run run_qwen3_tts.py \
  --function design_voice \
  --json_file mock/tasks_design.json \
  --resource_path /home/louis/Downloads/qwen3_tts_resources \
  --device cuda \
  --torch_path /path/to/torch \
  --nvidia_libs_path /path/to/nvidia
```

```bash
uv run run_qwen3_tts.py \
  --function custom_voice \
  --json_file mock/tasks_custom.json \
  --resource_path /home/louis/Downloads/qwen3_tts_resources \
  --device cuda \
  --torch_path /path/to/torch \
  --nvidia_libs_path /path/to/nvidia
```

```bash
uv run run_qwen3_tts.py \
  --function design_then_synthesize \
  --json_file mock/tasks_design_then_synthesize.json \
  --resource_path /home/louis/Downloads/qwen3_tts_resources \
  --device cuda \
  --torch_path /path/to/torch \
  --nvidia_libs_path /path/to/nvidia
```

## JSON Task Format

### synthesize_speech

```json
[
  {
    "text": "Hello from Qwen3.",
    "target_language": "English",
    "audio_path": "output/hello.wav",
    "speaker_reference_path": "mock/reference.wav",
    "reference_text": "mock/reference.txt",
    "x_vector_only_mode": false,
    "max_new_tokens": 1024
  }
]
```

### design_voice

```json
[
  {
    "text": "Hello from a designed voice.",
    "target_language": "English",
    "instruct": "Warm, calm, and friendly tone.",
    "audio_path": "output/voice_design.wav",
    "max_new_tokens": 1024
  }
]
```

### custom_voice

```json
[
  {
    "text": "Hello from a custom voice.",
    "target_language": "English",
    "speaker": "Ryan",
    "instruct": "Bright and confident tone.",
    "audio_path": "output/custom_voice.wav",
    "max_new_tokens": 1024
  }
]
```

### design_then_synthesize

```json
[
  {
    "design_text": "Hello, this is the reference for our new voice persona.",
    "design_language": "English",
    "design_instruct": "Warm, calm, and friendly tone.",
    "texts": [
      "First batch line using the designed voice.",
      "Second batch line using the same designed voice."
    ],
    "languages": ["English", "English"],
    "output_paths": [
      "output/design_batch_1.wav",
      "output/design_batch_2.wav"
    ],
    "design_max_new_tokens": 256,
    "max_new_tokens": 256
  }
]
```

Notes:
- `speaker_reference_path` and `reference_text` are required for voice cloning unless `x_vector_only_mode` is true.
- `target_language` accepts `Auto` or explicit language names (e.g., English, Chinese).
- Tasks are batched per function call using the Qwen3-TTS batch APIs.
- Set `QWEN3_TTS_USE_SYSTEM_DEPS=1` to prefer system `sox/torchaudio` over the built-in stubs.
