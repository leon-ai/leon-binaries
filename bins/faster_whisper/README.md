# Faster Whisper Binary

This binary provides Whisper-based transcription using faster-whisper.

## Model Resources

Models must be downloaded manually and loaded locally via `--model_size_or_path` and `--download_root`.

Example downloads:

```bash
huggingface-cli download Systran/faster-whisper-base \
  --local-dir "/home/louis/Downloads/faster_whisper/base" \
  --local-dir-use-symlinks False
```

## Usage

```bash
uv run run_faster_whisper.py \
  --function transcribe_to_file \
  --input input.wav \
  --output output/transcript.txt \
  --model_size_or_path base \
  --download_root /home/louis/Downloads/faster_whisper \
  --device auto
```

## JSON Task Format

### transcribe_to_file

This tool uses CLI arguments instead of JSON tasks.

Notes:
- `--local_files_only` is enabled by default; download models ahead of time.
- `--download_root` must match the directory where the model is stored.
