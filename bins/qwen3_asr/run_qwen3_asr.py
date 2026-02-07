#!/usr/bin/env python3
"""
Qwen3-ASR 1.7B - Speech Recognition CLI Tool (Optimized)

This module provides high-performance ASR inference using PyTorch.
Compatible with Linux (CUDA), Linux (CPU), and macOS (CPU/MPS).
"""

import os
import sys
import json
import argparse
import multiprocessing
import warnings

# Suppress warnings early
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def _parse_cuda_path_early():
    """Parse --cuda_runtime_path from argv before argparse runs."""
    for i, arg in enumerate(sys.argv):
        if arg == "--cuda_runtime_path" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def _parse_torch_path_early():
    """Parse --torch_path from argv before argparse runs."""
    for i, arg in enumerate(sys.argv):
        if arg == "--torch_path" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def _load_torch_from_path(torch_path):
    """
    Load PyTorch from a custom path by adding it to sys.path.
    Returns True if torch was successfully loaded.
    """
    if not torch_path or not os.path.exists(torch_path):
        if torch_path:
            print(f"[Warning] PyTorch path does not exist: {torch_path}")
        return False

    # Add torch path to sys.path
    if torch_path not in sys.path:
        sys.path.insert(0, torch_path)
        print(f"[Info] Added PyTorch path to sys.path: {torch_path}")

    # Set library path for torch/lib
    torch_lib_path = os.path.join(torch_path, "torch", "lib")
    if os.path.exists(torch_lib_path):
        if sys.platform.startswith("win"):
            os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ.get("PATH", "")
        else:
            path_var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
            os.environ[path_var] = torch_lib_path + os.pathsep + os.environ.get(path_var, "")
            print(f"[Info] Added torch/lib to {path_var}: {torch_lib_path}")

    # Try to import torch
    try:
        import torch
        print(f"[Info] Successfully loaded PyTorch {torch.__version__} from: {torch.__file__}")
        return True
    except ImportError as e:
        print(f"[Error] Failed to import PyTorch from {torch_path}: {e}")
        print(f"[Info] Expected: {torch_path}/torch/__init__.py and {torch_path}/torch/lib/")
        return False


def _load_cuda_libraries(cuda_runtime_path):
    """Load CUDA libraries from the specified path."""
    if sys.platform == "darwin" or not cuda_runtime_path:
        return False

    if not os.path.exists(cuda_runtime_path):
        print(f"[Warning] CUDA runtime path not found: {cuda_runtime_path}")
        return False

    import ctypes
    import glob

    # Define paths and patterns based on platform
    if sys.platform.startswith("win"):
        cuda_paths = [
            os.path.join(cuda_runtime_path, "bin"),
            os.path.join(cuda_runtime_path, "cudnn", "bin"),
            os.path.join(cuda_runtime_path, "cublas", "bin"),
        ]
        patterns = ["cudnn*.dll", "cublas*.dll", "cusparse*.dll", "cudart*.dll"]
        path_var = "PATH"
    else:  # Linux
        cuda_paths = [
            os.path.join(cuda_runtime_path, "lib64"),
            os.path.join(cuda_runtime_path, "lib"),
            os.path.join(cuda_runtime_path, "cudnn", "lib"),
            os.path.join(cuda_runtime_path, "cublas", "lib"),
            os.path.join(cuda_runtime_path, "cusparse", "lib"),
            os.path.join(cuda_runtime_path, "nccl", "lib"),
            os.path.join(cuda_runtime_path, "nvshmem", "lib"),
        ]
        patterns = [
            "libcudnn.so.*", "libcublas.so.*", "libcusparseLt.so.*",
            "libcusparse.so.*", "libcudart.so.*", "libnccl.so.*",
            "libnvshmem_host.so.*",
        ]
        path_var = "LD_LIBRARY_PATH"

    # Filter valid paths
    valid_paths = [p for p in cuda_paths if os.path.exists(p)]
    if not valid_paths:
        print(f"[Warning] No valid CUDA library paths found in: {cuda_runtime_path}")
        return False

    # Add to environment
    os.environ[path_var] = os.pathsep.join(valid_paths) + os.pathsep + os.environ.get(path_var, "")
    print(f"[Info] Added CUDA libraries to path: {', '.join(valid_paths)}")

    # Pre-load libraries
    loaded = []
    for base in valid_paths:
        for pattern in patterns:
            for lib in glob.glob(os.path.join(base, pattern)):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(lib))
                    break
                except OSError:
                    continue

    if loaded:
        print(f"[Info] Loaded CUDA libs: {', '.join(sorted(set(loaded)))}")
        return True

    print("[Warning] CUDA libraries were not loaded from the provided path.")
    return False


# Parse and load early
_cuda_runtime_path = _parse_cuda_path_early()
_cuda_env_setup = _load_cuda_libraries(_cuda_runtime_path)
_torch_path = _parse_torch_path_early()
_torch_loaded = _load_torch_from_path(_torch_path)


def configure_torch_threads():
    """Configure PyTorch threading for optimal CPU performance."""
    import torch

    total_cores = os.cpu_count() or 4
    num_threads = max(1, total_cores - 2) if total_cores >= 8 else max(1, total_cores - 1) if total_cores >= 4 else max(1, total_cores // 2)

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)
    return num_threads


class Qwen3ASRProcessor:
    """Qwen3-ASR 1.7B Speech Recognition Processor."""

    def __init__(
        self,
        model_path,
        device="auto",
        batch_size=4,
        language="auto",
        return_timestamps=True,
        forced_aligner_model_path=None,
        chunk_duration=30,
        cpu_batch_size=None,
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.language = language
        self.return_timestamps = return_timestamps
        self.forced_aligner_model_path = forced_aligner_model_path
        self.chunk_duration = chunk_duration
        self.cpu_batch_size = cpu_batch_size
        self.model = None
        self.device_type = None
        self.num_threads = None
        self._supports_array_input = None

        self._load_model()

    def _get_adaptive_cpu_batch_size(self):
        """Adapt CPU batch size based on available memory."""
        base = max(1, min(self.cpu_batch_size or 2, self.batch_size))

        # Try to get available memory
        if hasattr(os, "sysconf"):
            try:
                pages = os.sysconf("SC_AVPHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                if isinstance(pages, int) and isinstance(page_size, int):
                    mem_gb = (pages * page_size) / (1024 ** 3)
                    extra = 2 if mem_gb >= 32 else 1 if mem_gb >= 20 else 0
                    adaptive = min(self.batch_size, base + extra, 8)
                    if adaptive != base:
                        print(f"[Info] CPU batch size adjusted based on RAM: {base} -> {adaptive}")
                    return adaptive
            except (ValueError, OSError, AttributeError):
                pass

        return base

    def _load_model(self):
        """Load the Qwen3-ASR model."""
        import torch

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as e:
            print(f"[Error] qwen_asr package not found. Please install it with: pip install qwen-asr")
            print(f"[Error] Import error: {e}")
            raise

        print(f"[Info] Loading Qwen3-ASR model from: {self.model_path}")
        if self.forced_aligner_model_path:
            print(f"[Info] Forced Aligner: {self.forced_aligner_model_path}")

        # Determine device
        if self.device == "auto":
            if sys.platform == "darwin":
                self.device_type = "mps" if torch.backends.mps.is_available() else "cpu"
                print(f"[Info] macOS: Using {'MPS (Metal)' if self.device_type == 'mps' else 'CPU'}")
            else:
                self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_type = self.device
            if self.device_type == "cuda" and sys.platform == "darwin":
                print("[Warning] CUDA not supported on macOS. Falling back to CPU.")
                self.device_type = "cpu"

        # Check CUDA availability
        if self.device_type == "cuda" and not torch.cuda.is_available():
            print("[Error] CUDA requested but not available. Use --cuda_runtime_path or --device cpu")
            raise RuntimeError("CUDA not available")

        print(f"[Info] Using device: {self.device_type}")

        # Configure device-specific settings
        if self.device_type == "cpu":
            self.num_threads = configure_torch_threads()
            print(f"[Info] CPU Mode: Using {self.num_threads} threads")
            if hasattr(torch.backends, 'mkldnn'):
                torch.backends.mkldnn.enabled = True
            if self.cpu_batch_size is None:
                total_cores = os.cpu_count() or 4
                self.cpu_batch_size = 4 if total_cores >= 16 else 3 if total_cores >= 8 else 2
        elif self.device_type == "mps":
            print(f"[Info] MPS Mode: Using Apple Metal acceleration")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        else:  # CUDA
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(f"[Info] GPU Mode: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()

        # Load model
        print("[Info] Loading model (this may take a while)...")
        try:
            model_kwargs = {
                "max_inference_batch_size": self.batch_size,
                "max_new_tokens": 256,
            }

            if self.device_type == "cuda":
                model_kwargs["dtype"] = torch.bfloat16
                model_kwargs["device_map"] = "cuda:0"
            elif self.device_type == "mps":
                model_kwargs["dtype"] = torch.float16
                model_kwargs["device_map"] = "mps"
            else:
                model_kwargs["dtype"] = torch.float32
                model_kwargs["device_map"] = "cpu"

            if self.return_timestamps and self.forced_aligner_model_path:
                model_kwargs["forced_aligner"] = self.forced_aligner_model_path
                model_kwargs["forced_aligner_kwargs"] = {
                    "dtype": model_kwargs["dtype"],
                    "device_map": model_kwargs["device_map"],
                }

            if os.path.exists(self.model_path) and os.path.isdir(self.model_path):
                if os.path.exists(os.path.join(self.model_path, "config.json")):
                    model_kwargs["local_files_only"] = True

            self.model = Qwen3ASRModel.from_pretrained(self.model_path, **model_kwargs)
        except Exception as e:
            print(f"[Error] Failed to load model: {e}")
            raise

        print("[Info] Model loaded successfully")

    def transcribe_single(self, audio_path, language=None, context=None):
        """Transcribe a single audio file."""
        lang = None if (language or self.language) in ("auto", None) else (language or self.language)
        transcribe_kwargs = {"return_time_stamps": True} if self.return_timestamps else {}

        try:
            results = self.model.transcribe(audio=audio_path, language=lang, **transcribe_kwargs)

            if results and len(results) > 0:
                result = results[0]
                transcription = result.text
                detected_lang = getattr(result, "language", lang or "unknown")

                if self.return_timestamps and hasattr(result, "time_stamps") and result.time_stamps:
                    transcription += f" [Timestamps: {result.time_stamps[0]}]"
            else:
                transcription = ""
                detected_lang = "unknown"
        except Exception as e:
            print(f"[Error] Transcription failed for {audio_path}: {e}")
            transcription = ""
            detected_lang = "unknown"

        return {"text": transcription, "language": detected_lang}

    def transcribe_batch(self, tasks):
        """Transcribe multiple audio files in batches."""
        print(f"[Info] Processing {len(tasks)} tasks with batch optimization...")

        results = []
        audio_paths = []
        languages = []
        task_mapping = []

        # Filter valid tasks
        for task in tasks:
            if os.path.exists(task.get("audio_path", "")):
                audio_paths.append(task["audio_path"])
                lang = task.get("language", self.language)
                languages.append(None if lang in ("auto", None) else lang)
                task_mapping.append(task)
            else:
                print(f"[Warning] Audio file not found: {task.get('audio_path', 'unknown')}")
                results.append({
                    "audio_path": task.get("audio_path", ""),
                    "error": "Audio file not found",
                    "text": "",
                    "language": "",
                })

        if not audio_paths:
            return results

        print(f"[Info] Processing all {len(audio_paths)} audio files in one batch...")

        try:
            transcribe_kwargs = {"return_time_stamps": True} if self.return_timestamps else {}
            batch_results = self.model.transcribe(audio=audio_paths, language=languages, **transcribe_kwargs)

            for idx, task in enumerate(task_mapping):
                if idx < len(batch_results):
                    result = batch_results[idx]
                    transcription = result.text
                    detected_lang = getattr(result, "language", languages[idx] or "unknown")

                    if self.return_timestamps and hasattr(result, "time_stamps") and result.time_stamps:
                        timestamp_info = [f"[{ts.start_time:.2f}-{ts.end_time:.2f}s] {ts.text}" for ts in result.time_stamps]
                        if timestamp_info:
                            transcription = result.text + "\n" + "\n".join(timestamp_info)

                    result_dict = {
                        "audio_path": task["audio_path"],
                        "text": transcription,
                        "language": detected_lang,
                    }

                    if task.get("output_path"):
                        os.makedirs(os.path.dirname(task["output_path"]), exist_ok=True)
                        with open(task["output_path"], "w", encoding="utf-8") as f:
                            f.write(transcription)
                        print(f"[Info] Saved transcription to: {task['output_path']}")

                    results.append(result_dict)
                else:
                    results.append({
                        "audio_path": task["audio_path"],
                        "error": "No transcription result",
                        "text": "",
                        "language": "",
                    })
        except Exception as e:
            print(f"[Error] Batch transcription failed: {e}")
            for task in task_mapping:
                results.append({
                    "audio_path": task["audio_path"],
                    "error": str(e),
                    "text": "",
                    "language": "",
                })

        return results

    def transcribe_long_audio(self, audio_path, language=None, context=None, chunk_duration=None):
        """Transcribe long audio files by chunking them."""
        import librosa
        import soundfile as sf
        import tempfile

        lang = None if (language or self.language) in ("auto", None) else (language or self.language)
        chunk_duration = chunk_duration or self.chunk_duration

        print(f"[Info] Processing long audio: {audio_path}")

        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            duration = librosa.get_duration(y=audio, sr=sr)

            if duration <= chunk_duration:
                print(f"[Info] Audio is {duration:.1f}s, processing without chunking")
                return self.transcribe_single(audio_path, language, context)

            print(f"[Info] Audio is {duration:.1f}s, auto-chunking into {chunk_duration}s segments")

            temp_dir = None
            chunk_starts = []
            chunk_files = []
            chunk_arrays = []
            use_cpu_arrays = self.device_type == "cpu" and self._supports_array_input is not False

            # Split audio into chunks
            for i in range(0, int(duration), chunk_duration):
                start_time = i
                end_time = min(i + chunk_duration, duration)
                chunk_starts.append(start_time)
                chunk_audio = audio[int(start_time * sr):int(end_time * sr)]

                if self.device_type == "cuda" or not use_cpu_arrays:
                    if temp_dir is None:
                        temp_dir = tempfile.TemporaryDirectory()
                    chunk_path = os.path.join(temp_dir.name, f"chunk_{i:03d}_{int(end_time):03d}.wav")
                    sf.write(chunk_path, chunk_audio, sr)
                    chunk_files.append(chunk_path)
                    chunk_label = len(chunk_files)
                else:
                    chunk_arrays.append(chunk_audio)
                    chunk_label = len(chunk_arrays)

                print(f"[Info] Created chunk {chunk_label}: {start_time:.1f}s-{end_time:.1f}s")

            # Process chunks
            transcribe_kwargs = {"return_time_stamps": self.return_timestamps}
            chunk_transcriptions = []
            all_timestamps = []
            detected_lang = "unknown"

            if self.device_type == "cuda":
                print(f"[Info] Transcribing {len(chunk_files)} chunks in batch (GPU)...")
                batch_results = self.model.transcribe(audio=chunk_files, language=None, **transcribe_kwargs)

                for i, result in enumerate(batch_results):
                    if result.text:
                        chunk_transcriptions.append(result.text)
                        print(f"[Info] Chunk {i + 1}/{len(chunk_files)} transcribed successfully")

                        if self.return_timestamps and hasattr(result, "time_stamps") and result.time_stamps:
                            chunk_start = i * chunk_duration
                            for ts in result.time_stamps:
                                all_timestamps.append({
                                    "start": chunk_start + ts.start_time,
                                    "end": chunk_start + ts.end_time,
                                    "text": ts.text,
                                })
                    else:
                        print(f"[Warning] Chunk {i + 1}/{len(chunk_files)} failed to transcribe")

                detected_lang = next((r.language for r in batch_results if getattr(r, "language", None) and r.language != "unknown"), "unknown")
            else:
                # CPU batch processing
                cpu_batch_size = self._get_adaptive_cpu_batch_size()

                def process_cpu_batches(batch_inputs, total_count):
                    nonlocal detected_lang
                    for batch_start in range(0, total_count, cpu_batch_size):
                        batch_end = min(batch_start + cpu_batch_size, total_count)
                        batch_items = batch_inputs[batch_start:batch_end]

                        batch_results = self.model.transcribe(audio=batch_items, language=None, **transcribe_kwargs)

                        for idx_in_batch, result in enumerate(batch_results):
                            chunk_idx = batch_start + idx_in_batch
                            chunk_start_time = chunk_starts[chunk_idx]

                            if result.text:
                                chunk_transcriptions.append(result.text)
                                print(f"[Info] Chunk {chunk_idx + 1}/{total_count} transcribed successfully")

                                if self.return_timestamps and hasattr(result, "time_stamps") and result.time_stamps:
                                    for ts in result.time_stamps:
                                        all_timestamps.append({
                                            "start": chunk_start_time + ts.start_time,
                                            "end": chunk_start_time + ts.end_time,
                                            "text": ts.text,
                                        })

                                if detected_lang == "unknown":
                                    lang = getattr(result, "language", None)
                                    if lang and lang != "unknown":
                                        detected_lang = lang
                            else:
                                print(f"[Warning] Chunk {chunk_idx + 1}/{total_count} failed to transcribe")

                if chunk_arrays:
                    print(f"[Info] Transcribing {len(chunk_arrays)} chunks in batches of {cpu_batch_size} (CPU)...")
                    try:
                        process_cpu_batches(chunk_arrays, len(chunk_arrays))
                    except Exception as e:
                        if "Unsupported audio input type" in str(e):
                            self._supports_array_input = False
                            print("[Info] Backend does not support in-memory audio; falling back to temp files.")
                        else:
                            print(f"[Warning] In-memory CPU batching failed, falling back to temp files: {e}")

                        chunk_transcriptions = []
                        all_timestamps = []
                        detected_lang = "unknown"

                        if temp_dir is None:
                            temp_dir = tempfile.TemporaryDirectory()
                        chunk_files = []

                        for idx, chunk_audio in enumerate(chunk_arrays):
                            end_time = min(chunk_starts[idx] + chunk_duration, duration)
                            chunk_path = os.path.join(temp_dir.name, f"chunk_{chunk_starts[idx]:03d}_{int(end_time):03d}.wav")
                            sf.write(chunk_path, chunk_audio, sr)
                            chunk_files.append(chunk_path)

                        print(f"[Info] Transcribing {len(chunk_files)} chunks in batches of {cpu_batch_size} (CPU)...")
                        process_cpu_batches(chunk_files, len(chunk_files))
                else:
                    print(f"[Info] Transcribing {len(chunk_files)} chunks in batches of {cpu_batch_size} (CPU)...")
                    process_cpu_batches(chunk_files, len(chunk_files))

            # Merge results
            if chunk_transcriptions:
                merged_transcription = " ".join(chunk_transcriptions).strip()
                print(f"[Info] Merged {len(chunk_transcriptions)} chunk transcriptions")

                if all_timestamps:
                    timestamp_lines = [f"[{ts['start']:.2f}-{ts['end']:.2f}s] {ts['text']}" for ts in all_timestamps]
                    merged_transcription = merged_transcription + "\n" + "\n".join(timestamp_lines)
            else:
                merged_transcription = ""

            # Cleanup
            if temp_dir is not None:
                try:
                    temp_dir.cleanup()
                except:
                    pass

        except Exception as e:
            print(f"[Error] Long audio processing failed: {e}")
            return {"text": "", "language": "unknown"}

        return {"text": merged_transcription, "language": detected_lang}


def main():
    """CLI entry point."""
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Qwen3-ASR 1.7B - Speech Recognition CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  qwen3_asr --function transcribe_audio --json_file tasks.json --model_path /path/to/model
  
  # With external PyTorch
  qwen3_asr --function transcribe_audio --json_file tasks.json --model_path /path/to/model --torch_path /path/to/torch
  
  # With CUDA (Linux/Windows only)
  qwen3_asr --function transcribe_audio --json_file tasks.json --model_path /path/to/model --cuda_runtime_path /usr/local/cuda
  
  # With timestamps
  qwen3_asr --function transcribe_audio --json_file tasks.json --model_path /path/to/model --return_timestamps true --forced_aligner_model_path Qwen/Qwen3-ForcedAligner-0.6B
        """,
    )

    parser.add_argument("--function", type=str, required=True, choices=["transcribe_audio"], help="Function to execute")
    parser.add_argument("--json_file", type=str, required=True, help="Path to JSON file with transcription tasks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to local Qwen3-ASR model directory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device to use for inference (default: auto)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing (default: 4)")
    parser.add_argument("--language", type=str, default="auto", help="Language code for transcription (default: auto)")
    parser.add_argument("--return_timestamps", type=str, default="true", choices=["true", "false"], help="Return timestamps in output (default: true)")
    parser.add_argument("--cuda_runtime_path", type=str, default=None, help="Path to CUDA runtime directory (Linux/Windows only)")
    parser.add_argument("--torch_path", type=str, default=None, help="Path to PyTorch installation directory (site-packages containing torch)")
    parser.add_argument("--forced_aligner_model_path", type=str, default=None, help="Path to ForcedAligner model for timestamp generation")
    parser.add_argument("--chunk_duration", type=int, default=30, help="Chunk duration in seconds for long audio (default: 30)")
    parser.add_argument("--cpu_batch_size", type=int, default=None, help="CPU batch size for long audio (default: auto)")

    args = parser.parse_args()

    # Handle CUDA runtime path
    if args.cuda_runtime_path and not _cuda_env_setup:
        if sys.platform == "darwin":
            print("[Info] --cuda_runtime_path is ignored on macOS (CUDA not supported)")
        else:
            print(f"[Info] Setting up CUDA runtime from: {args.cuda_runtime_path}")
            _load_cuda_libraries(args.cuda_runtime_path)

    # Handle PyTorch path
    if args.torch_path and not _torch_loaded:
        print(f"[Info] Loading PyTorch from: {args.torch_path}")
        if not _load_torch_from_path(args.torch_path):
            print("[Error] Failed to load PyTorch from the specified path")
            sys.exit(1)

    # Validate arguments
    if not os.path.exists(args.json_file):
        print(f"[Error] JSON file not found: {args.json_file}")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"[Error] Model path not found: {args.model_path}")
        sys.exit(1)

    # Load tasks
    try:
        with open(args.json_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load JSON: {e}")
        sys.exit(1)

    if not isinstance(tasks, list):
        print("[Error] JSON file must contain a list of tasks")
        sys.exit(1)

    print(f"[Info] Loaded {len(tasks)} transcription tasks")

    # Initialize processor
    try:
        processor = Qwen3ASRProcessor(
            model_path=args.model_path,
            device=args.device,
            batch_size=args.batch_size,
            language=args.language,
            return_timestamps=args.return_timestamps.lower() == "true",
            forced_aligner_model_path=args.forced_aligner_model_path,
            chunk_duration=args.chunk_duration,
            cpu_batch_size=args.cpu_batch_size,
        )
    except Exception as e:
        print(f"[Error] Failed to initialize processor: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Process tasks
    import librosa

    success_count = 0
    error_count = 0
    long_audio_tasks = []
    regular_tasks = []

    print(f"\n[Info] Starting transcription...")
    print(f"[Info] Batch size: {args.batch_size}")
    print(f"[Info] Language: {args.language}")

    # Classify tasks by duration
    for task in tasks:
        try:
            duration = librosa.get_duration(path=task["audio_path"])
            (long_audio_tasks if duration > 20 else regular_tasks).append(task)
        except Exception as e:
            print(f"[Warning] Could not check duration for {task.get('audio_path', 'unknown')}: {e}")
            regular_tasks.append(task)

    # Process regular tasks
    if regular_tasks:
        print(f"\n[Info] Processing {len(regular_tasks)} regular audio files...")
        results = processor.transcribe_batch(regular_tasks)
        for result in results:
            if "error" in result:
                error_count += 1
            else:
                success_count += 1

    # Process long audio files
    if long_audio_tasks:
        print(f"\n[Info] Processing {len(long_audio_tasks)} long audio files with chunking...")
        for task in long_audio_tasks:
            try:
                result = processor.transcribe_long_audio(
                    task["audio_path"],
                    language=task.get("language"),
                    context=task.get("context"),
                    chunk_duration=args.chunk_duration,
                )

                if task.get("output_path"):
                    os.makedirs(os.path.dirname(task["output_path"]), exist_ok=True)
                    with open(task["output_path"], "w", encoding="utf-8") as f:
                        f.write(result["text"])
                    print(f"[Info] Saved transcription to: {task['output_path']}")

                success_count += 1
            except Exception as e:
                print(f"[Error] Failed to process {task['audio_path']}: {e}")
                error_count += 1

    print(f"\n[Info] Transcription complete!")
    print(f"[Info] Success: {success_count}/{len(tasks)}")
    print(f"[Info] Errors: {error_count}/{len(tasks)}")

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
