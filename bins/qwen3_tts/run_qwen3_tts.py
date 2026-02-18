#!/usr/bin/env python3
"""
Qwen3-TTS - Text-to-Speech and Voice Design CLI Tool.
"""

import argparse
import json
import multiprocessing
import os
import sys
import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def _demote_script_dir_for_system_deps():
    if os.environ.get("QWEN3_TTS_USE_SYSTEM_DEPS") != "1":
        return
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir in sys.path:
        sys.path.remove(script_dir)
        sys.path.append(script_dir)


_demote_script_dir_for_system_deps()


def _parse_nvidia_libs_path_early() -> Optional[str]:
    for i, arg in enumerate(sys.argv):
        if arg == "--nvidia_libs_path" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def _parse_torch_path_early() -> Optional[str]:
    for i, arg in enumerate(sys.argv):
        if arg == "--torch_path" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def _load_torch_from_path(torch_path: Optional[str]) -> bool:
    if not torch_path or not os.path.exists(torch_path):
        if torch_path:
            print(f"[Warning] PyTorch path does not exist: {torch_path}")
        return False

    if torch_path not in sys.path:
        sys.path.insert(0, torch_path)
        print(f"[Info] Added PyTorch path to sys.path: {torch_path}")

    torch_lib_path = os.path.join(torch_path, "torch", "lib")
    if os.path.exists(torch_lib_path):
        if sys.platform.startswith("win"):
            os.environ["PATH"] = (
                torch_lib_path + os.pathsep + os.environ.get("PATH", "")
            )
        else:
            path_var = (
                "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
            )
            os.environ[path_var] = (
                torch_lib_path + os.pathsep + os.environ.get(path_var, "")
            )
            print(f"[Info] Added torch/lib to {path_var}: {torch_lib_path}")

    try:
        import torch  # noqa: F401

        print(f"[Info] Successfully loaded PyTorch from: {torch_path}")
        return True
    except ImportError as e:
        print(f"[Error] Failed to import PyTorch from {torch_path}: {e}")
        print(
            f"[Info] Expected: {torch_path}/torch/__init__.py and {torch_path}/torch/lib/"
        )
        return False


def _load_nvidia_libraries(nvidia_libs_path: Optional[str]) -> bool:
    if not nvidia_libs_path or sys.platform == "darwin":
        return False

    if not os.path.exists(nvidia_libs_path):
        print(f"[Warning] NVIDIA libs path not found: {nvidia_libs_path}")
        return False

    import ctypes
    import glob

    if sys.platform.startswith("win"):
        cuda_paths = [
            os.path.join(nvidia_libs_path, "bin"),
            os.path.join(nvidia_libs_path, "cudnn", "bin"),
            os.path.join(nvidia_libs_path, "cublas", "bin"),
            os.path.join(nvidia_libs_path, "cusparse", "bin"),
        ]
        patterns = ["cudnn*.dll", "cublas*.dll", "cusparse*.dll", "cudart*.dll"]
        path_var = "PATH"
    else:
        cuda_paths = [
            os.path.join(nvidia_libs_path, "lib64"),
            os.path.join(nvidia_libs_path, "lib"),
            os.path.join(nvidia_libs_path, "cudnn", "lib"),
            os.path.join(nvidia_libs_path, "cublas", "lib"),
            os.path.join(nvidia_libs_path, "cusparse", "lib"),
            os.path.join(nvidia_libs_path, "nccl", "lib"),
            os.path.join(nvidia_libs_path, "nvshmem", "lib"),
        ]
        patterns = [
            "libcudnn.so.*",
            "libcublas.so.*",
            "libcusparseLt.so.*",
            "libcusparse.so.*",
            "libcudart.so.*",
            "libnccl.so.*",
            "libnvshmem_host.so.*",
        ]
        path_var = "LD_LIBRARY_PATH"

    valid_paths = [p for p in cuda_paths if os.path.exists(p)]
    if not valid_paths:
        print(f"[Warning] No valid CUDA library paths found in: {nvidia_libs_path}")
        return False

    os.environ[path_var] = (
        os.pathsep.join(valid_paths) + os.pathsep + os.environ.get(path_var, "")
    )
    print(f"[Info] Added NVIDIA libs to {path_var}: {', '.join(valid_paths)}")

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
        print(f"[Info] Loaded NVIDIA libs: {', '.join(sorted(set(loaded)))}")
        return True

    print("[Warning] NVIDIA libraries were not loaded from the provided path.")
    return False


_nvidia_libs_path = _parse_nvidia_libs_path_early()
_nvidia_libs_loaded = _load_nvidia_libraries(_nvidia_libs_path)
_nvidia_path_failed = _nvidia_libs_path and not _nvidia_libs_loaded

_torch_path = _parse_torch_path_early()
_torch_loaded = _load_torch_from_path(_torch_path)

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


MODEL_DIR_BASE_NAME = "Qwen3-TTS-12Hz-1.7B-Base"
MODEL_DIR_DESIGN_NAME = "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_DIR_CUSTOM_NAME = "Qwen3-TTS-12Hz-1.7B-CustomVoice"


def _resolve_path(path_value: str, base_dir: Path) -> str:
    if not path_value:
        return path_value
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(base_dir / path)


def _read_text_arg(text_or_path: str) -> str:
    if not text_or_path:
        return text_or_path

    if len(text_or_path) > 255:
        return text_or_path

    if any(ch in text_or_path for ch in ("\n", "\r")):
        return text_or_path

    looks_like_path = (
        os.sep in text_or_path
        or text_or_path.startswith(".")
        or text_or_path.startswith("~")
        or text_or_path.endswith((".txt", ".md", ".json"))
    )

    if not looks_like_path:
        return text_or_path

    try:
        path = Path(text_or_path).expanduser()
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return text_or_path

    return text_or_path


def _configure_torch_threads() -> int:
    total_cores = os.cpu_count() or 4
    num_threads = (
        max(1, total_cores - 2)
        if total_cores >= 8
        else max(1, total_cores - 1)
        if total_cores >= 4
        else max(1, total_cores // 2)
    )
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)
    return num_threads


def _detect_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if _nvidia_path_failed:
            print(
                "[Warning] NVIDIA libs path was provided but loading failed. Using CPU mode."
            )
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_dtype(device: str):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _load_model(model_path: Path, device: str) -> Qwen3TTSModel:
    dtype = _get_dtype(device)
    device_map = "cpu" if device == "cpu" else "cuda:0" if device == "cuda" else "mps"
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map=device_map,
        dtype=dtype,
        local_files_only=True,
    )
    return model


def _ensure_model_dir(resource_path: Path, model_dir_name: str) -> Path:
    model_path = resource_path / model_dir_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not (model_path / "model.safetensors").exists():
        raise FileNotFoundError(f"model.safetensors not found in: {model_path}")
    return model_path


def _validate_torch_loaded():
    if not _torch_loaded:
        print(
            "[Error] PyTorch was not loaded. Provide --torch_path pointing to a valid torch install."
        )
        sys.exit(1)


def synthesize_speech(json_file: str, resource_path: str, device: str):
    if not os.path.exists(json_file):
        print(f"[Error] JSON not found: {json_file}")
        sys.exit(1)
    if not os.path.exists(resource_path):
        print(f"[Error] Resource path not found: {resource_path}")
        sys.exit(1)

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"[Error] JSON load failed: {e}")
        sys.exit(1)

    if not isinstance(tasks, list):
        print("[Error] JSON root must be a list of tasks")
        sys.exit(1)

    _validate_torch_loaded()
    resolved_device = _detect_device(device)
    if resolved_device == "cpu":
        threads = _configure_torch_threads()
        print(f"[Info] CPU Mode: Using {threads} threads")

    base_dir = Path(json_file).resolve().parent
    resource_dir = Path(resource_path)

    print(f"[Info] Device: {resolved_device}")
    print(f"[Info] Tasks: {len(tasks)}")

    texts: List[str] = []
    languages: List[str] = []
    ref_audios: List[str] = []
    ref_texts: List[Optional[str]] = []
    xvec_modes: List[bool] = []
    output_paths: List[str] = []
    gen_kwargs: List[dict] = []

    for task in tasks:
        text = task.get("text")
        output_path = task.get("audio_path") or task.get("output_path")
        ref_audio = task.get("speaker_reference_path") or task.get(
            "reference_audio_path"
        )
        ref_text = task.get("reference_text")

        if not text or not output_path:
            print("[Warning] Skipped task: missing text/output_path")
            continue
        if not ref_audio:
            print(
                "[Warning] Skipped task: missing speaker_reference_path/reference_audio_path"
            )
            continue

        language = task.get("target_language", task.get("language", "Auto"))
        xvec_only = bool(task.get("x_vector_only_mode", False))
        if not xvec_only and not ref_text:
            print(
                "[Warning] Skipped task: reference_text required when x_vector_only_mode is false"
            )
            continue

        resolved_output = _resolve_path(output_path, base_dir)
        texts.append(_read_text_arg(text))
        languages.append(language)
        ref_audios.append(_resolve_path(ref_audio, base_dir))
        ref_texts.append(_read_text_arg(ref_text) if ref_text else None)
        xvec_modes.append(xvec_only)
        output_paths.append(resolved_output)

        gen_kwargs.append(
            {
                "max_new_tokens": int(task.get("max_new_tokens", 1024)),
                "do_sample": bool(task.get("do_sample", True)),
                "top_k": int(task.get("top_k", 50)),
                "top_p": float(task.get("top_p", 1.0)),
                "temperature": float(task.get("temperature", 0.9)),
                "repetition_penalty": float(task.get("repetition_penalty", 1.05)),
                "subtalker_dosample": bool(task.get("subtalker_dosample", True)),
                "subtalker_top_k": int(task.get("subtalker_top_k", 50)),
                "subtalker_top_p": float(task.get("subtalker_top_p", 1.0)),
                "subtalker_temperature": float(task.get("subtalker_temperature", 0.9)),
            }
        )

    if not texts:
        print("[Error] No valid tasks to process")
        sys.exit(1)

    model_path = _ensure_model_dir(resource_dir, MODEL_DIR_BASE_NAME)
    model = _load_model(model_path, resolved_device)

    if len(texts) > 1:
        common = gen_kwargs[0]
        if not all(kwargs == common for kwargs in gen_kwargs):
            print(
                "[Warning] Mixed generation params detected. Using first task's settings for batch."
            )
        gen_kwargs = [common] * len(texts)

    kwargs = gen_kwargs[0]
    ref_audio_inputs: List[Any] = list(ref_audios)
    wavs, sr = model.generate_voice_clone(
        text=texts,
        language=languages,
        ref_audio=ref_audio_inputs,
        ref_text=ref_texts,
        x_vector_only_mode=xvec_modes,
        **kwargs,
    )

    for wav, output_path in zip(wavs, output_paths):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, wav, sr)
        print(f"[OK] Wrote {output_path}")


def design_voice(json_file: str, resource_path: str, device: str):
    if not os.path.exists(json_file):
        print(f"[Error] JSON not found: {json_file}")
        sys.exit(1)
    if not os.path.exists(resource_path):
        print(f"[Error] Resource path not found: {resource_path}")
        sys.exit(1)

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"[Error] JSON load failed: {e}")
        sys.exit(1)

    if not isinstance(tasks, list):
        print("[Error] JSON root must be a list of tasks")
        sys.exit(1)

    _validate_torch_loaded()
    resolved_device = _detect_device(device)
    if resolved_device == "cpu":
        threads = _configure_torch_threads()
        print(f"[Info] CPU Mode: Using {threads} threads")

    base_dir = Path(json_file).resolve().parent
    resource_dir = Path(resource_path)

    print(f"[Info] Device: {resolved_device}")
    print(f"[Info] Tasks: {len(tasks)}")

    texts: List[str] = []
    languages: List[str] = []
    instructs: List[str] = []
    output_paths: List[str] = []
    gen_kwargs: List[dict] = []

    for task in tasks:
        text = task.get("text")
        output_path = task.get("audio_path") or task.get("output_path")
        if not text or not output_path:
            print("[Warning] Skipped task: missing text/output_path")
            continue

        language = task.get("target_language", task.get("language", "Auto"))
        instruct = task.get("instruct", "")

        resolved_output = _resolve_path(output_path, base_dir)
        texts.append(_read_text_arg(text))
        languages.append(language)
        instructs.append(_read_text_arg(instruct) if instruct else "")
        output_paths.append(resolved_output)

        gen_kwargs.append(
            {
                "max_new_tokens": int(task.get("max_new_tokens", 1024)),
                "do_sample": bool(task.get("do_sample", True)),
                "top_k": int(task.get("top_k", 50)),
                "top_p": float(task.get("top_p", 1.0)),
                "temperature": float(task.get("temperature", 0.9)),
                "repetition_penalty": float(task.get("repetition_penalty", 1.05)),
                "subtalker_dosample": bool(task.get("subtalker_dosample", True)),
                "subtalker_top_k": int(task.get("subtalker_top_k", 50)),
                "subtalker_top_p": float(task.get("subtalker_top_p", 1.0)),
                "subtalker_temperature": float(task.get("subtalker_temperature", 0.9)),
            }
        )

    if not texts:
        print("[Error] No valid tasks to process")
        sys.exit(1)

    model_path = _ensure_model_dir(resource_dir, MODEL_DIR_DESIGN_NAME)
    model = _load_model(model_path, resolved_device)

    if len(texts) > 1:
        common = gen_kwargs[0]
        if not all(kwargs == common for kwargs in gen_kwargs):
            print(
                "[Warning] Mixed generation params detected. Using first task's settings for batch."
            )
        gen_kwargs = [common] * len(texts)

    kwargs = gen_kwargs[0]
    wavs, sr = model.generate_voice_design(
        text=texts,
        language=languages,
        instruct=instructs,
        **kwargs,
    )

    for wav, output_path in zip(wavs, output_paths):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, wav, sr)
        print(f"[OK] Wrote {output_path}")


def custom_voice(json_file: str, resource_path: str, device: str):
    if not os.path.exists(json_file):
        print(f"[Error] JSON not found: {json_file}")
        sys.exit(1)
    if not os.path.exists(resource_path):
        print(f"[Error] Resource path not found: {resource_path}")
        sys.exit(1)

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"[Error] JSON load failed: {e}")
        sys.exit(1)

    if not isinstance(tasks, list):
        print("[Error] JSON root must be a list of tasks")
        sys.exit(1)

    _validate_torch_loaded()
    resolved_device = _detect_device(device)
    if resolved_device == "cpu":
        threads = _configure_torch_threads()
        print(f"[Info] CPU Mode: Using {threads} threads")

    base_dir = Path(json_file).resolve().parent
    resource_dir = Path(resource_path)

    print(f"[Info] Device: {resolved_device}")
    print(f"[Info] Tasks: {len(tasks)}")

    texts: List[str] = []
    languages: List[str] = []
    speakers: List[str] = []
    instructs: List[str] = []
    output_paths: List[str] = []
    gen_kwargs: List[dict] = []

    for task in tasks:
        text = task.get("text")
        output_path = task.get("audio_path") or task.get("output_path")
        speaker = task.get("speaker")
        if not text or not output_path or not speaker:
            print("[Warning] Skipped task: missing text/output_path/speaker")
            continue

        language = task.get("target_language", task.get("language", "Auto"))
        instruct = task.get("instruct", "")

        texts.append(_read_text_arg(text))
        languages.append(language)
        speakers.append(speaker)
        instructs.append(_read_text_arg(instruct) if instruct else "")
        output_paths.append(_resolve_path(output_path, base_dir))

        gen_kwargs.append(
            {
                "max_new_tokens": int(task.get("max_new_tokens", 1024)),
                "do_sample": bool(task.get("do_sample", True)),
                "top_k": int(task.get("top_k", 50)),
                "top_p": float(task.get("top_p", 1.0)),
                "temperature": float(task.get("temperature", 0.9)),
                "repetition_penalty": float(task.get("repetition_penalty", 1.05)),
                "subtalker_dosample": bool(task.get("subtalker_dosample", True)),
                "subtalker_top_k": int(task.get("subtalker_top_k", 50)),
                "subtalker_top_p": float(task.get("subtalker_top_p", 1.0)),
                "subtalker_temperature": float(task.get("subtalker_temperature", 0.9)),
            }
        )

    if not texts:
        print("[Error] No valid tasks to process")
        sys.exit(1)

    model_path = _ensure_model_dir(resource_dir, MODEL_DIR_CUSTOM_NAME)
    model = _load_model(model_path, resolved_device)

    if len(texts) > 1:
        common = gen_kwargs[0]
        if not all(kwargs == common for kwargs in gen_kwargs):
            print(
                "[Warning] Mixed generation params detected. Using first task's settings for batch."
            )
        gen_kwargs = [common] * len(texts)

    kwargs = gen_kwargs[0]
    wavs, sr = model.generate_custom_voice(
        text=texts,
        language=languages,
        speaker=speakers,
        instruct=instructs,
        **kwargs,
    )

    for wav, output_path in zip(wavs, output_paths):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, wav, sr)
        print(f"[OK] Wrote {output_path}")


def design_then_synthesize(json_file: str, resource_path: str, device: str):
    if not os.path.exists(json_file):
        print(f"[Error] JSON not found: {json_file}")
        sys.exit(1)
    if not os.path.exists(resource_path):
        print(f"[Error] Resource path not found: {resource_path}")
        sys.exit(1)

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"[Error] JSON load failed: {e}")
        sys.exit(1)

    if not isinstance(tasks, list):
        print("[Error] JSON root must be a list of tasks")
        sys.exit(1)

    _validate_torch_loaded()
    resolved_device = _detect_device(device)
    if resolved_device == "cpu":
        threads = _configure_torch_threads()
        print(f"[Info] CPU Mode: Using {threads} threads")

    base_dir = Path(json_file).resolve().parent
    resource_dir = Path(resource_path)

    print(f"[Info] Device: {resolved_device}")
    print(f"[Info] Tasks: {len(tasks)}")

    design_model_path = _ensure_model_dir(resource_dir, MODEL_DIR_DESIGN_NAME)
    clone_model_path = _ensure_model_dir(resource_dir, MODEL_DIR_BASE_NAME)

    design_model = _load_model(design_model_path, resolved_device)
    clone_model = _load_model(clone_model_path, resolved_device)

    for index, task in enumerate(tasks, start=1):
        design_text = task.get("design_text")
        design_instruct = task.get("design_instruct", "")
        design_language = task.get("design_language", "Auto")
        texts = task.get("texts")
        output_paths = task.get("output_paths")

        if not design_text or not texts or not output_paths:
            print("[Warning] Skipped task: missing design_text/texts/output_paths")
            continue
        if not isinstance(texts, list) or not isinstance(output_paths, list):
            print("[Warning] Skipped task: texts/output_paths must be lists")
            continue
        if len(texts) != len(output_paths):
            print("[Warning] Skipped task: texts/output_paths length mismatch")
            continue

        languages = task.get("languages")
        if languages is None:
            languages = [design_language] * len(texts)
        if not isinstance(languages, list) or len(languages) != len(texts):
            print(
                "[Warning] Skipped task: languages must be a list matching texts length"
            )
            continue

        design_kwargs = {
            "max_new_tokens": int(task.get("design_max_new_tokens", 256)),
            "do_sample": bool(task.get("design_do_sample", True)),
            "top_k": int(task.get("design_top_k", 50)),
            "top_p": float(task.get("design_top_p", 1.0)),
            "temperature": float(task.get("design_temperature", 0.9)),
            "repetition_penalty": float(task.get("design_repetition_penalty", 1.05)),
            "subtalker_dosample": bool(task.get("design_subtalker_dosample", True)),
            "subtalker_top_k": int(task.get("design_subtalker_top_k", 50)),
            "subtalker_top_p": float(task.get("design_subtalker_top_p", 1.0)),
            "subtalker_temperature": float(
                task.get("design_subtalker_temperature", 0.9)
            ),
        }

        clone_kwargs = {
            "max_new_tokens": int(task.get("max_new_tokens", 256)),
            "do_sample": bool(task.get("do_sample", True)),
            "top_k": int(task.get("top_k", 50)),
            "top_p": float(task.get("top_p", 1.0)),
            "temperature": float(task.get("temperature", 0.9)),
            "repetition_penalty": float(task.get("repetition_penalty", 1.05)),
            "subtalker_dosample": bool(task.get("subtalker_dosample", True)),
            "subtalker_top_k": int(task.get("subtalker_top_k", 50)),
            "subtalker_top_p": float(task.get("subtalker_top_p", 1.0)),
            "subtalker_temperature": float(task.get("subtalker_temperature", 0.9)),
        }

        design_wavs, design_sr = design_model.generate_voice_design(
            text=_read_text_arg(design_text),
            language=design_language,
            instruct=_read_text_arg(design_instruct) if design_instruct else "",
            **design_kwargs,
        )

        voice_clone_prompt = clone_model.create_voice_clone_prompt(
            ref_audio=(design_wavs[0], design_sr),
            ref_text=_read_text_arg(design_text),
        )

        wavs, sr = clone_model.generate_voice_clone(
            text=[_read_text_arg(t) for t in texts],
            language=languages,
            voice_clone_prompt=voice_clone_prompt,
            **clone_kwargs,
        )

        for wav, output_path in zip(wavs, output_paths):
            resolved_output = _resolve_path(output_path, base_dir)
            os.makedirs(
                os.path.dirname(os.path.abspath(resolved_output)), exist_ok=True
            )
            sf.write(resolved_output, wav, sr)
            print(f"[Task {index}] [OK] Wrote {resolved_output}")


def main():
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Qwen3-TTS - Text-to-Speech and Voice Design CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_qwen3_tts.py --function synthesize_speech --json_file mock/tasks_synthesize.json --resource_path /path/to/cache --device auto --torch_path /path/to/torch
  python run_qwen3_tts.py --function design_voice --json_file mock/tasks_design.json --resource_path /path/to/cache --device cuda --torch_path /path/to/torch --nvidia_libs_path /path/to/nvidia
""",
    )

    parser.add_argument(
        "--function",
        required=True,
        choices=[
            "synthesize_speech",
            "design_voice",
            "custom_voice",
            "design_then_synthesize",
        ],
        help="Function to execute",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps"
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="Path to input JSON file with tasks",
    )
    parser.add_argument(
        "--resource_path", type=str, required=True, help="Path to model cache directory"
    )
    parser.add_argument(
        "--nvidia_libs_path",
        type=str,
        default=None,
        help="Path to NVIDIA CUDA libraries (cudnn/cublas/etc) to load dynamically",
    )
    parser.add_argument(
        "--torch_path",
        type=str,
        required=True,
        help="Path to external PyTorch installation (must contain torch/ and torch/lib)",
    )

    args = parser.parse_args()

    if args.function == "synthesize_speech":
        synthesize_speech(args.json_file, args.resource_path, args.device)
    elif args.function == "design_voice":
        design_voice(args.json_file, args.resource_path, args.device)
    elif args.function == "custom_voice":
        custom_voice(args.json_file, args.resource_path, args.device)
    elif args.function == "design_then_synthesize":
        design_then_synthesize(args.json_file, args.resource_path, args.device)
    else:
        print(f"[Error] Unknown function '{args.function}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
