#!/usr/bin/env python3
"""
Chatterbox ONNX - Text-to-Speech Synthesis CLI Tool

This module provides high-performance TTS inference using ONNX Runtime with
automatic CUDA/CPU fallback and multi-core optimization.
"""

import os
import sys
import argparse

# =============================================================================
# CUDA LIBRARY LOADING (Must happen before importing ONNX Runtime)
# =============================================================================

def _parse_cuda_path_early():
    """Parse --cuda_runtime_path from argv before argparse runs."""
    for i, arg in enumerate(sys.argv):
        if arg == "--cuda_runtime_path" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

def _load_cuda_libraries(cuda_runtime_path):
    """
    Attempt to load CUDA libraries from the specified path or nvidia-* packages.
    Returns True if libraries were successfully loaded.
    """
    if cuda_runtime_path:
        return _load_cuda_from_path(cuda_runtime_path)
    elif sys.platform == "linux":
        return _load_cuda_from_nvidia_packages()
    return False

def _load_cuda_from_path(cuda_runtime_path):
    """Load CUDA libraries from a custom path."""
    import ctypes
    import glob
    
    print(f"🔧 Using custom CUDA runtime path: {cuda_runtime_path}")
    
    if sys.platform == "win32":
        cudnn_lib = os.path.join(cuda_runtime_path, "cudnn", "bin")
        cublas_lib = os.path.join(cuda_runtime_path, "cublas", "bin")
        cudnn_pattern, cublas_pattern = "cudnn*.dll", "cublas*.dll"
    else:
        cudnn_lib = os.path.join(cuda_runtime_path, "cudnn", "lib")
        cublas_lib = os.path.join(cuda_runtime_path, "cublas", "lib")
        cudnn_pattern, cublas_pattern = "libcudnn.so.*", "libcublas.so.*"
    
    if not (os.path.exists(cudnn_lib) and os.path.exists(cublas_lib)):
        expected = "'cudnn/bin' and 'cublas/bin'" if sys.platform == "win32" else "'cudnn/lib' and 'cublas/lib'"
        print(f"⚠️ CUDA runtime path invalid. Expected {expected} subdirectories.")
        print("⚠️ Will fallback to CPU mode with multi-core optimization.")
        return False
    
    # Update library search path
    if sys.platform == "win32":
        os.environ["PATH"] = f"{cudnn_lib};{cublas_lib};{os.environ.get('PATH', '')}"
    else:
        os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{cublas_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    try:
        cudnn_libs = glob.glob(os.path.join(cudnn_lib, cudnn_pattern))
        cublas_libs = glob.glob(os.path.join(cublas_lib, cublas_pattern))
        
        if cudnn_libs:
            ctypes.CDLL(cudnn_libs[0])
            print(f"✅ Loaded cuDNN from: {cudnn_libs[0]}")
        if cublas_libs:
            ctypes.CDLL(cublas_libs[0])
            print(f"✅ Loaded cuBLAS from: {cublas_libs[0]}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to load CUDA libraries: {e}")
        print("⚠️ Will fallback to CPU mode with multi-core optimization.")
        return False

def _load_cuda_from_nvidia_packages():
    """Load CUDA libraries from nvidia-* Python packages (Linux only)."""
    try:
        import nvidia.cudnn
        import nvidia.cublas
        import ctypes
        
        def get_lib_path(module):
            if getattr(module, '__file__', None):
                return os.path.join(os.path.dirname(module.__file__), "lib")
            elif hasattr(module, '__path__'):
                return os.path.join(list(module.__path__)[0], "lib")
            return ""
        
        cudnn_lib = get_lib_path(nvidia.cudnn)
        cublas_lib = get_lib_path(nvidia.cublas)
        
        if cudnn_lib and cublas_lib:
            os.environ["LD_LIBRARY_PATH"] = f"{cudnn_lib}:{cublas_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            ctypes.CDLL(os.path.join(cudnn_lib, "libcudnn.so.9"))
            ctypes.CDLL(os.path.join(cublas_lib, "libcublas.so.12"))
            return True
    except (ImportError, Exception):
        pass
    return False

# Initialize CUDA before importing ONNX Runtime
_cuda_runtime_path = _parse_cuda_path_early()
_cuda_libs_loaded = _load_cuda_libraries(_cuda_runtime_path)
_cuda_path_failed = _cuda_runtime_path and not _cuda_libs_loaded

# =============================================================================
# IMPORTS
# =============================================================================

import json
import threading
import time
import unicodedata
from unicodedata import category

import concurrent.futures
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562

# Model architecture constants
NUM_LAYERS = 30
NUM_HEADS = 16
HEAD_DIM = 64

# =============================================================================
# GLOBAL STATE
# =============================================================================

class GlobalState:
    """Container for shared global state."""
    models = {}
    tokenizer = None
    voice_cache = {}
    cache_lock = threading.Lock()
    cangjie_converter = None
    kakasi = None
    dicta = None
    models_dir = None
    accelerator = None
    max_concurrent_jobs = 1
    threads_per_job = 1

_state = GlobalState()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

def _get_base_path():
    """Get the base path for resources (handles PyInstaller frozen apps)."""
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        if os.path.exists(os.path.join(exe_dir, "default_voices")):
            return exe_dir
        return getattr(sys, '_MEIPASS', exe_dir)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = _get_base_path()
DEFAULT_VOICES_DIR = os.path.join(BASE_DIR, "default_voices")

# =============================================================================
# HARDWARE DETECTION & OPTIMIZATION
# =============================================================================

def _detect_accelerator():
    """Detect available hardware accelerator."""
    if _cuda_path_failed:
        print("⚠️ CUDA path was provided but libraries failed to load. Using CPU mode.")
        return 'CPU'
    
    providers = ort.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        return 'CUDA'
    elif 'CoreMLExecutionProvider' in providers:
        return 'CoreML'
    return 'CPU'

def _calculate_optimal_settings(accelerator):
    """
    Calculate optimal thread/job settings for the detected hardware.
    
    For TTS inference (autoregressive generation):
    - Memory-bound workload (large model weights, KV cache)
    - Sequential per-task (each token depends on previous)
    - Parallelizable within ONNX operations (matrix multiplications)
    """
    if accelerator in ('CUDA', 'CoreML'):
        return 1, 1
    
    total_cores = os.cpu_count() or 1
    reserved_cores = max(1, min(2, total_cores // 8))
    available_cores = max(1, total_cores - reserved_cores)
    
    # Optimize based on core count to avoid memory bandwidth saturation
    if available_cores <= 8:
        concurrent_jobs, threads_per_job = 1, available_cores
    elif available_cores <= 16:
        concurrent_jobs, threads_per_job = 2, available_cores // 2
    else:
        concurrent_jobs = min(4, available_cores // 6)
        threads_per_job = available_cores // concurrent_jobs
    
    threads_per_job = max(2, threads_per_job)
    concurrent_jobs = max(1, concurrent_jobs)
    
    print(f"💻 CPU Mode: {total_cores} cores detected, using {available_cores} cores")
    print(f"   Strategy: {concurrent_jobs} parallel job(s) × {threads_per_job} threads each")
    
    return concurrent_jobs, threads_per_job

def _create_session_options(accelerator, threads_per_job):
    """Create optimized ONNX Runtime session options."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    if accelerator == 'CUDA':
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
    else:
        opts.intra_op_num_threads = threads_per_job
        opts.inter_op_num_threads = max(1, threads_per_job // 2)
        opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True
    
    return opts

def _get_providers(accelerator):
    """Get ONNX Runtime execution providers for the accelerator."""
    if accelerator == 'CUDA':
        return [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested',
                'gpu_mem_limit': 0,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]
    elif accelerator == 'CoreML':
        return ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    return ['CPUExecutionProvider']

# =============================================================================
# TEXT PROCESSING
# =============================================================================

class RepetitionPenaltyProcessor:
    """Applies repetition penalty to logits during token generation."""
    
    def __init__(self, penalty: float = 1.2):
        self.penalty = penalty
    
    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        score = np.take_along_axis(scores, input_ids, axis=1)
        score = np.where(score < 0, score * self.penalty, score / self.penalty)
        result = scores.copy()
        np.put_along_axis(result, input_ids, score, axis=1)
        return result

class ChineseCangjieConverter:
    """Converts Chinese text to Cangjie encoding for TTS."""
    
    def __init__(self, models_dir):
        self.word2cj = {}
        self.cj2word = {}
        self.segmenter = None
        self._load_mapping(models_dir)
        self._init_segmenter()
    
    def _load_mapping(self, models_dir):
        try:
            path = os.path.join(models_dir, "Cangjie5_TC.json")
            with open(path, "r", encoding="utf-8") as f:
                for entry in json.load(f):
                    word, code = entry.split("\t")[:2]
                    self.word2cj[word] = code
                    self.cj2word.setdefault(code, []).append(word)
        except Exception as e:
            print(f"❌ Could not load Cangjie mapping: {e}")
    
    def _init_segmenter(self):
        try:
            import pkuseg
            self.segmenter = pkuseg.pkuseg()
        except ImportError:
            print("⚠️ pkuseg not available - Chinese segmentation will be skipped")
    
    def _encode_char(self, char):
        code = self.word2cj.get(char)
        if code is None:
            return None
        try:
            idx = self.cj2word[code].index(char)
            suffix = str(idx) if idx > 0 else ""
            return code + suffix
        except ValueError:
            return code
    
    def __call__(self, text):
        if self.segmenter:
            text = " ".join(self.segmenter.cut(text))
        
        output = []
        for char in text:
            if category(char) == "Lo":
                cangjie = self._encode_char(char)
                if cangjie:
                    output.append("".join(f"[cj_{c}]" for c in cangjie) + "[cj_.]")
            elif char == " " or '\u0020' <= char <= '\u007E':
                output.append(char)
            else:
                output.append(" ")
        return "".join(output)

def _normalize_japanese(text):
    """Convert Japanese text to hiragana for TTS."""
    if _state.kakasi is None:
        try:
            import pykakasi
            _state.kakasi = pykakasi.kakasi()
        except ImportError:
            print("⚠️ pykakasi not available - Japanese text processing skipped")
            return text
    
    result = _state.kakasi.convert(text)
    output = []
    for r in result:
        orig, hira = r['orig'], r['hira']
        # Check for kanji (CJK Unified Ideographs)
        has_kanji = any(19968 <= ord(c) <= 40959 for c in orig)
        # Check for katakana
        all_katakana = all(12449 <= ord(c) <= 12538 for c in orig) if orig else False
        
        if has_kanji:
            if hira and hira[0] in "はへ":
                hira = " " + hira
            output.append(hira)
        elif all_katakana:
            output.append(orig)
        else:
            output.append(orig)
    
    return unicodedata.normalize('NFKD', "".join(output))

def _add_hebrew_diacritics(text):
    """Add diacritics to Hebrew text for TTS."""
    if _state.dicta is None:
        try:
            from dicta_onnx import Dicta
            _state.dicta = Dicta()
        except ImportError:
            print("⚠️ dicta_onnx not available - Hebrew text processing skipped")
            return text
        except Exception as e:
            print(f"⚠️ Hebrew diacritization failed: {e}")
            return text
    return _state.dicta.add_diacritics(text)

def _normalize_korean(text):
    """Decompose Korean Hangul syllables to Jamo for TTS."""
    def decompose(char):
        if not '\uac00' <= char <= '\ud7af':
            return char
        base = ord(char) - 0xAC00
        initial = chr(0x1100 + base // (21 * 28))
        medial = chr(0x1161 + (base % (21 * 28)) // 28)
        final = chr(0x11A7 + base % 28) if base % 28 > 0 else ''
        return initial + medial + final
    return ''.join(decompose(c) for c in text).strip()

def prepare_text(text, language):
    """Prepare text for TTS based on language."""
    if language == 'zh' and _state.cangjie_converter:
        text = _state.cangjie_converter(text)
    elif language == 'ja':
        text = _normalize_japanese(text)
    elif language == 'he':
        text = _add_hebrew_diacritics(text)
    elif language == 'ko':
        text = _normalize_korean(text)
    
    if language:
        text = f"[{language.lower()}]{text}"
    return text

# =============================================================================
# INFERENCE UTILITIES
# =============================================================================

def _softmax(x):
    """Compute softmax values for array x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def _sample_token(logits, temperature=0.5, top_k=50):
    """Sample a token from logits using temperature and top-k sampling."""
    logits = logits / temperature
    probs = _softmax(logits)
    
    # Top-k filtering
    threshold = np.partition(probs, -top_k)[-top_k]
    probs[probs < threshold] = 0
    probs /= probs.sum()
    
    return np.random.choice(len(probs), p=probs)

# =============================================================================
# MODEL LOADING
# =============================================================================

def _load_models(resource_path):
    """Load all ONNX models and tokenizer."""
    global _state
    
    _state.models_dir = resource_path
    if not os.path.exists(resource_path):
        raise FileNotFoundError(f"Resource path does not exist: {resource_path}")
    
    _state.accelerator = _detect_accelerator()
    _state.max_concurrent_jobs, _state.threads_per_job = _calculate_optimal_settings(_state.accelerator)
    
    print(f"📂 Loading models from: {resource_path}")
    print(f"🧠 Detected accelerator: {_state.accelerator}")
    
    providers = _get_providers(_state.accelerator)
    sess_opts = _create_session_options(_state.accelerator, _state.threads_per_job)
    
    # Model file mapping
    model_files = {
        "speech_encoder": "speech_encoder.onnx",
        "embed_tokens": "embed_tokens.onnx",
        "conditional_decoder": "conditional_decoder.onnx",
        "language_model": "language_model_q4.onnx",
    }
    
    # Resolve paths
    model_paths = {}
    for name, filename in model_files.items():
        path = os.path.join(resource_path, filename)
        if not os.path.exists(path):
            path = os.path.join(resource_path, "onnx", filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model file: {filename}")
        model_paths[name] = path
    
    # Load models with CUDA fallback
    cuda_failed = False
    for name, path in model_paths.items():
        try:
            _state.models[name] = ort.InferenceSession(path, sess_opts, providers=providers)
        except Exception as e:
            if _state.accelerator == 'CUDA' and not cuda_failed:
                print(f"❌ CUDA initialization failed: {e}")
                print("⚠️ Switching to CPU mode with multi-core optimization...")
                cuda_failed = True
                _state.accelerator = 'CPU'
                
                # Reconfigure for CPU
                providers = ['CPUExecutionProvider']
                _state.max_concurrent_jobs, _state.threads_per_job = _calculate_optimal_settings('CPU')
                sess_opts = _create_session_options('CPU', _state.threads_per_job)
                
                # Reload previously loaded models
                print("🔄 Reloading previously loaded models with CPU settings...")
                for prev_name, prev_path in model_paths.items():
                    if prev_name == name:
                        break
                    if prev_name in _state.models:
                        _state.models[prev_name] = ort.InferenceSession(prev_path, sess_opts, providers=providers)
                        print(f"   ✅ Reloaded {prev_name}")
                
                _state.models[name] = ort.InferenceSession(path, sess_opts, providers=providers)
            else:
                raise
    
    # Load tokenizer
    tokenizer_path = os.path.join(resource_path, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    
    print(f"📖 Loading tokenizer from: {tokenizer_path}")
    _state.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    _state.cangjie_converter = ChineseCangjieConverter(resource_path)

# =============================================================================
# VOICE HANDLING
# =============================================================================

def _resolve_voice_path(task):
    """Resolve the voice file path from task parameters."""
    lang = task.get("target_language", "en")
    ref_path = task.get("speaker_reference_path")
    voice_name = task.get("voice_name")
    
    # Custom reference path
    if ref_path:
        if os.path.exists(ref_path):
            return ref_path
        print(f"⚠️ Reference {ref_path} not found. Falling back.")
    
    # Named voice
    if voice_name:
        filename = voice_name if voice_name.endswith(".wav") else f"{voice_name}.wav"
        path = os.path.join(DEFAULT_VOICES_DIR, filename)
        if os.path.exists(path):
            return path
        print(f"⚠️ Voice '{voice_name}' not found.")
    
    # Default voice for language
    lang_code = "no" if lang == "nb" else lang
    for candidate in [f"{lang_code}_female.wav", f"{lang_code}_male.wav", "en_female.wav"]:
        path = os.path.join(DEFAULT_VOICES_DIR, candidate)
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"No voice found for language '{lang}'.")

def _get_voice_embedding(voice_path, cfg_strength):
    """Get or compute voice embedding with caching."""
    cache_key = (voice_path, cfg_strength)
    
    with _state.cache_lock:
        if cache_key in _state.voice_cache:
            return _state.voice_cache[cache_key]
    
    # Load and encode voice
    audio, _ = librosa.load(voice_path, sr=SAMPLE_RATE)
    audio = audio[np.newaxis, :].astype(np.float32)
    
    cond_emb, prompt_token, ref_x_vector, prompt_feat = _state.models["speech_encoder"].run(
        None, {"audio_values": audio}
    )
    cond_emb = cond_emb * cfg_strength
    
    result = (cond_emb, prompt_token, ref_x_vector, prompt_feat)
    
    with _state.cache_lock:
        _state.voice_cache[cache_key] = result
    
    return result

# =============================================================================
# SPEECH SYNTHESIS
# =============================================================================

def _generate_speech(task):
    """Generate speech for a single task."""
    text = task.get("text")
    lang = task.get("target_language", "en")
    output_path = task.get("audio_path")
    cfg_strength = task.get("cfg_strength", 0.5)
    exaggeration = task.get("exaggeration", 0.5)
    
    if not text or not output_path:
        return "❌ Skipped: Missing required fields"
    
    try:
        # Prepare
        voice_path = _resolve_voice_path(task)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Process text
        processed_text = prepare_text(text, lang)
        input_ids = _state.tokenizer(processed_text, return_tensors="np")["input_ids"].astype(np.int64)
        position_ids = np.where(
            input_ids >= START_SPEECH_TOKEN, 0,
            np.arange(input_ids.shape[1])[np.newaxis, :] - 1
        ).astype(np.int64)
        
        # Get voice embedding
        cond_emb, prompt_token, ref_x_vector, prompt_feat = _get_voice_embedding(voice_path, cfg_strength)
        
        # Initial text embedding
        text_embeds = _state.models["embed_tokens"].run(None, {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "exaggeration": np.array([exaggeration], dtype=np.float32)
        })[0]
        inputs_embeds = np.concatenate((cond_emb, text_embeds), axis=1)
        
        # Initialize KV cache
        batch_size, seq_len = 1, inputs_embeds.shape[1]
        past_kv = {
            f"past_key_values.{i}.{kv}": np.zeros([batch_size, NUM_HEADS, 0, HEAD_DIM], dtype=np.float32)
            for i in range(NUM_LAYERS) for kv in ("key", "value")
        }
        attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
        
        # Token generation
        generated = [START_SPEECH_TOKEN]
        penalty_processor = RepetitionPenaltyProcessor(penalty=1.2)
        max_tokens = min(200 + len(text) * 10, 1500)
        
        for step in range(max_tokens):
            # Run language model
            outputs = _state.models["language_model"].run(None, {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                **past_kv
            })
            logits, *new_kv = outputs
            
            # Sample next token
            gen_array = np.array([generated], dtype=np.int64)
            processed_logits = penalty_processor(gen_array, logits[:, -1, :])
            next_token = _sample_token(processed_logits[0])
            generated.append(next_token)
            
            if next_token == STOP_SPEECH_TOKEN:
                break
            
            # Prepare next iteration
            inputs_embeds = _state.models["embed_tokens"].run(None, {
                "input_ids": np.array([[next_token]], dtype=np.int64),
                "position_ids": np.array([[step + 1]], dtype=np.int64),
                "exaggeration": np.array([exaggeration], dtype=np.float32)
            })[0]
            
            attention_mask = np.concatenate([
                attention_mask, np.ones((batch_size, 1), dtype=np.int64)
            ], axis=1)
            
            for j, key in enumerate(past_kv):
                past_kv[key] = new_kv[j]
        
        # Decode speech
        speech_tokens = np.concatenate([
            prompt_token,
            np.array([generated[1:-1]], dtype=np.int64)
        ], axis=1)
        
        wav = _state.models["conditional_decoder"].run(None, {
            "speech_tokens": speech_tokens,
            "speaker_embeddings": ref_x_vector,
            "speaker_features": prompt_feat,
        })[0]
        
        sf.write(output_path, np.squeeze(wav), SAMPLE_RATE)
        return f"✅ Saved {output_path}"
    
    except Exception as e:
        return f"❌ Error: {e}"

# =============================================================================
# MAIN ENTRY POINTS
# =============================================================================

def synthesize_speech(json_file, resource_path):
    """Main function to synthesize speech from a JSON task file."""
    if not os.path.exists(json_file):
        print(f"❌ JSON not found: {json_file}")
        sys.exit(1)
    
    if not os.path.exists(resource_path):
        print(f"❌ Resource path not found: {resource_path}")
        sys.exit(1)
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"❌ JSON Error: {e}")
        sys.exit(1)
    
    print(f"📂 Default Voices: {DEFAULT_VOICES_DIR}")
    if not os.path.exists(DEFAULT_VOICES_DIR):
        print("⚠️ Warning: 'default_voices' folder missing.")
    
    _load_models(resource_path)
    
    print(f"\n🚀 Device: {_state.accelerator} | Workers: {_state.max_concurrent_jobs}")
    print(f"📂 Tasks: {len(tasks)}")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=_state.max_concurrent_jobs) as executor:
        futures = {executor.submit(_generate_speech, task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            msg = future.result()
            if "❌" in msg:
                tqdm.write(msg)
    
    print(f"\n✨ Done in {time.time() - start_time:.2f}s")

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Chatterbox ONNX - Text-to-Speech Synthesis CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_chatterbox_onnx.py --function synthesize_speech --json_file tasks.json --resource_path ./models
  python run_chatterbox_onnx.py --function synthesize_speech --json_file tasks.json --resource_path ./models --cuda_runtime_path /usr/local/cuda
        """
    )
    
    parser.add_argument("--function", required=True, choices=["synthesize_speech"],
                        help="Function to execute")
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to input JSON file with synthesis tasks")
    parser.add_argument("--resource_path", type=str, required=True,
                        help="Path to directory containing ONNX models and tokenizer")
    parser.add_argument("--cuda_runtime_path", type=str, default=None,
                        help="Path to custom CUDA runtime (should contain 'cudnn' and 'cublas' subdirectories)")
    
    args = parser.parse_args()
    
    if args.function == "synthesize_speech":
        synthesize_speech(args.json_file, args.resource_path)
    else:
        print(f"Error: Unknown function '{args.function}'")
        sys.exit(1)

if __name__ == "__main__":
    main()
