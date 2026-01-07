import os
import sys
import json
import argparse
import warnings
import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort

# Suppress warnings
warnings.filterwarnings("ignore")

def find_cuda_lib_paths(root_path):
    found_paths = set()
    required_prefixes = ('libcudnn.so', 'libcublas.so', 'libcufft.so')
    for root, dirs, files in os.walk(root_path):
        if any(f.startswith(required_prefixes) for f in files):
            found_paths.add(os.path.abspath(root))
    return list(found_paths)

def ensure_cuda_env(cuda_path):
    if not cuda_path or not os.path.exists(cuda_path): return
    if sys.platform == 'win32':
        for root, dirs, files in os.walk(cuda_path):
            if any(f.endswith('.dll') for f in files):
                try: os.add_dll_directory(root)
                except: pass
        os.environ['PATH'] = cuda_path + os.pathsep + os.environ['PATH']
        return

    lib_paths = find_cuda_lib_paths(cuda_path)
    if not lib_paths: return

    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    missing = [p for p in lib_paths if p not in current_ld]

    if missing:
        new_ld = os.pathsep.join(missing + [current_ld]) if current_ld else os.pathsep.join(missing)
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = new_ld
        if getattr(sys, 'frozen', False):
            os.execve(sys.executable, [sys.executable] + sys.argv[1:], env)
        else:
            os.execve(sys.executable, [sys.executable] + sys.argv, env)
        sys.exit(0)

class MDXNetSeparator:
    def __init__(self, model_path, use_gpu=True):
        self.model_path = model_path
        self.dim_f = 3072
        self.dim_t = 256
        self.n_fft = 6144
        self.hop = 1024

        # --- SMART THREADING CONFIGURATION ---
        total_cores = os.cpu_count() or 2

        # Reserve cores for OS
        if total_cores >= 8:
            num_threads = total_cores - 2
        elif total_cores >= 4:
            num_threads = total_cores - 1
        else:
            num_threads = 1

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 'intra_op': Parallelize the heavy math inside the model
        sess_options.intra_op_num_threads = num_threads
        # 'inter_op': Sequential execution (better for audio time-series)
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = ['CPUExecutionProvider']
        if use_gpu:
            providers.insert(0, 'CUDAExecutionProvider')

        try:
            self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        except Exception as e:
            print(f"[Warning] Loading model failed ({e}), falling back to CPU.")
            self.session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

        # Configure Batching
        active_provider = self.session.get_providers()[0]
        self.using_cpu = 'CPU' in active_provider

        if self.using_cpu:
            print(f"[Info] CPU Mode: Using {num_threads} threads with Batch Size 8.")
            self.batch_size = 8
        else:
            print(f"[Info] GPU Mode: {active_provider}")
            self.batch_size = 1

    def _stft(self, audio):
        return librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop, window='hann', center=True)

    def _istft(self, spec):
        return librosa.istft(spec, hop_length=self.hop, window='hann', n_fft=self.n_fft, center=True)

    def process_audio(self, audio_path, vocal_path, instrumental_path, aggression):
        mix, sr = librosa.load(audio_path, sr=44100, mono=False)
        if mix.ndim == 1: mix = np.stack([mix, mix])

        spec = self._stft(mix)

        # Padding
        if spec.shape[1] > self.dim_f:
            spec_cropped = spec[:, :self.dim_f, :]
        elif spec.shape[1] < self.dim_f:
            pad = np.zeros((spec.shape[0], self.dim_f - spec.shape[1], spec.shape[2]))
            spec_cropped = np.concatenate([spec, pad], axis=1)
        else:
            spec_cropped = spec

        input_tensor = np.concatenate([spec_cropped.real, spec_cropped.imag], axis=0)

        chunk_size = self.dim_t
        step = chunk_size // 2
        pad_frames = chunk_size
        input_padded = np.pad(input_tensor, ((0,0), (0,0), (pad_frames, pad_frames)), mode='reflect')

        output_buffer = np.zeros_like(input_padded)
        count_buffer = np.zeros(input_padded.shape[2])
        window = np.hanning(chunk_size)

        input_name = self.session.get_inputs()[0].name

        batch_inputs = []
        batch_indices = []

        total_frames = input_padded.shape[2]

        # Inference Loop
        for i in range(0, total_frames - chunk_size, step):
            start = i
            end = i + chunk_size
            chunk = input_padded[:, :, start:end]

            if chunk.shape[2] < chunk_size: continue

            batch_inputs.append(chunk)
            batch_indices.append((start, end))

            # Run Batch if full
            if len(batch_inputs) >= self.batch_size:
                self._run_batch(batch_inputs, batch_indices, input_name, output_buffer, count_buffer, window)
                batch_inputs = []
                batch_indices = []

        # Run remaining
        if len(batch_inputs) > 0:
            self._run_batch(batch_inputs, batch_indices, input_name, output_buffer, count_buffer, window)

        count_buffer[count_buffer < 1e-6] = 1.0
        output_buffer /= count_buffer[None, None, :]

        original_length = input_tensor.shape[2]
        output_buffer = output_buffer[:, :, pad_frames:pad_frames + original_length]

        est_spec = output_buffer[0:2, :, :] + 1j * output_buffer[2:4, :, :]
        wav_inst = self._istft(est_spec)

        # Aggressive Vocals Isolation
        if est_spec.shape[1] < spec.shape[1]:
            pad = np.zeros((est_spec.shape[0], spec.shape[1] - est_spec.shape[1], est_spec.shape[2]))
            est_spec_full = np.concatenate([est_spec, pad], axis=1)
        else:
            est_spec_full = est_spec

        mix_mag = np.abs(spec)
        mix_phase = np.angle(spec)
        inst_mag = np.abs(est_spec_full)

        vocal_mag = mix_mag - (inst_mag * aggression)
        vocal_mag = np.maximum(vocal_mag, 0)
        vocal_spec = vocal_mag * np.exp(1j * mix_phase)
        wav_vocals = self._istft(vocal_spec)

        length = min(mix.shape[1], wav_inst.shape[1], wav_vocals.shape[1])
        sf.write(instrumental_path, wav_inst[:, :length].T, sr)
        sf.write(vocal_path, wav_vocals[:, :length].T, sr)

    def _run_batch(self, inputs, indices, input_name, out_buf, count_buf, window):
        stack = np.stack(inputs).astype(np.float32)
        predictions = self.session.run(None, {input_name: stack})[0]

        for i, pred_chunk in enumerate(predictions):
            start, end = indices[i]
            out_buf[:, :, start:end] += pred_chunk * window
            count_buf[start:end] += window

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, required=True)
    parser.add_argument('--json_file', type=str, required=True)
    parser.add_argument('--resource_path', type=str, required=True)
    parser.add_argument('--cuda_runtime_path', type=str, default=None)
    args = parser.parse_args()

    if args.cuda_runtime_path:
        ensure_cuda_env(args.cuda_runtime_path)

    try:
        with open(args.json_file, 'r') as f:
            tasks = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    print(f"Loading model from {args.resource_path}...")
    try:
        separator = MDXNetSeparator(args.resource_path, use_gpu=(args.cuda_runtime_path is not None))
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Processing {len(tasks)} tasks...")
    success = 0
    for i, task in enumerate(tasks):
        input_path = task.get('audio_path')
        vocal_out = task.get('vocal_output_path')
        inst_out = task.get('instrumental_output_path')

        # EXTRACT PER-TASK AGGRESSION
        # Default to 1.3 if not specified in JSON
        aggression = float(task.get('aggression', 1.3))

        if not input_path or not os.path.exists(input_path):
            print(f"Skipping task {i}: Input not found.")
            continue

        print(f"[{i+1}/{len(tasks)}] Processing: {os.path.basename(input_path)} (Aggression: {aggression})")
        try:
            separator.process_audio(input_path, vocal_out, inst_out, aggression)
            success += 1
        except Exception as e:
            print(f"Error task {i}: {e}")
            import traceback
            traceback.print_exc()

    print(f"Batch processing complete. {success}/{len(tasks)} successful.")

if __name__ == "__main__":
    main()
