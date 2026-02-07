# -*- mode: python ; coding: utf-8 -*-

import os
import sys
import glob

# =============================================================================
# CONFIGURATION
# =============================================================================

PACKAGE_NAME = 'qwen3_asr'
RUN_MAIN_SCRIPT = 'run_qwen3_asr.py'

spec_root = os.getcwd()
main_script = os.path.join(spec_root, RUN_MAIN_SCRIPT)

# =============================================================================
# PYTORCH EXCLUSION
# =============================================================================
# PyTorch is excluded from the binary to reduce size.
# Use --torch_path argument at runtime to load PyTorch from an external location.

# =============================================================================
# COLLECT PACKAGE DATA FILES
# =============================================================================

def collect_package_data(package_name):
    """
    Collect data files from a package.
    Returns list of (source_path, dest_dir) tuples.
    dest_dir is relative to the bundle root (e.g., 'qwen_asr/inference/assets')
    """
    datas = []

    try:
        pkg = __import__(package_name)
        pkg_dir = os.path.dirname(pkg.__file__)

        # Data directories to look for
        data_dirs = ['data', 'assets', 'config', 'models', 'dictionaries']

        for data_dir_name in data_dirs:
            full_data_dir = os.path.join(pkg_dir, data_dir_name)
            if os.path.exists(full_data_dir):
                for root, dirs, files in os.walk(full_data_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        # Calculate relative path from package root
                        rel_path = os.path.relpath(full_path, pkg_dir)
                        # Destination is the directory containing the file
                        dest_dir = os.path.join(package_name, os.path.dirname(rel_path))
                        datas.append((full_path, dest_dir))
                        print(f"[INFO] {package_name} data: {rel_path} -> {dest_dir}")

        # Also check for files directly in package root that might be data
        for item in os.listdir(pkg_dir):
            item_path = os.path.join(pkg_dir, item)
            if os.path.isfile(item_path) and item.endswith(('.dict', '.txt', '.json', '.yaml', '.yml')):
                datas.append((item_path, package_name))
                print(f"[INFO] {package_name} root data: {item} -> {package_name}")

    except ImportError:
        print(f"[WARNING] {package_name} not installed")

    return datas

def collect_specific_files(package_name, relative_paths):
    """
    Collect specific files by relative path within a package.
    relative_paths: list of paths like ['inference/assets/korean_dict_jieba.dict']
    """
    datas = []

    try:
        pkg = __import__(package_name)
        pkg_dir = os.path.dirname(pkg.__file__)

        for rel_path in relative_paths:
            full_path = os.path.join(pkg_dir, rel_path)
            if os.path.exists(full_path):
                dest_dir = os.path.join(package_name, os.path.dirname(rel_path))
                datas.append((full_path, dest_dir))
                print(f"[INFO] {package_name} specific: {rel_path} -> {dest_dir}")
            else:
                print(f"[WARNING] Not found: {full_path}")
    except ImportError:
        print(f"[WARNING] {package_name} not installed")

    return datas

# Collect all data files
all_data = []

# Nagisa data
all_data.extend(collect_package_data('nagisa'))

# Qwen-asr data - try automatic collection first
all_data.extend(collect_package_data('qwen_asr'))

# Also add specific known files to ensure they're included
qwen_asr_specific = [
    'inference/assets/korean_dict_jieba.dict',
    'inference/assets/chinese_dict_jieba.dict',  # Likely also needed
]
all_data.extend(collect_specific_files('qwen_asr', qwen_asr_specific))

# =============================================================================
# NVIDIA CUDA LIBRARIES TO EXCLUDE (Provided manually)
# =============================================================================

excluded_binaries = [
    'cudart', 'cudart64', 'cudart32', 'libcudart',
    'cublas', 'cublas64', 'cublasLt', 'cublasLt64', 'libcublas', 'libcublasLt',
    'cudnn', 'cudnn64', 'cudnn_adv', 'cudnn_cnn', 'cudnn_ops', 'cudnn_graph',
    'libcudnn',
    'cusparse', 'cusparse64', 'cusparseLt', 'libcusparse', 'libcusparseLt',
    'nvrtc', 'nvrtc64', 'nvrtc-builtins', 'libnvrtc',
    'nvToolsExt', 'nvToolsExt64', 'libnvToolsExt',
    'cuda', 'cuda64', 'libcuda', 'nvcuda',
    'curand', 'curand64', 'libcurand',
    'cusolver', 'cusolver64', 'libcusolver',
    'cufft', 'cufft64', 'libcufft',
    'cupti', 'cupti64', 'libcupti',
    'nvinfer', 'nvinfer64', 'libnvinfer',
    'nccl', 'nccl64', 'libnccl',
]

# =============================================================================
# ANALYSIS
# =============================================================================

hiddenimports = [
    # Standard library
    'backports.tarfile',

    # Nagisa and ALL its submodules
    'nagisa',
    'nagisa.prepro',
    'nagisa.train',
    'nagisa.utils',
    'nagisa.tagger',
    'nagisa.mecab_system_eval',
    'nagisa.mecab_element',
    'nagisa.mecab_cost_learner',
    'nagisa.mecab_tokenizer',
    'nagisa.nagisa_utils',

    # Qwen ASR
    'qwen_asr',

    # Jieba (Chinese text segmentation - likely used by qwen_asr)
    'jieba',
    'jieba.posseg',
    'jieba.analyse',
    
    # Dependencies needed by external PyTorch
    'sympy',
    'mpmath',  # sympy dependency
]

# Exclude NVIDIA packages and PyTorch to reduce binary size
excludes = [
    # NVIDIA CUDA packages
    'nvidia.cuda_runtime',
    'nvidia.cublas',
    'nvidia.cudnn',
    'nvidia.cusparse',
    'nvidia.curand',
    'nvidia.cusolver',
    'nvidia.cufft',
    'nvidia.cuda_nvrtc',
    'nvidia.nvtx',
    'nvidia.nccl',
    
    # PyTorch and related packages (load dynamically at runtime)
    'torch',
    'torch.distributed',
    'torch.nn',
    'torch.optim',
    'torch.utils',
    'torch.autograd',
    'torch.cuda',
    'torch.jit',
    'torch.onnx',
    'torch.quantization',
    'torch.sparse',
    'torch.futures',
    'torch.fx',
    'torch._C',
    'torch._dynamo',
    'torch._inductor',
    'torch._functorch',
    'torchvision',
    'torchaudio',
    'caffe2',
]

a = Analysis(
    [main_script],
    pathex=[spec_root],
    binaries=[],
    datas=all_data,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

# =============================================================================
# FILTER PYTHON MODULES - Remove ALL torch Python modules
# =============================================================================

print("[INFO] Filtering Python modules (a.pure)...")
filtered_pure = []
excluded_torch_modules = 0

for name, src, typecode in a.pure:
    # Exclude anything that starts with 'torch'
    if name.startswith('torch') or name.startswith('caffe2'):
        excluded_torch_modules += 1
        print(f"[INFO] Excluding torch module: {name}")
    else:
        filtered_pure.append((name, src, typecode))

print(f"[INFO] Excluded {excluded_torch_modules} torch Python modules")
print(f"[INFO] Remaining Python modules: {len(filtered_pure)}")

a.pure = filtered_pure

# =============================================================================
# FILTER BINARIES - Remove NVIDIA CUDA libs and PyTorch libs
# =============================================================================

print("[INFO] Filtering binary libraries...")
filtered_binaries = []
excluded_cuda_count = 0
excluded_torch_count = 0

for dest, src, typecode in a.binaries:
    binary_name = os.path.basename(src).lower()
    should_exclude = False

    # Exclude PyTorch libraries (more comprehensive patterns)
    # NOTE: libgomp is NOT excluded - it's needed by qwen_asr!
    torch_patterns = [
        'libtorch', 'libc10', 'libshm', 'torch_python', 'torch_cpu', 'torch_cuda',
        'c10.so', 'c10_cuda', 'torch.so', '_C.', 'torch_global_deps',
        'libnvfuser', 'libtorch_global', 'libcaffe2', 'libnvshmem',
        'libtorch_python', 'libmkl',  # libgomp REMOVED - needed by qwen_asr
    ]
    
    if any(pattern in binary_name for pattern in torch_patterns):
        should_exclude = True
        excluded_torch_count += 1
        print(f"[INFO] Excluding PyTorch lib: {binary_name}")
    
    # Exclude NVIDIA CUDA libs
    if not should_exclude:
        for pattern in excluded_binaries:
            if pattern.lower() in binary_name:
                should_exclude = True
                excluded_cuda_count += 1
                print(f"[INFO] Excluding NVIDIA lib: {binary_name}")
                break

    if not should_exclude:
        filtered_binaries.append((dest, src, typecode))

print(f"[INFO] Excluded {excluded_cuda_count} NVIDIA libraries")
print(f"[INFO] Excluded {excluded_torch_count} PyTorch libraries")
print(f"[INFO] Total binaries to bundle: {len(filtered_binaries)}")
print(f"[INFO] Total data files: {len(all_data)}")

a.binaries = filtered_binaries

# =============================================================================
# BUILD
# =============================================================================

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=PACKAGE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    onefile=True,
)
