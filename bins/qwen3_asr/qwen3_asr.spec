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
# COLLECT PYTORCH LIBRARIES
# =============================================================================

def collect_torch_libs():
    """Collect PyTorch libraries needed for CUDA support."""
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    torch_libs = []

    if not os.path.exists(torch_lib):
        print(f"[WARNING] PyTorch lib directory not found: {torch_lib}")
        return torch_libs

    print(f"[INFO] Scanning PyTorch libs: {torch_lib}")

    seen = set()
    critical_patterns = [
        'libc10.so*',
        'libc10_cuda.so*',
        'libtorch.so*',
        'libtorch_cpu.so*',
        'libtorch_cuda.so*',
        'libtorch_python.so*',
        'libtorch_global_deps.so*',
        'libshm.so*',
    ]

    for pattern in critical_patterns:
        for lib_path in glob.glob(os.path.join(torch_lib, pattern)):
            if os.path.isfile(lib_path):
                lib_name = os.path.basename(lib_path)
                if lib_name not in seen:
                    seen.add(lib_name)
                    torch_libs.append((lib_name, lib_path, 'BINARY'))
                    print(f"[INFO] Including PyTorch lib: {lib_name}")

    return torch_libs

torch_binaries = collect_torch_libs()

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
]

# ONLY exclude NVIDIA packages
excludes = [
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
# FILTER BINARIES - Remove only NVIDIA CUDA libs
# =============================================================================

filtered_binaries = []
excluded_count = 0

for dest, src, typecode in a.binaries:
    binary_name = os.path.basename(src).lower()
    should_exclude = False

    # Keep PyTorch libs
    is_torch_lib = any(
        os.path.basename(torch_lib[1]).lower() == binary_name
        for torch_lib in torch_binaries
    )

    if is_torch_lib or 'libtorch' in binary_name or 'libc10' in binary_name or 'libshm' in binary_name:
        if not any(os.path.basename(f[1]).lower() == binary_name for f in filtered_binaries):
            filtered_binaries.append((dest, src, typecode))
        continue

    # Exclude ONLY NVIDIA CUDA libs
    for pattern in excluded_binaries:
        if pattern.lower() in binary_name:
            should_exclude = True
            excluded_count += 1
            print(f"[INFO] Excluding NVIDIA lib: {binary_name}")
            break

    if not should_exclude:
        filtered_binaries.append((dest, src, typecode))

# Add torch binaries
existing_basenames = {os.path.basename(f[1]).lower() for f in filtered_binaries}

for torch_lib in torch_binaries:
    lib_basename = os.path.basename(torch_lib[1]).lower()
    if lib_basename not in existing_basenames:
        filtered_binaries.append(torch_lib)
        print(f"[INFO] Adding PyTorch lib to bundle: {torch_lib[0]}")

print(f"[INFO] Excluded {excluded_count} NVIDIA libraries")
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
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    onefile=True,
)
