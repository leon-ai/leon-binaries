# -*- mode: python ; coding: utf-8 -*-

import os

PACKAGE_NAME = 'qwen3_tts'
RUN_MAIN_SCRIPT = 'run_qwen3_tts.py'

spec_root = os.getcwd()
main_script = os.path.join(spec_root, RUN_MAIN_SCRIPT)


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


def _filter_cuda_libraries(items):
    filtered = []
    for item in items:
        name = item[0] if isinstance(item, tuple) else str(item)
        lower = name.lower()
        if any(lib in lower for lib in excluded_binaries):
            continue
        filtered.append(item)
    return filtered


hiddenimports = [
    'backports.tarfile',
    'mpmath',
    'sympy',
    'qwen_tts',
    'qwen_tts.core',
    'qwen_tts.core.models',
    'qwen_tts.core.tokenizer_12hz',
    'qwen_tts.inference',
]

datas = [
    (os.path.join(spec_root, 'sox.py'), '.'),
    (os.path.join(spec_root, 'torchaudio', '__init__.py'), 'torchaudio'),
    (os.path.join(spec_root, 'torchaudio', 'compliance', '__init__.py'), 'torchaudio/compliance'),
    (os.path.join(spec_root, 'torchaudio', 'compliance', 'kaldi.py'), 'torchaudio/compliance'),
]

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
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

a.binaries = _filter_cuda_libraries(a.binaries)
a.datas = _filter_cuda_libraries(a.datas)

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
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    optimize=2,
)
