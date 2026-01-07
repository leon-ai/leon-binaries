# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files

PACKAGE_NAME = 'ultimate_vocal_remover_onnx'
RUN_MAIN_SCRIPT = 'run_ultimate_vocal_remover_onnx.py'

spec_root = os.getcwd()
main_script = os.path.join(spec_root, RUN_MAIN_SCRIPT)

block_cipher = None

# --- HIDDEN IMPORTS ---
hidden_imports = [
    # Core Dependencies
    'sklearn.utils._cython_blas',
    'sklearn.neighbors.typedefs',
    'sklearn.neighbors.quad_tree',
    'sklearn.tree._utils',
    'sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel',
    'scipy.special.cython_special',
    'scipy.spatial.transform._rotation_groups',
    'onnxruntime',
    'backports',
    'backports.tarfile',
    'jaraco.text',
    'jaraco.classes',
    'jaraco.context',
    'jaraco.functools',
    'more_itertools',
    'platformdirs',
    'packaging',
    'packaging.version',
    'packaging.specifiers',
    'packaging.requirements',
]

# --- DATAS ---
datas = []
datas += collect_data_files('librosa')
datas += collect_data_files('soundfile')
datas += collect_data_files('jaraco')

a = Analysis(
    [main_script],
    pathex=[spec_root],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # --- EXCLUDES ---
    excludes=[
        'torch', 'torchaudio', 'torchvision',
        'matplotlib', 'tkinter', 'doctest', 'unittest',
        'pdb',
        # 'distutils', # Do NOT exclude distutils (Setuptools needs it)
        # 'setuptools', # Do NOT exclude setuptools (Librosa needs it)
        'cv2', 'PIL',
        'pandas', 'PyQt5', 'pyside2'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# =============================================================================
#  AGGRESSIVE BINARY FILTERING (SIZE OPTIMIZATION)
# =============================================================================
exclusion_patterns = [
    'tensorrt', 'openvino', 'dnnl', 'tvm', 'rocm', 'cann', 'azure', 'triton',
    'nvinfer', 'nvparsers', 'libnvinfer', 'libnvonnxparser',
    'cudnn', 'cublas', 'cufft', 'curand', 'libtbb'
]

def is_excluded(filename):
    fname = filename.lower()
    return any(pattern in fname for pattern in exclusion_patterns)

# Filter Binaries and Datas
a.binaries = [x for x in a.binaries if not is_excluded(x[0])]
a.datas = [x for x in a.datas if not is_excluded(x[0])]
# =============================================================================

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ultimate_vocal_remover_onnx',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
