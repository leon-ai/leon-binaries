# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Chatterbox ONNX

This spec file builds a standalone binary that excludes CUDA libraries (cuDNN, cuBLAS)
since they are loaded externally via the --cuda_runtime_path CLI argument.
"""

import os

# =============================================================================
# CONFIGURATION
# =============================================================================

PACKAGE_NAME = 'chatterbox_onnx'
RUN_MAIN_SCRIPT = 'run_chatterbox_onnx.py'

spec_root = os.getcwd()
main_script = os.path.join(spec_root, RUN_MAIN_SCRIPT)

# =============================================================================
# DATA COLLECTION
# =============================================================================

def _collect_data_files():
    """Collect all data files to include in the binary."""
    datas = []
    
    # Default voices
    default_voices_path = os.path.join(spec_root, 'default_voices')
    if os.path.exists(default_voices_path):
        for file in os.listdir(default_voices_path):
            if file.endswith('.wav'):
                datas.append((
                    os.path.join(default_voices_path, file),
                    'default_voices'
                ))
    
    # pkuseg data (Chinese segmentation)
    try:
        import pkuseg
        pkuseg_path = os.path.dirname(pkuseg.__file__)
        for subdir in ['dicts', 'models']:
            pkuseg_data_path = os.path.join(pkuseg_path, subdir)
            if os.path.exists(pkuseg_data_path):
                datas.append((pkuseg_data_path, f'pkuseg/{subdir}'))
    except ImportError:
        pass
    
    # pykakasi data (Japanese conversion)
    try:
        import pykakasi
        pykakasi_path = os.path.dirname(pykakasi.__file__)
        pykakasi_data = os.path.join(pykakasi_path, 'data')
        if os.path.exists(pykakasi_data):
            datas.append((pykakasi_data, 'pykakasi/data'))
    except ImportError:
        pass
    
    return datas

# =============================================================================
# CUDA LIBRARY FILTERING
# =============================================================================

def _filter_cuda_libraries(items):
    """
    Filter out CUDA libraries from binaries and data.
    
    CUDA libraries (cuDNN, cuBLAS) are loaded externally via
    --cuda_runtime_path CLI argument and should NOT be bundled.
    """
    filtered = []
    for item in items:
        name = item[0] if isinstance(item, tuple) else str(item)
        
        # Exclude CUDA libraries
        if 'cudnn' in name.lower() or 'cublas' in name.lower():
            continue
        
        filtered.append(item)
    
    return filtered

# =============================================================================
# PYINSTALLER ANALYSIS
# =============================================================================

a = Analysis(
    [main_script],
    pathex=[spec_root],
    binaries=[],
    datas=_collect_data_files(),
    hiddenimports=[
        'onnxruntime.capi._pybind_state',
        'backports',
        'backports.tarfile',
        'jaraco',
        'jaraco.text',
        'jaraco.context',
        'jaraco.functools',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'tkinter', 'PyQt5', 'PyQt6', 'jupyter', 'notebook',
        'torch', 'tensorflow',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# =============================================================================
# FILTER CUDA LIBRARIES
# =============================================================================

a.binaries = _filter_cuda_libraries(a.binaries)
a.datas = _filter_cuda_libraries(a.datas)

# =============================================================================
# BUILD EXECUTABLE
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
    strip=True,           # Remove debug symbols for smaller, faster-loading binary
    upx=False,            # Disable UPX compression (adds significant build time)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    optimize=2,            # Optimize Python bytecode (0=none, 1=optimize, 2=optimize+remove docstrings)
)
