# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path

VERSION = '1.0.0'

# Get the directory where this spec file is located
spec_root = Path(__file__).parent.absolute()

# Define the main script path
main_script = spec_root / "faster_whisper.py"

# Analysis configuration
a = Analysis(
    [str(main_script)],
    pathex=[str(spec_root)],
    binaries=[],
    datas=[
        # Add any data files needed by faster-whisper here
        # Example: ('path/to/data', 'destination/in/bundle')
    ],
    hiddenimports=[
        # Core faster-whisper dependencies
        'faster_whisper',
        'faster_whisper.transcribe',
        'faster_whisper.audio',
        'faster_whisper.utils',
        'faster_whisper.vad',
        
        # CTranslate2 dependencies
        'ctranslate2',
        'ctranslate2._C',
        
        # Audio processing dependencies
        'av',
        'av.audio',
        'av.container',
        'av.codec',
        'av.stream',
        
        # Tokenization and NLP
        'tokenizers',
        'tokenizers.implementations',
        'tokenizers.models',
        'tokenizers.normalizers',
        'tokenizers.pre_tokenizers',
        'tokenizers.processors',
        'tokenizers.trainers',
        'tokenizers.decoders',
        
        # Hugging Face transformers
        'transformers',
        'transformers.models',
        'transformers.models.whisper',
        'transformers.tokenization_utils',
        
        # NumPy and scientific computing
        'numpy',
        'numpy.core',
        'numpy.core._methods',
        'numpy.lib',
        'numpy.lib.format',
        
        # Torch (if using CUDA)
        'torch',
        'torch.nn',
        'torch.cuda',
        'torch._C',
        'torch._dynamo',
        
        # Other common dependencies
        'pathlib',
        'typing',
        'typing_extensions',
        'packaging',
        'packaging.version',
        'packaging.specifiers',
        'packaging.requirements',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce binary size
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'jupyter',
        'notebook',
        'ipython',
        'pandas',
        'scipy',
        'sklearn',
        'PIL',
        'cv2',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# PYZ archive
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Executable configuration
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='faster_whisper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Use UPX compression to reduce binary size
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Console application
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add path to .ico file if you have one
)

# For creating a single directory distribution (alternative to single file)
# Uncomment the following if you prefer a directory distribution instead of single file
"""
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='faster_whisper'
)
"""
