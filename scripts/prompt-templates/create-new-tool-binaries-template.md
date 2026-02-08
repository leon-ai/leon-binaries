# Create New Tool Binaries for Leon AI

I'm developing Leon AI, an open-source personal AI assistant. It has a granular structure: skills > actions > tools > functions > binaries.

For example, you can imagine a `Video Translator` skill that contains multiple actions. One of the action can be the `transcribe_audio` action, which requires different tools that can transcribe the audio. E.g. `faster_whisper` tool, `qwen3_asr` tool, etc. Each of these tools could have a function called `transcribeAudioToFile` and these functions would call a binary to transcribe audio to file.

## Goal

Your goal is to create a new binary that will be used as a CLI from the core of Leon AI (repository is `leon-ai/leon`).

You must strictly follow the purpose requirement and technical requirements.

This `leon-ai/leon-binaries` repository already contains several binaries. Feel free to use these existing binaries for your reference to get a better understanding.

## Purpose Requirement

You must create a new CLI binary for `{TOOL_NAME}`. {TOOL_DESCRIPTION}
{PURPOSE_REQUIREMENTS}

## Technical Requirements

- The name must be `{BINARY_NAME}`.
- The CLI arguments must be: {CLI_ARGUMENTS}
- You must provide a `README.md` file about the usage. No other markdown file should be created.
- The following OSes and architectures must be supported:
  - Linux x86_64
  - Linux AArch64
  - macOS ARM64
  - macOS x86_64 (Intel)
  - Windows AMD64
- You must search if ONNX variation exists. This is a priority before falling back to PyTorch. You can search on Hugging Face or on the web if any ONNX version exists. If yes, then you must use it and do not use PyTorch. It will help the binary tool to be more portable, lighter and faster at runtime.
- You must use `uv`.
- You must use Python 3.11.9.
- You must respect the following standard file structure:
  - `mock/` # Folder containing mock files to test
  - `version.py`
  - `run_{BINARY_NAME}.py`
  - `pyproject.toml`
  - `{BINARY_NAME}.spec`
  - `README.md`
- Dependencies version must be locked. E.g. `"torch==2.9.0"`

### Code Requirements

- `version.py` must contain this code:
```python
__version__ = "1.0.0"
```
- When using CPU, you must ensure to use multiple cores according to the number of available cores. You can look at the `chatterbox_onnx` as reference.

### PyInstaller and `.spec` File Requirements

You must use PyInstaller to compile the source code into the binary. Follow these requirements:

- Install `pyinstaller==6.18.0` as a dev dependency.
- You must make sure no unnecessary libraries, data, deps, etc. are embedded into the binary. Try your best to make the binary light and have a minimal build time.
- The top of the spec file must contain the following:
```spec
# -*- mode: python ; coding: utf-8 -*-

# ...Imports here...

PACKAGE_NAME = '{BINARY_NAME}'
RUN_MAIN_SCRIPT = 'run_{BINARY_NAME}.py'

spec_root = os.getcwd()
main_script = os.path.join(spec_root, RUN_MAIN_SCRIPT)

# Reuse these variables

...
```

### pyproject.toml Requirements

This file must follow a specific standard. The following properties must be included:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{BINARY_NAME}"
dynamic = ["version"]
description = "{CHOOSE_A_DESCRIPTION}"
requires-python = "==3.11.9"
dependencies = [ ... ]

[tool.uv]
# Tell uv to resolve for multiple platforms to ensure compatibility
environments = [
  "sys_platform == 'linux' and platform_machine == 'x86_64'",
  "sys_platform == 'linux' and platform_machine == 'aarch64'",
  "sys_platform == 'darwin' and platform_machine == 'x86_64'",
  "sys_platform == 'darwin' and platform_machine == 'arm64'",
  "sys_platform == 'win32' and platform_machine == 'AMD64'",
]

[tool.hatch.build.targets.wheel]
packages = ["."]
only-include = ["run_{BINARY_NAME}.py", "version.py"]

[tool.hatch.version]
path = "version.py"

[dependency-groups]
dev = [ ... ]
```

### ONNX Requirements

If ONNX can be used, then follow the following requirements:

- You must install `onnxruntime-gpu==1.24.1` for Linux x86_64 and Windows AMD64. And `onnxruntime==1.24.1` for other platforms.

### PyTorch Requirements

If PyTorch is used, follow these requirements:

- You must use `torch==2.9.0` version, and the `2.2.2` for macOS Intel.
- You must install `llvmlite==0.43.0` for macOS Intel only. Because PyTorch has a strong dependency for this platform.

### Transformers Requirements

If Transformers is used, follow these requirements:

- Install the `transformers==4.57.6`. Do not install the latest version since too many projects do not support the next major version.

## Tests

- You must create mock files to test. These files must be saved under the `mock/` folder. E.g. `mock/tasks.json` so that you can do `--json_file mock/tasks.json`.
- You must run the code. If there is any error, you must fix it and try again. Use this command to run the code:
```bash
uv run run_{BINARY_NAME}.py
```
- You must build the binary. If there is any error you must fix it and try again. Use this command to build the binary:
```bash
pnpm run build {BINARY_NAME}
```
- Once the binary is built, you must run it. If there is any error, you must fix it and try again. Use the binary name and the CLI arguments to run the binary. You may need to create mock files
