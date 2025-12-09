# Leon Binaries for Tools

## Release a New Binary

1. Go to the Actions tab
2. “Build Binaries”
3. Run workflow
4. Set “bin” to the folder name under `bins/` (e.g., faster_whisper).
5. The job will:
   - Set up Python and uv.
   - Run npm run build {bin} using your scripts/build.js.
   - Read bins/{bin}/version.py to get version. 
   - Upload a single artifact per architecture named: {bin}_{version}-{arch}[.exe].

## Debug GitHub Action in Local Env

- https://nektosact.com/installation/index.html
