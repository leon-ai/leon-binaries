#!/usr/bin/env node
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

function run(cmd, args, options = {}) {
  const result = spawnSync(cmd, args, {
    stdio: 'inherit',
    shell: false,
    ...options
  })

  if (result.error) {
    throw result.error
  }
  if (typeof result.status === "number" && result.status !== 0) {
    process.exit(result.status)
  }
}

function usageAndExit() {
  console.error('Usage: npm run build <bin_name> [--bins-dir=<dir>]')
  console.error('Example: npm run build faster_whisper')

  process.exit(1)
}

async function main() {
  const args = process.argv.slice(2)

  if (args.length < 1) {
    usageAndExit()
  }

  const binName = args[0]
  const binsDirFlag = args.find((a) => a.startsWith('--bins-dir='))
  const binsDir = (binsDirFlag ? binsDirFlag.split('=')[1] : process.env.BINS_DIR) || 'bins'

  const repoRoot = process.cwd()
  const binDir = path.resolve(repoRoot, binsDir, binName)
  const specFile = path.join(binDir, `${binName}.spec`)

  if (!fs.existsSync(binDir) || !fs.statSync(binDir).isDirectory()) {
    console.error(`Error: bin directory not found: ${binDir}`)

    process.exit(1)
  }
  if (!fs.existsSync(specFile)) {
    console.error(`Error: spec file not found: ${specFile}`)

    process.exit(1)
  }

  console.log(`Building "${binName}" using PyInstaller...`)
  console.log(`- Working directory: ${binDir}`)
  console.log(`- Spec file: ${specFile}`)

  const pyprojectPath = path.join(binDir, 'pyproject.toml')
  const hasPyproject = fs.existsSync(pyprojectPath)


  // Install dependencies using uv
  console.log('Installing dependencies with uv...')
  const uvSyncArgs = ['sync', '--frozen']
  if (binName === 'qwen3_tts') {
    uvSyncArgs.push('--reinstall-package', 'onnxruntime')
  }
  run('uv', uvSyncArgs, {
    cwd: binDir
  })

  // Execute: uv run pyinstaller <bin>.spec
  console.log('Running PyInstaller...')
  run('uv', ['run', 'pyinstaller', `${binName}.spec`], {
    cwd: binDir
  })

  console.log('Build completed.')
}

main().catch((err) => {
  console.error(err)

  process.exit(1)
})
