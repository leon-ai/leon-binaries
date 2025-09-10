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

  const lockPath = path.join(binDir, 'Pipfile.lock')
  const hasLock = fs.existsSync(lockPath)

  // Install dependencies in the Pipenv environment
  console.log(hasLock ? 'Installing with pipenv (locked)...' : 'Installing with pipenv...')
  run(process.env.PYTHON || 'python', ['-m', 'pipenv', 'install', ...(hasLock ? ['--deploy'] : [])], {
    cwd: binDir
  })

  // Execute: pipenv run pyinstaller <bin>.spec (using cwd instead of `cd`)
  console.log('Running PyInstaller...')
  run(process.env.PYTHON || 'python', ['-m', 'pipenv', 'run', 'pyinstaller', `${binName}.spec`], {
    cwd: binDir
  })

  console.log('Build completed.')
}

main().catch((err) => {
  console.error(err)

  process.exit(1)
})
