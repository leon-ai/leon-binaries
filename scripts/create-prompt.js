/**
 * It will generate a prompt that can then
 * be passed to an agentic coding solution as OpenCode.
 */

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const PYTHON_VERSION = '3.11.9'
const TOOL_NAME = 'Qwen3-TTS'
const TOOL_DESCRIPTION = 'The goal is to be able to do speech synthesize (text-to-speech) and voice cloning.'
const BINARY_NAME = 'qwen3_tts'
const PURPOSE_REQUIREMENTS = `
- You must understand the usage and docs before creating the tool binary: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- You can use the \`chatterbox_onnx\` binary as a reference since it has very similar use case.
`
const CLI_ARGUMENTS = `
  - \`--function\`
  - \`--device\`
  - \`--json_file\`. The data structure must be similar to the one from \`chatterbox_onnx\`.
  - \`--resource_path\`. Download the resources to \`/home/louis/Downloads\` for the tests.
  - \`--nvidia_libs_path\`. Use \`/home/louis/Workspace/leon/leon/bin/nvidia/\` for the tests.
`

/**
 * Reads a markdown template file, replaces placeholders with actual values,
 * and saves the result to the scripts/out folder
 * @param {string} templatePath - Path to the markdown template file
 * @returns {string} Path to the generated output file
 */
function createMarkdownFromTemplate(templatePath) {
  // Read the template file
  const templateContent = fs.readFileSync(templatePath, 'utf-8')
  
  // Define placeholder replacements
  const replacements = {
    '{PYTHON_VERSION}': PYTHON_VERSION,
    '{TOOL_NAME}': TOOL_NAME,
    '{TOOL_DESCRIPTION}': TOOL_DESCRIPTION,
    '{BINARY_NAME}': BINARY_NAME,
    '{PURPOSE_REQUIREMENTS}': PURPOSE_REQUIREMENTS,
    '{CLI_ARGUMENTS}': CLI_ARGUMENTS,
    '{CHOOSE_A_DESCRIPTION}': TOOL_DESCRIPTION
  }
  
  // Replace all placeholders
  let outputContent = templateContent
  for (const [placeholder, value] of Object.entries(replacements)) {
    outputContent = outputContent.replaceAll(placeholder, value)
  }
  
  // Create output directory if it doesn't exist
  const outDir = path.join(__dirname, 'out')
  if (!fs.existsSync(outDir)) {
    fs.mkdirSync(outDir, { recursive: true })
  }
  
  // Generate output file path
  const templateFileName = path.basename(templatePath, '.md')
  const outputFileName = templateFileName.replace('-template', '') + '.md'
  const outputPath = path.join(outDir, outputFileName)
  
  // Write the output file
  fs.writeFileSync(outputPath, outputContent, 'utf-8')
  
  console.log(`✓ Generated: ${outputPath}`)
  return outputPath
}

// Run the function with the template
const templatePath = path.join(__dirname, 'prompt-templates', 'create-new-tool-binaries-template.md')
createMarkdownFromTemplate(templatePath)
