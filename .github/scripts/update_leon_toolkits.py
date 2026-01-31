#!/usr/bin/env python3
"""
Leon Toolkit Updater
Updates toolkit.json files in the leon-ai/leon repository when binaries are updated.
"""

import datetime
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List


def load_config() -> Dict:
    """Load binaries configuration."""
    config_path = Path(__file__).parent.parent / "config" / "binaries.json"
    with open(config_path) as f:
        return json.load(f)


# Hardcoded URL template for leon-binaries releases
# Format: https://github.com/leon-ai/leon-binaries/releases/download/{binary_name}-v{version}/{binary_name}_{version}-{os}-{arch}{ext}
BINARIES_URL_TEMPLATE = "https://github.com/leon-ai/leon-binaries/releases/download/{binary_name}-v{version}/{binary_name}_{version}-{os}-{arch}{ext}"


def clone_leon_repo(leon_repo_config: Dict, token: str, clone_dir: Path) -> Path:
    """Clone or update the Leon repository."""
    owner = leon_repo_config['owner']
    name = leon_repo_config['name']
    branch = leon_repo_config.get('branch', 'develop')

    repo_url = f"https://x-access-token:{token}@github.com/{owner}/{name}.git"
    clone_path = clone_dir / name

    # Clone if it doesn't exist, otherwise pull
    if clone_path.exists():
        print(f"  🔄 Updating existing Leon repository...")
        subprocess.run(['git', '-C', str(clone_path), 'fetch', 'origin', str(branch)],
                      check=True)
        subprocess.run(['git', '-C', str(clone_path), 'checkout', str(branch)],
                      check=True)
        subprocess.run(['git', '-C', str(clone_path), 'pull', 'origin', str(branch)],
                      check=True)
    else:
        print(f"  📥 Cloning Leon repository...")
        clone_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--depth', '1', '-b', str(branch),
                        repo_url, str(clone_path)], check=True)

    return clone_path


def update_toolkit_json(toolkit_path: Path, tool_name: str, binary_name: str,
                       version: str) -> bool:
    """
    Update a toolkit.json file with new binary URLs.

    Args:
        toolkit_path: Path to the toolkit.json file
        tool_name: Name of the tool within the toolkit (e.g., "ytdlp")
        binary_name: Name of the binary (e.g., "yt-dlp")
        version: New binary version

    Returns:
        True if file was modified, False otherwise
    """
    with open(toolkit_path, 'r') as f:
        toolkit_data = json.load(f)

    # Check if tool exists
    if tool_name not in toolkit_data.get('tools', {}):
        print(f"  ⚠️  Tool '{tool_name}' not found in toolkit")
        return False

    tool_config = toolkit_data['tools'][tool_name]
    binaries = tool_config.get('binaries', {})

    if not binaries:
        print(f"  ⚠️  No binaries found for tool '{tool_name}'")
        return False

    # Track if we made any changes
    modified = False

    # Update each binary URL
    for platform, old_url in binaries.items():
        # Parse the old URL to extract OS, arch, and extension
        # URL format: https://github.com/leon-ai/leon-binaries/releases/download/{binary_name}_{version}/{binary_name}_{version}-{os}-{arch}{ext}
        #
        # Examples:
        # - opencode: opencode_1.1.34/opencode_1.1.34-linux-x86_64.tar.gz
        # - yt-dlp: yt-dlp_2025.12.08/yt-dlp_2025.12.08-linux-x86_64

        # Get just the filename part (after the last '/')
        filename = old_url.split('/')[-1]

        # Pattern: {binary_name}_{version}-{os}-{arch}{ext}
        # Examples:
        # - opencode_1.1.34-linux-x86_64.tar.gz
        # - yt-dlp_2025.12.08-linux-x86_64
        # - ffmpeg_7.0.2-win-amd64.exe

        # Strategy: Parse using known patterns
        # Format is: {binary_name}_{version}-{os}-{arch}{ext}
        # Find the version first (it separates the binary name from OS-ARCH-EXT)

        # Version pattern: numbers.dots (e.g., 1.1.34, 2025.12.08, 7.0.2)
        version_pattern = r'(?P<version>\d+(\.\d+)+)'

        # Match the pattern to extract version
        match = re.search(r'_(?P<version>\d+(\.\d+)+)_', filename)

        if not match:
            # Alternative: version might be at the end of the first segment
            match = re.search(r'_(?P<version>\d+(\.\d+)+)-', filename)

        if not match:
            print(f"  ⚠️  Could not parse URL for {platform}: {old_url}")
            print(f"       Version pattern not found")
            continue

        version_str = match.group('version')
        print(f"       Found version: {version_str}")

        # Get everything after the version
        after_version_start = match.end()

        # Extract the OS-ARCH-EXT part (everything after version)
        # First remove the version from the filename to isolate OS-ARCH-EXT
        filename_after_version = filename[after_version_start:]

        # Remove leading underscores or hyphens
        filename_after_version = filename_after_version.lstrip('_-')

        if not filename_after_version:
            print(f"  ⚠️  Could not parse URL for {platform}: {old_url}")
            print(f"       Nothing after version")
            continue

        print(f"       After version: '{filename_after_version}'")

        # Now extract OS, ARCH, EXT using pattern matching
        # OS names are: linux, macosx, win (case insensitive)
        os_pattern = r'(linux|macosx|win)'
        os_match = re.search(os_pattern, filename_after_version, re.IGNORECASE)

        if not os_match:
            print(f"  ⚠️  Could not parse URL for {platform}: {old_url}")
            print(f"       OS not found in: {filename_after_version}")
            continue

        os_name = os_match.group(1).lower()
        print(f"       Found OS: {os_name}")

        # Everything after OS is ARCH.EXT
        after_os_start = os_match.end()
        after_os = filename_after_version[after_os_start:].lstrip('-')

        # ARCH and EXT
        # Check if there's an extension
        if '.' in after_os:
            dot_pos = after_os.find('.')
            arch = after_os[:dot_pos]
            ext = after_os[dot_pos:]
        else:
            arch = after_os
            ext = ''

        # Verify we found OS and ARCH
        if not os_name or not arch:
            print(f"  ⚠️  Could not parse URL for {platform}: {old_url}")
            print(f"       Could not extract OS or ARCH")
            continue

        print(f"       Extracted: OS={os_name}, ARCH={arch}, EXT={ext}")

        # Build new URL
        new_url = BINARIES_URL_TEMPLATE.format(
            binary_name=binary_name,
            version=version,
            os=os_name,
            arch=arch,
            ext=ext
        )

        if old_url != new_url:
            binaries[platform] = new_url
            modified = True
            print(f"    ✏️  Updated {platform}: {old_url}")

    if modified:
        # Write updated JSON back
        with open(toolkit_path, 'w') as f:
            json.dump(toolkit_data, f, indent=2)
        print(f"  ✅ Updated toolkit: {toolkit_path}")
        return True
    else:
        print(f"  ℹ️  No changes needed for {tool_name}")
        return False


def commit_and_push_changes(repo_path: Path, updates: List[str]) -> bool:
    """
    Commit and push changes to a repository on a feature branch.

    Args:
        repo_path: Path to the repository
        updates: List of strings describing the updates made

    Returns:
        True if commit was successful, False otherwise
    """
    try:
        # Configure git
        subprocess.run(['git', '-C', str(repo_path), 'config',
                       'user.name', 'github-actions[bot]'], check=True)
        subprocess.run(['git', '-C', str(repo_path), 'config',
                       'user.email', 'github-actions[bot]@users.noreply.github.com'],
                      check=True)

        # Check for changes
        result = subprocess.run(['git', '-C', str(repo_path), 'status',
                               '--porcelain'], capture_output=True, text=True)

        if not result.stdout.strip():
            print("  ℹ️  No changes to commit")
            return False

        # Create a unique branch name
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        branch_name = f"chore/update-binaries-{timestamp}"

        print(f"  🌿 Creating branch: {branch_name}")

        # Create and checkout new branch
        subprocess.run(['git', '-C', str(repo_path), 'checkout', '-b', branch_name],
                      check=True)

        # Add all changes
        subprocess.run(['git', '-C', str(repo_path), 'add', '.'], check=True)

        # Commit with descriptive message
        update_summary = ', '.join(updates[:3])
        if len(updates) > 3:
            update_summary += f' and {len(updates) - 3} more'

        commit_message = f"chore: update toolkit binary URLs ({update_summary})"

        subprocess.run(['git', '-C', str(repo_path), 'commit', '-m', commit_message],
                      check=True)

        # Push branch to origin
        result = subprocess.run(['git', '-C', str(repo_path), 'push', '-u', 'origin', branch_name],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✅ Successfully pushed branch: {branch_name}")
            print(f"  📋 Please review and merge this branch manually")
            return True
        else:
            print(f"  ❌ Failed to push: {result.stderr}")
            return False

    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error during git operations: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("🔧 Leon Toolkit Updater")
    print("="*60 + "\n")

    # Get GitHub token
    leon_token = os.environ.get('LEON_REPO_TOKEN')
    if not leon_token:
        print("❌ LEON_REPO_TOKEN environment variable not set")
        return

    # Load configuration
    config = load_config()

    # Check if there's any binary_toolkit_mappings configured
    toolkit_mappings = config.get('binary_toolkit_mappings', {})
    if not toolkit_mappings:
        print("ℹ️  No toolkit mappings configured, skipping Leon repository update")
        return

    # Get binary versions from the update script
    workspace = Path(os.environ.get('GITHUB_WORKSPACE', '.'))
    versions_file = workspace / ".github" / "data" / "binary_versions.json"

    if not versions_file.exists():
        print("❌ Binary versions file not found")
        return

    with open(versions_file) as f:
        versions = json.load(f)

    leon_repo_config = config.get('leon_repo', {})
    if not leon_repo_config:
        print("⚠️  Leon repository configuration not found")
        return

    print(f"📦 Checking for toolkit updates in {leon_repo_config['owner']}/{leon_repo_config['name']}")

    # Clone/update Leon repository
    clone_dir = Path("/tmp") / "leon_repos"
    leon_repo_path = clone_leon_repo(leon_repo_config, leon_token, clone_dir)

    updates_made = []

    # Process each binary that has a toolkit mapping
    for binary_name, mapping in toolkit_mappings.items():
        if binary_name not in versions:
            print(f"  ℹ️  Skipping {binary_name} (not in versions)")
            continue

        version = versions[binary_name]
        toolkit_rel_path = mapping['toolkit_path']
        tool_name = mapping['tool_name']

        toolkit_path = leon_repo_path / toolkit_rel_path

        if not toolkit_path.exists():
            print(f"  ⚠️  Toolkit file not found: {toolkit_rel_path}")
            continue

        print(f"\n📝 Processing {binary_name} (v{version})...")
        print(f"     Toolkit: {toolkit_rel_path}")
        print(f"     Tool: {tool_name}")

        if update_toolkit_json(toolkit_path, tool_name, binary_name, version):
            updates_made.append(f"{binary_name} → v{version}")

    # Commit and push if there were changes
    if updates_made:
        print(f"\n{'='*60}")
        print(f"📤 Committing {len(updates_made)} update(s) to Leon repository...")
        print(f"{'='*60}\n")

        commit_and_push_changes(leon_repo_path, updates_made)
    else:
        print("\n✅ No toolkit updates needed")


if __name__ == "__main__":
    main()
