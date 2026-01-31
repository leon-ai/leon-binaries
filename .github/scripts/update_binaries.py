#!/usr/bin/env python3
"""
Leon Binary Updater
Checks for new releases of specified binaries and uploads them to leon-binaries repo
with standardized naming convention.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
import requests
from github import Github, GithubException


def load_config() -> Dict:
    """Load binaries configuration."""
    config_path = Path(__file__).parent.parent / "config" / "binaries.json"
    with open(config_path) as f:
        return json.load(f)


def get_version_tracking_file(repo_path: str) -> Path:
    """Get path to version tracking file."""
    return Path(repo_path) / ".github" / "data" / "binary_versions.json"


def load_tracked_versions(tracking_file: Path) -> Dict[str, str]:
    """Load previously tracked versions."""
    if tracking_file.exists():
        with open(tracking_file) as f:
            return json.load(f)
    return {}


def save_tracked_versions(tracking_file: Path, versions: Dict[str, str]):
    """Save tracked versions to file."""
    tracking_file.parent.mkdir(parents=True, exist_ok=True)
    with open(tracking_file, 'w') as f:
        json.dump(versions, f, indent=2)


def get_latest_release(repo: str, github_token: str) -> Optional[Dict]:
    """Get latest release info from a GitHub repository."""
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        print(f"⚠️  No releases found for {repo}")
        return None
    else:
        print(f"❌ Error fetching release for {repo}: {response.status_code}")
        return None


def normalize_version(version: str) -> str:
    """Normalize version string by removing 'v' prefix."""
    return version.lstrip('v')


def find_matching_asset(assets: List[Dict], pattern: str) -> Optional[Dict]:
    """Find asset matching the given pattern."""
    # Convert pattern to regex (handle wildcards if needed)
    pattern_regex = pattern.replace(".", r"\.").replace("*", ".*")
    
    for asset in assets:
        if re.match(f"^{pattern_regex}$", asset['name']):
            return asset
    
    return None


def download_asset(asset: Dict, download_dir: Path, github_token: str) -> Optional[Path]:
    """Download a release asset."""
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/octet-stream'
    }
    
    print(f"  📥 Downloading {asset['name']}...")
    
    response = requests.get(asset['url'], headers=headers, stream=True, allow_redirects=True)
    
    if response.status_code == 200:
        file_path = download_dir / asset['name']
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return file_path
    else:
        print(f"  ❌ Failed to download {asset['name']}: {response.status_code}")
        return None


def create_target_filename(binary_name: str, version: str, os_name: str, arch: str, ext: str) -> str:
    """Create standardized filename for target repo."""
    base = f"{binary_name}_{version}-{os_name}-{arch}"
    if ext:
        return f"{base}.{ext}" if not ext.startswith('.') else f"{base}{ext}"
    return base


def upload_to_binaries_repo(
    gh: Github,
    target_repo: str,
    binary_name: str,
    version: str,
    files: List[tuple]
) -> bool:
    """Upload binaries to leon-binaries repository."""
    try:
        repo = gh.get_repo(target_repo)
        
        # Create release tag
        tag_name = f"{binary_name}-v{version}"
        release_name = f"{binary_name} v{version}"
        release_body = f"Automated release for {binary_name} version {version}. Includes raw binaries for multiple architectures."
        
        # Check if release already exists
        try:
            existing_release = repo.get_release(tag_name)
            print(f"  ℹ️  Release {tag_name} already exists, updating assets...")
            release = existing_release
        except GithubException:
            # Create new release
            print(f"  📦 Creating release {tag_name}...")
            release = repo.create_git_release(
                tag=tag_name,
                name=release_name,
                message=release_body,
                draft=False,
                prerelease=False
            )
        
        # Upload assets
        for file_path, target_name in files:
            print(f"  📤 Uploading {target_name}...")
            
            # Check if asset already exists and delete it
            for asset in release.get_assets():
                if asset.name == target_name:
                    print(f"  🗑️  Removing existing asset {target_name}...")
                    asset.delete_asset()
            
            with open(file_path, 'rb') as f:
                release.upload_asset(
                    path=str(file_path),
                    label=target_name,
                    name=target_name
                )
        
        print(f"  ✅ Successfully uploaded {binary_name} v{version}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error uploading to {target_repo}: {str(e)}")
        return False


def process_binary(
    binary_config: Dict,
    target_repo: str,
    tracked_versions: Dict[str, str],
    github_token: str,
    gh: Github
) -> tuple[bool, Optional[str]]:
    """Process a single binary update."""
    binary_name = binary_config['name']
    source_repo = binary_config['source_repo']
    
    print(f"\n🔍 Checking {binary_name} ({source_repo})...")
    
    # Get latest release
    release = get_latest_release(source_repo, github_token)
    if not release:
        return False, None
    
    version = normalize_version(release['tag_name'])
    last_version = tracked_versions.get(binary_name)
    
    if last_version == version:
        print(f"  ✓ Already up to date (v{version})")
        return False, version
    
    print(f"  🆕 New version found: v{version} (previous: {last_version or 'none'})")
    
    # Download and rename assets
    download_dir = Path("/tmp") / "leon_binaries" / binary_name
    download_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_upload = []
    
    for mapping in binary_config['asset_mappings']:
        asset = find_matching_asset(release['assets'], mapping['source_pattern'])
        
        if not asset:
            print(f"  ⚠️  Asset not found: {mapping['source_pattern']}")
            continue
        
        # Download asset
        downloaded_file = download_asset(asset, download_dir, github_token)
        if not downloaded_file:
            continue
        
        # Create target filename
        target_name = create_target_filename(
            binary_name,
            version,
            mapping['target_os'],
            mapping['target_arch'],
            mapping['target_ext']
        )
        
        files_to_upload.append((downloaded_file, target_name))
    
    if not files_to_upload:
        print(f"  ❌ No assets downloaded for {binary_name}")
        return False, None
    
    # Upload to leon-binaries
    success = upload_to_binaries_repo(gh, target_repo, binary_name, version, files_to_upload)
    
    # Cleanup
    for file_path, _ in files_to_upload:
        file_path.unlink(missing_ok=True)
    
    return success, version if success else None


def create_summary(updates: Dict[str, str], errors: List[str]):
    """Create GitHub Actions job summary."""
    summary_lines = ["# Binary Update Summary\n"]
    
    if updates:
        summary_lines.append("## ✅ Updated Binaries\n")
        for binary, version in updates.items():
            summary_lines.append(f"- **{binary}**: v{version}")
        summary_lines.append("")
    
    if errors:
        summary_lines.append("## ❌ Errors\n")
        for error in errors:
            summary_lines.append(f"- {error}")
        summary_lines.append("")
    
    if not updates and not errors:
        summary_lines.append("✓ All binaries are up to date. No updates needed.\n")
    
    summary_path = Path("/tmp/binary_update_summary.md")
    summary_path.write_text("\n".join(summary_lines))


def main():
    """Main execution function."""
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        print("❌ GITHUB_TOKEN environment variable not set")
        sys.exit(1)
    
    # Load configuration
    config = load_config()
    target_repo = config['target_repo']
    
    # Initialize GitHub client
    gh = Github(github_token)
    
    # Load tracked versions
    workspace = Path(os.environ.get('GITHUB_WORKSPACE', '.'))
    tracking_file = get_version_tracking_file(workspace)
    tracked_versions = load_tracked_versions(tracking_file)
    
    print("🚀 Leon Binary Updater")
    print(f"📦 Target repository: {target_repo}")
    print(f"🔢 Monitoring {len(config['binaries'])} binaries\n")
    
    updates = {}
    errors = []
    
    # Process each binary
    for binary_config in config['binaries']:
        try:
            success, new_version = process_binary(
                binary_config,
                target_repo,
                tracked_versions,
                github_token,
                gh
            )
            
            if success and new_version:
                binary_name = binary_config['name']
                tracked_versions[binary_name] = new_version
                updates[binary_name] = new_version
                
        except Exception as e:
            error_msg = f"{binary_config['name']}: {str(e)}"
            print(f"\n❌ Error: {error_msg}")
            errors.append(error_msg)
    
    # Save updated versions
    if updates:
        save_tracked_versions(tracking_file, tracked_versions)
        
        # Commit version file back to repo if running in CI
        if os.environ.get('GITHUB_ACTIONS'):
            print("\n📝 Committing version tracking file...")
            os.system(f"git config user.name 'github-actions[bot]'")
            os.system(f"git config user.email 'github-actions[bot]@users.noreply.github.com'")
            os.system(f"git add {tracking_file}")
            os.system(f"git commit -m 'chore: update binary versions'")
            os.system(f"git push")
    
    # Create summary
    create_summary(updates, errors)
    
    print("\n" + "="*60)
    print(f"✅ Updated: {len(updates)} | ❌ Errors: {len(errors)}")
    print("="*60)
    
    # Exit with error if there were any errors
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
