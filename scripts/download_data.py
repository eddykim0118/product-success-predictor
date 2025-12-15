#!/usr/bin/env python
"""
Download RSNA Intracranial Aneurysm Detection Data
==================================================

Download competition data from Kaggle.
Requires Kaggle API credentials (~/.kaggle/kaggle.json)
"""

import os
import sys
import argparse
from pathlib import Path


def setup_kaggle():
    """Check and setup Kaggle API credentials."""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("=" * 60)
        print("Kaggle API credentials not found!")
        print("=" * 60)
        print("\nTo download data, you need to:")
        print("1. Create a Kaggle account at https://www.kaggle.com")
        print("2. Go to Account Settings > API > Create New Token")
        print("3. Place the downloaded kaggle.json in ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("\nAlternatively, set environment variables:")
        print("  export KAGGLE_USERNAME=your_username")
        print("  export KAGGLE_KEY=your_api_key")
        print("=" * 60)
        return False
    
    # Ensure proper permissions
    os.chmod(kaggle_json, 0o600)
    return True


def download_competition_data(
    competition: str,
    output_dir: str,
    files: list = None,
    unzip: bool = True,
):
    """
    Download competition data from Kaggle.
    
    Args:
        competition: Kaggle competition name/slug
        output_dir: Directory to save data
        files: Specific files to download (None = all)
        unzip: Whether to unzip downloaded files
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)
    
    # Initialize API
    api = KaggleApi()
    api.authenticate()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading data from: {competition}")
    print(f"Saving to: {output_path}")
    
    if files:
        # Download specific files
        for file in files:
            print(f"\nDownloading: {file}")
            api.competition_download_file(
                competition=competition,
                file_name=file,
                path=str(output_path),
                quiet=False
            )
    else:
        # Download all files
        print("\nDownloading all competition files...")
        api.competition_download_files(
            competition=competition,
            path=str(output_path),
            quiet=False
        )
    
    # Unzip if requested
    if unzip:
        import zipfile
        
        for zip_file in output_path.glob("*.zip"):
            print(f"\nExtracting: {zip_file.name}")
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(output_path)
            
            # Optionally remove zip file after extraction
            # zip_file.unlink()
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    
    # List downloaded files
    print("\nDownloaded files:")
    for f in output_path.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
        print(f"  {f.name} ({size_mb:.1f} MB)" if f.is_file() else f"  {f.name}/")


def main():
    parser = argparse.ArgumentParser(
        description="Download RSNA Intracranial Aneurysm Detection data from Kaggle"
    )
    parser.add_argument(
        "--competition",
        type=str,
        default="rsna-2024-lumbar-spine-degenerative-classification",
        help="Kaggle competition name/slug"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=None,
        help="Specific files to download (default: all)"
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Don't unzip downloaded files"
    )
    
    args = parser.parse_args()
    
    # Check Kaggle credentials
    if not setup_kaggle():
        # Check environment variables as fallback
        if not (os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY')):
            sys.exit(1)
    
    # Download data
    download_competition_data(
        competition=args.competition,
        output_dir=args.output_dir,
        files=args.files,
        unzip=not args.no_unzip,
    )


if __name__ == "__main__":
    main()

