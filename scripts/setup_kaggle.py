#!/usr/bin/env python3
"""
Kaggle API Setup and NIH CXR14 Dataset Download
================================================
Sets up Kaggle credentials and downloads NIH ChestX-ray14 dataset.

Setup Instructions:
1. Create Kaggle account: https://www.kaggle.com/settings/account
2. Generate API token: Click "Create New API Token" 
3. Run this script: python scripts/setup_kaggle.py --set-credentials

Download Dataset:
python scripts/setup_kaggle.py --download-nih-cxr14 --extract --target-dir ./data/raw
"""

import os
import sys
import json
import shutil
import zipfile
import tarfile
import argparse
from pathlib import Path
from typing import Optional

def setup_kaggle_credentials() -> bool:
    """
    Set up Kaggle API credentials interactively.
    Returns: True if successful, False otherwise
    """
    print("\n" + "="*60)
    print("Kaggle API Setup")
    print("="*60)
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        print("[✓] Kaggle credentials already exist at ~/.kaggle/kaggle.json")
        overwrite = input("Overwrite existing credentials? (y/n): ").lower()
        if overwrite != 'y':
            return True
    
    print("\n[*] Get your Kaggle API token:")
    print("    1. Go to https://www.kaggle.com/settings/account")
    print("    2. Click 'Create New API Token'")
    print("    3. This downloads kaggle.json to your Downloads folder")
    print("    4. Copy the contents of that file when prompted")
    
    api_key = input("\n[?] Enter your Kaggle API key: ").strip()
    username = input("[?] Enter your Kaggle username: ").strip()
    
    if not api_key or not username:
        print("[✗] Missing credentials")
        return False
    
    # Create kaggle.json
    creds = {
        "username": username,
        "key": api_key
    }
    
    with open(kaggle_json, 'w') as f:
        json.dump(creds, f)
    
    # Set proper permissions
    os.chmod(kaggle_json, 0o600)
    
    print(f"[✓] Credentials saved to {kaggle_json}")
    
    # Test authentication
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.auth()
        print("[✓] Kaggle authentication successful!")
        return True
    except Exception as e:
        print(f"[✗] Authentication failed: {e}")
        return False


def download_nih_cxr14(target_dir: str = "./data/raw", extract: bool = True) -> bool:
    """
    Download NIH ChestX-ray14 dataset from Kaggle.
    
    Dataset: ~40GB uncompressed
    - 112,120 frontal-view X-ray images
    - CSV with labels for 14 diseases
    - BBox annotations
    
    Args:
        target_dir: Directory to save dataset
        extract: Whether to extract archives
    
    Returns: True if successful, False otherwise
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[✗] kaggle-api not installed. Run: pip install kaggle")
        return False
    
    print("\n" + "="*60)
    print("Downloading NIH ChestX-ray14 Dataset")
    print("="*60)
    print("[*] This dataset is ~40GB. Ensure you have sufficient storage.")
    print("[*] Download may take 30+ minutes depending on connection speed.")
    
    target_path = Path(target_dir) / "nih_cxr14"
    target_path.mkdir(parents=True, exist_ok=True)
    
    try:
        api = KaggleApi()
        api.auth()
        
        print(f"\n[*] Downloading to: {target_path}")
        
        # Download dataset files
        api.dataset_download_files(
            'nih-chest-xrays/data',
            path=target_path,
            unzip=extract
        )
        
        print(f"[✓] Download complete!")
        
        if extract:
            # List downloaded files
            files = list(target_path.glob('*'))
            print(f"\n[✓] {len(files)} files/directories created")
            for f in sorted(files)[:10]:
                size = f.stat().st_size / (1024**3) if f.is_file() else 0
                print(f"    - {f.name}" + (f" ({size:.2f} GB)" if size > 0 else ""))
        
        return True
        
    except Exception as e:
        print(f"[✗] Download failed: {e}")
        print(f"\n[!] Common issues:")
        print(f"    - Kaggle credentials not set up properly")
        print(f"    - Not enough disk space")
        print(f"    - Dataset access not granted (check Kaggle account)")
        return False


def verify_dataset(dataset_dir: str) -> dict:
    """
    Verify downloaded dataset integrity.
    
    Returns: Dictionary with verification results
    """
    dataset_path = Path(dataset_dir)
    results = {
        "exists": dataset_path.exists(),
        "images_dir": False,
        "csv_files": [],
        "image_count": 0,
        "size_gb": 0.0
    }
    
    if not results["exists"]:
        return results
    
    # Check for images
    images_dir = dataset_path / "images"
    if images_dir.exists():
        results["images_dir"] = True
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        results["image_count"] = len(image_files)
    
    # Check for CSV files
    csv_files = list(dataset_path.glob("*.csv"))
    results["csv_files"] = [f.name for f in csv_files]
    
    # Calculate size
    total_size = 0
    for f in dataset_path.rglob("*"):
        if f.is_file():
            total_size += f.stat().st_size
    results["size_gb"] = total_size / (1024**3)
    
    return results


def print_dataset_info(dataset_dir: str):
    """Print information about downloaded dataset."""
    results = verify_dataset(dataset_dir)
    
    print("\n" + "="*60)
    print("Dataset Verification")
    print("="*60)
    
    if not results["exists"]:
        print(f"[✗] Dataset directory not found: {dataset_dir}")
        return
    
    print(f"[✓] Dataset location: {dataset_dir}")
    print(f"[*] Total size: {results['size_gb']:.2f} GB")
    print(f"[*] Images directory found: {results['images_dir']}")
    print(f"[*] Number of images: {results['image_count']:,}")
    print(f"[*] CSV files: {', '.join(results['csv_files']) if results['csv_files'] else 'None found'}")
    
    if results['image_count'] > 110000:
        print("\n[✓] Dataset appears complete (112,120 images expected)")
    elif results['image_count'] > 50000:
        print(f"\n[!] Dataset appears partial ({results['image_count']}/112,120 images)")
    else:
        print(f"\n[!] Dataset may not be extracted properly")


def main():
    parser = argparse.ArgumentParser(
        description="Setup Kaggle API and download NIH CXR14 dataset"
    )
    parser.add_argument(
        '--set-credentials',
        action='store_true',
        help='Interactively set Kaggle credentials'
    )
    parser.add_argument(
        '--download-nih-cxr14',
        action='store_true',
        help='Download NIH ChestX-ray14 dataset'
    )
    parser.add_argument(
        '--target-dir',
        type=str,
        default='./data/raw',
        help='Target directory for dataset (default: ./data/raw)'
    )
    parser.add_argument(
        '--extract',
        action='store_true',
        help='Extract downloaded archives'
    )
    parser.add_argument(
        '--verify',
        type=str,
        help='Verify dataset at specified directory'
    )
    
    args = parser.parse_args()
    
    if not any([args.set_credentials, args.download_nih_cxr14, args.verify]):
        parser.print_help()
        return
    
    if args.set_credentials:
        setup_kaggle_credentials()
    
    if args.download_nih_cxr14:
        success = download_nih_cxr14(
            target_dir=args.target_dir,
            extract=args.extract
        )
        if success and args.extract:
            print_dataset_info(f"{args.target_dir}/nih_cxr14")
    
    if args.verify:
        print_dataset_info(args.verify)


if __name__ == "__main__":
    main()
