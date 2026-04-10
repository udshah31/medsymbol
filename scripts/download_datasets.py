#!/usr/bin/env python3
"""
Dataset Download Script for MedSymbol
======================================
Downloads freely available datasets for initial prototyping.

Datasets:
1. NIH ChestX-ray14 — 112,120 frontal-view X-rays (42GB)
2. OpenI/Indiana CXR — ~8,000 chest X-rays with reports (~1GB)

Usage:
    python scripts/download_datasets.py --dataset nih_cxr14
    python scripts/download_datasets.py --dataset openi
    python scripts/download_datasets.py --dataset all
"""

import argparse
import os
import sys
import urllib.request
import zipfile
import tarfile
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn
from rich.table import Table

console = Console()

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# ──────────────────────────────────────────────────
# NIH ChestX-ray14 URLs
# ──────────────────────────────────────────────────
NIH_BASE = "https://nihcc.box.com/shared/static"
NIH_IMAGE_URLS = {
    "images_001.tar.gz": f"{NIH_BASE}/vfk49d74nhbxq3nqjg0900w5nvkq7by8.tar.gz",
    "images_002.tar.gz": f"{NIH_BASE}/i28rlms7qhsedj727v1fq1ma2jiangga.tar.gz",
    "images_003.tar.gz": f"{NIH_BASE}/f1t00wrtdk94satdfb9olcolqx20teleq.tar.gz",
    "images_004.tar.gz": f"{NIH_BASE}/0aowwzs5lhjrceb3qp67ohp6rlf54ih9.tar.gz",
    "images_005.tar.gz": f"{NIH_BASE}/v5e3goj22zr6h8tzualxfsqlqkwbu46z.tar.gz",
    "images_006.tar.gz": f"{NIH_BASE}/asi7ikud9jwnkjq090e4bbyamf7g7ulc.tar.gz",
    "images_007.tar.gz": f"{NIH_BASE}/rber1jbps617yta3ras3bvpqnxqmmawes.tar.gz",
    "images_008.tar.gz": f"{NIH_BASE}/uu0f3nqadqkfj3opkhakbzm12dwubsxb.tar.gz",
    "images_009.tar.gz": f"{NIH_BASE}/8sf4th3lq0c4cby8fnnp47d1bk1rxawz.tar.gz",
    "images_010.tar.gz": f"{NIH_BASE}/3t9g8e97e88q1x2idh3r1a6dpqnt5x84.tar.gz",
    "images_011.tar.gz": f"{NIH_BASE}/jwymb6rsdqg0cmrpe4a7ejsa0rfx71xm.tar.gz",
    "images_012.tar.gz": f"{NIH_BASE}/3yiszde3wkrcj0405k4twamjdakyzo3t.tar.gz",
}
NIH_LABEL_URL = f"{NIH_BASE}/zuowkr553brg3xlz7pm1jbaw1b0wdduf.csv"  # Data_Entry_2017_v2020.csv
NIH_BBOX_URL = f"{NIH_BASE}/qu9jw9slchgrz5a4gslxq9xkgalerzjv.csv"   # BBox_List_2017.csv

# ──────────────────────────────────────────────────
# OpenI / Indiana CXR — using academic sources
# ──────────────────────────────────────────────────
OPENI_HF_DATASET = "alkzar90/NIH-Chest-X-ray-dataset"  # fallback


def download_file(url: str, dest: Path, desc: str = "", skip_existing: bool = True):
    """Download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if skip_existing:
            console.print(f"  [yellow]⊘[/yellow] Already exists: {dest.name}")
            return True
        else:
            dest.unlink()

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            total = int(response.headers.get("Content-Length", 0))

            with Progress(
                SpinnerColumn(),
                TextColumn(f"[cyan]{desc or dest.name}[/cyan]"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("download", total=total if total > 0 else None)
                with open(dest, "wb") as f:
                    downloaded = 0
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.update(task, advance=len(chunk))

        console.print(f"  [green]✓[/green] Downloaded: {dest.name}")
        return True
    except Exception as e:
        console.print(f"  [red]✗[/red] Failed to download {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def extract_tar_gz(filepath: Path, extract_dir: Path, desc: str = ""):
    """Extract tar.gz file with progress."""
    if not filepath.exists():
        console.print(f"  [red]✗[/red] File not found: {filepath}")
        return False
    
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"  [cyan]Extracting {filepath.name}...[/cyan]")
        with tarfile.open(filepath, 'r:gz') as tar:
            members = tar.getmembers()
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[cyan]{desc or 'Extracting'}[/cyan]"),
                BarColumn(),
                transient=True
            ) as progress:
                task = progress.add_task("extract", total=len(members))
                for member in tar.getmembers():
                    tar.extract(member, path=extract_dir)
                    progress.advance(task)
        
        console.print(f"  [green]✓[/green] Extracted {len(members)} files")
        return True
    except Exception as e:
        console.print(f"  [red]✗[/red] Failed to extract {filepath.name}: {e}")
        return False


def analyze_nih_metadata(csv_path: Path):
    """Analyze NIH dataset metadata and print statistics."""
    if not csv_path.exists():
        console.print(f"  [red]✗[/red] CSV not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        console.print(f"\n[bold]NIH ChestX-ray14 Dataset Statistics[/bold]")
        console.print(f"  Total Images: {len(df):,}")
        
        # Parse findings
        all_findings = []
        for findings_str in df.get('Finding Labels', []):
            if isinstance(findings_str, str) and findings_str != 'No Finding':
                findings = findings_str.split('|')
                all_findings.extend(findings)
        
        # Count disease frequencies
        disease_counts = Counter(all_findings)
        
        table = Table(title="Disease Label Frequencies")
        table.add_column("Disease", style="cyan")
        table.add_column("Count", style="magenta")
        table.add_column("Percentage", style="green")
        
        for disease, count in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(df)) * 100
            table.add_row(disease, str(count), f"{pct:.1f}%")
        
        console.print(table)
        
        # Demographics if available
        if 'Patient Age' in df.columns:
            ages = pd.to_numeric(df['Patient Age'], errors='coerce').dropna()
            console.print(f"\n[bold]Demographics[/bold]")
            console.print(f"  Age Range: {ages.min():.0f} - {ages.max():.0f} years")
            console.print(f"  Age Mean: {ages.mean():.1f} years")
            console.print(f"  Age Median: {ages.median():.1f} years")
        
        if 'Patient Sex' in df.columns:
            sex_counts = df['Patient Sex'].value_counts()
            for sex, count in sex_counts.items():
                pct = (count / len(df)) * 100
                console.print(f"  {sex}: {count:,} ({pct:.1f}%)")
        
        return df
    
    except Exception as e:
        console.print(f"  [red]✗[/red] Error analyzing metadata: {e}")
        return None


def download_nih_cxr14(extract: bool = False):
    """Download NIH ChestX-ray14 dataset."""
    console.print("\n[bold blue]═══ NIH ChestX-ray14 Download ═══[/bold blue]")
    console.print("[dim]112,120 frontal-view chest X-rays, 14 disease labels[/dim]")
    console.print("[dim]Total size: ~42GB across 12 tar.gz archives[/dim]\n")

    nih_dir = DATA_DIR / "nih_cxr14"
    nih_dir.mkdir(parents=True, exist_ok=True)

    # Download labels first (small)
    console.print("[cyan]Step 1: Downloading metadata...[/cyan]")
    label_path = nih_dir / "Data_Entry_2017_v2020.csv"
    bbox_path = nih_dir / "BBox_List_2017.csv"
    
    download_file(NIH_LABEL_URL, label_path, "Labels CSV")
    download_file(NIH_BBOX_URL, bbox_path, "BBox CSV")
    
    # Analyze and show statistics
    if label_path.exists():
        analyze_nih_metadata(label_path)

    # Download image archives
    console.print(f"\n[cyan]Step 2: Downloading image archives (12 files, ~42GB total)...[/cyan]")
    console.print("[yellow]⚠ This will take a long time. You can interrupt and resume later.[/yellow]\n")

    all_downloaded = True
    for filename, url in NIH_IMAGE_URLS.items():
        if not download_file(url, nih_dir / filename, filename):
            all_downloaded = False

    # Extract if requested
    if extract and all_downloaded:
        console.print(f"\n[cyan]Step 3: Extracting archives...[/cyan]")
        images_dir = nih_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        for filename in NIH_IMAGE_URLS:
            archive_path = nih_dir / filename
            if archive_path.exists():
                extract_tar_gz(archive_path, images_dir, f"Extracting {filename}")

    # Summary
    if all_downloaded:
        console.print(f"\n[green bold]✓[/green bold] Download complete!")
        image_count = len(list((nih_dir / "images").glob("*.png"))) if (nih_dir / "images").exists() else 0
        if image_count > 0:
            console.print(f"[green]✓[/green] {image_count:,} images extracted and ready for training")
        else:
            console.print("[dim]To extract later, run:[/dim]")
            console.print(f"[dim]  cd {nih_dir} && for f in images_*.tar.gz; do tar xzf $f; done[/dim]")
    else:
        console.print(f"\n[yellow]⚠[/yellow] Some downloads failed or incomplete. Re-run this script to retry.")


def download_nih_labels_only():
    """Download just the NIH CXR14 labels and metadata (for quick start)."""
    console.print("\n[bold blue]═══ NIH ChestX-ray14 Labels Only ═══[/bold blue]")
    console.print("[dim]Download just labels + metadata for data exploration (~10MB)[/dim]\n")

    nih_dir = DATA_DIR / "nih_cxr14"
    nih_dir.mkdir(parents=True, exist_ok=True)

    download_file(NIH_LABEL_URL, nih_dir / "Data_Entry_2017_v2020.csv", "Labels CSV")
    download_file(NIH_BBOX_URL, nih_dir / "BBox_List_2017.csv", "BBox CSV")

    console.print(f"\n[green]✓[/green] Labels downloaded to {nih_dir}/")
    console.print("[dim]You can explore data distributions before downloading images.[/dim]")


def show_dataset_info():
    """Show available dataset status and statistics."""
    console.print("\n[bold blue]═══ Dataset Status ═══[/bold blue]\n")

    nih_dir = DATA_DIR / "nih_cxr14"
    
    # Check individual components
    status_table = Table(title="Dataset Components")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="magenta")
    status_table.add_column("Details", style="green")
    
    # Labels
    label_path = nih_dir / "Data_Entry_2017_v2020.csv"
    if label_path.exists():
        status_table.add_row("Labels CSV", "[green]✓[/green]", f"{label_path.stat().st_size / 1e6:.1f} MB")
    else:
        status_table.add_row("Labels CSV", "[red]✗[/red]", "Not found")
    
    # BBox
    bbox_path = nih_dir / "BBox_List_2017.csv"
    if bbox_path.exists():
        status_table.add_row("BBox CSV", "[green]✓[/green]", f"{bbox_path.stat().st_size / 1e6:.1f} MB")
    else:
        status_table.add_row("BBox CSV", "[red]✗[/red]", "Not found")
    
    # Archives
    archives_found = sum(1 for f in NIH_IMAGE_URLS if (nih_dir / f).exists())
    if archives_found > 0:
        total_size = sum((nih_dir / f).stat().st_size for f in NIH_IMAGE_URLS if (nih_dir / f).exists())
        status_table.add_row("Image Archives", "[yellow]◐[/yellow]", f"{archives_found}/12 ({total_size / 1e9:.1f} GB)")
    else:
        status_table.add_row("Image Archives", "[red]✗[/red]", "0/12")
    
    # Extracted images
    images_dir = nih_dir / "images"
    if images_dir.exists():
        image_count = len(list(images_dir.glob("*.png")))
        if image_count > 0:
            status_table.add_row("Extracted Images", "[green]✓[/green]", f"{image_count:,} files")
        else:
            status_table.add_row("Extracted Images", "[yellow]◐[/yellow]", "Directory exists but empty")
    else:
        status_table.add_row("Extracted Images", "[red]✗[/red]", "Not extracted")
    
    console.print(status_table)
    
    # Show statistics if metadata available
    if label_path.exists():
        console.print()
        analyze_nih_metadata(label_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare datasets for MedSymbol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check dataset status
  python scripts/download_datasets.py --status

  # Download and show NIH metadata only
  python scripts/download_datasets.py --dataset nih_labels

  # Download full NIH dataset (42GB)
  python scripts/download_datasets.py --dataset nih_cxr14

  # Download and extract (very long)
  python scripts/download_datasets.py --dataset nih_cxr14 --extract

  # Extract previously downloaded files
  python scripts/download_datasets.py --extract-only
        """
    )
    parser.add_argument(
        "--dataset",
        choices=["nih_cxr14", "nih_labels"],
        default=None,
        help="Which dataset to download",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Automatically extract tar.gz files after download (requires ~150GB temp space)"
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract previously downloaded archives"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show dataset status and statistics"
    )
    
    args = parser.parse_args()

    if args.extract_only:
        # Extract only mode
        nih_dir = DATA_DIR / "nih_cxr14"
        images_dir = nih_dir / "images"
        console.print("\n[bold blue]═══ Extracting Archives ═══[/bold blue]\n")
        
        for filename in NIH_IMAGE_URLS:
            archive_path = nih_dir / filename
            if archive_path.exists():
                extract_tar_gz(archive_path, images_dir, f"Extracting {filename}")
        
        image_count = len(list(images_dir.glob("*.png")))
        console.print(f"\n[green]✓[/green] Extraction complete! {image_count:,} images ready.")
    
    elif args.dataset == "nih_cxr14":
        download_nih_cxr14(extract=args.extract)
    
    elif args.dataset == "nih_labels":
        download_nih_labels_only()
    
    else:
        # Default: show status
        show_dataset_info()


if __name__ == "__main__":
    main()
