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
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn, TransferSpeedColumn

console = Console()

DATA_DIR = Path("data/raw")

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


def download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        console.print(f"  [yellow]⊘[/yellow] Already exists: {dest.name}")
        return

    try:
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))

            with Progress(
                SpinnerColumn(),
                TextColumn(f"[cyan]{desc or dest.name}[/cyan]"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
            ) as progress:
                task = progress.add_task("download", total=total)
                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

        console.print(f"  [green]✓[/green] Downloaded: {dest.name}")
    except Exception as e:
        console.print(f"  [red]✗[/red] Failed to download {dest.name}: {e}")
        if dest.exists():
            dest.unlink()


def download_nih_cxr14():
    """Download NIH ChestX-ray14 dataset."""
    console.print("\n[bold blue]═══ NIH ChestX-ray14 Download ═══[/bold blue]")
    console.print("[dim]112,120 frontal-view chest X-rays, 14 disease labels[/dim]")
    console.print("[dim]Total size: ~42GB across 12 tar.gz archives[/dim]\n")

    nih_dir = DATA_DIR / "nih_cxr14"
    nih_dir.mkdir(parents=True, exist_ok=True)

    # Download labels first (small)
    console.print("[cyan]Downloading labels...[/cyan]")
    download_file(NIH_LABEL_URL, nih_dir / "Data_Entry_2017_v2020.csv", "Labels CSV")
    download_file(NIH_BBOX_URL, nih_dir / "BBox_List_2017.csv", "BBox CSV")

    # Download image archives
    console.print(f"\n[cyan]Downloading image archives (12 files, ~42GB total)...[/cyan]")
    console.print("[yellow]⚠ This will take a long time. You can interrupt and resume later.[/yellow]\n")

    for filename, url in NIH_IMAGE_URLS.items():
        download_file(url, nih_dir / filename, filename)

    # Check if all archives exist
    all_downloaded = all((nih_dir / f).exists() for f in NIH_IMAGE_URLS)
    if all_downloaded:
        console.print(f"\n[green]✓[/green] All archives downloaded.")
        console.print("[dim]To extract, run:[/dim]")
        console.print(f"[dim]  cd {nih_dir} && for f in images_*.tar.gz; do tar xzf $f; done[/dim]")
    else:
        console.print(f"\n[yellow]⚠[/yellow] Some archives still need to be downloaded. Re-run this script.")


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
    """Show available dataset status."""
    console.print("\n[bold blue]═══ Dataset Status ═══[/bold blue]\n")

    datasets = {
        "NIH CXR14 Labels": DATA_DIR / "nih_cxr14" / "Data_Entry_2017_v2020.csv",
        "NIH CXR14 Images": DATA_DIR / "nih_cxr14" / "images_001.tar.gz",
        "MIMIC-CXR": DATA_DIR / "mimic_cxr",
        "CheXpert": DATA_DIR / "chexpert",
    }

    for name, path in datasets.items():
        if path.exists():
            console.print(f"  [green]✓[/green] {name}")
        else:
            console.print(f"  [red]✗[/red] {name} — not found")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for MedSymbol")
    parser.add_argument(
        "--dataset",
        choices=["nih_cxr14", "nih_labels", "status"],
        default="status",
        help="Which dataset to download",
    )
    args = parser.parse_args()

    if args.dataset == "nih_cxr14":
        download_nih_cxr14()
    elif args.dataset == "nih_labels":
        download_nih_labels_only()
    elif args.dataset == "status":
        show_dataset_info()


if __name__ == "__main__":
    main()
