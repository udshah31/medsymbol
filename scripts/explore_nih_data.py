#!/usr/bin/env python3
"""
NIH ChestX-ray14 Data Exploration
===================================
Explores the dataset to understand label distributions, patient demographics,
and mapping to SNOMED-CT/ICD-11 ontologies for MedSymbol.

Usage:
    python scripts/explore_nih_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

DATA_PATH = Path("data/raw/nih_cxr14/Data_Entry_2017.csv")


def main():
    console.print("\n[bold blue]═══ NIH ChestX-ray14 Data Exploration ═══[/bold blue]\n")

    # Load data
    df = pd.read_csv(DATA_PATH)
    console.print(f"[green]✓[/green] Loaded {len(df):,} records\n")

    # ── Basic Stats ──
    console.print("[bold cyan]1. Dataset Overview[/bold cyan]")
    console.print(f"   Total images:    {len(df):,}")
    console.print(f"   Unique patients: {df['Patient ID'].nunique():,}")
    console.print(f"   Age range:       {df['Patient Age'].min()} - {df['Patient Age'].max()}")
    console.print(f"   Gender split:    M={len(df[df['Patient Gender']=='M']):,} / F={len(df[df['Patient Gender']=='F']):,}")
    console.print(f"   View positions:  {dict(df['View Position'].value_counts())}\n")

    # ── Disease Label Distribution ──
    console.print("[bold cyan]2. Disease Label Distribution[/bold cyan]")

    # Split multi-labels
    all_labels = []
    for labels in df["Finding Labels"]:
        all_labels.extend(labels.split("|"))

    label_counts = pd.Series(all_labels).value_counts()

    # SNOMED-CT mapping for MedSymbol
    snomed_mapping = {
        "Atelectasis": "46621007",
        "Cardiomegaly": "8186001",
        "Consolidation": "95436001",
        "Edema": "267038008",
        "Effusion": "60046008",
        "Emphysema": "87433001",
        "Fibrosis": "51615001",
        "Hernia": "414403008",
        "Infiltration": "47693006",
        "Mass": "4147007",
        "No Finding": "—",
        "Nodule": "3529005",
        "Pleural_Thickening": "78381001",
        "Pneumonia": "233604007",
        "Pneumothorax": "36118008",
    }

    # ICD-11 mapping
    icd11_mapping = {
        "Atelectasis": "CB01",
        "Cardiomegaly": "BC40",
        "Consolidation": "CA40",
        "Edema": "KB00",
        "Effusion": "KB20",
        "Emphysema": "CA22",
        "Fibrosis": "CB03",
        "Hernia": "DA70",
        "Infiltration": "CA40",
        "Mass": "2C25",
        "No Finding": "—",
        "Nodule": "CB2Y",
        "Pleural_Thickening": "CB24",
        "Pneumonia": "CA40",
        "Pneumothorax": "CB20",
    }

    table = Table(title="Disease Labels", show_header=True, header_style="bold cyan")
    table.add_column("Finding", width=20)
    table.add_column("Count", justify="right", width=10)
    table.add_column("% of Total", justify="right", width=10)
    table.add_column("SNOMED-CT", width=12)
    table.add_column("ICD-11", width=8)

    for label, count in label_counts.items():
        pct = f"{count / len(df) * 100:.1f}%"
        snomed = snomed_mapping.get(label, "?")
        icd = icd11_mapping.get(label, "?")
        table.add_row(label, f"{count:,}", pct, snomed, icd)

    console.print(table)

    # ── Multi-label Statistics ──
    console.print("\n[bold cyan]3. Multi-Label Statistics[/bold cyan]")

    label_counts_per_image = df["Finding Labels"].apply(lambda x: len(x.split("|")))
    console.print(f"   Single-label images:   {(label_counts_per_image == 1).sum():,} ({(label_counts_per_image == 1).mean()*100:.1f}%)")
    console.print(f"   Multi-label images:    {(label_counts_per_image > 1).sum():,} ({(label_counts_per_image > 1).mean()*100:.1f}%)")
    console.print(f"   Max labels per image:  {label_counts_per_image.max()}")
    console.print(f"   Mean labels per image: {label_counts_per_image.mean():.2f}")

    # ── Age Distribution ──
    console.print("\n[bold cyan]4. Age Distribution[/bold cyan]")

    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_labels = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90+"]
    df["age_group"] = pd.cut(df["Patient Age"], bins=age_bins, labels=age_labels, right=False)

    age_table = Table(title="Age Distribution", show_header=True, header_style="bold cyan")
    age_table.add_column("Age Group", width=10)
    age_table.add_column("Count", justify="right", width=10)
    age_table.add_column("Percentage", justify="right", width=10)

    for group in age_labels:
        count = len(df[df["age_group"] == group])
        if count > 0:
            age_table.add_row(group, f"{count:,}", f"{count/len(df)*100:.1f}%")

    console.print(age_table)

    # ── Disease by Gender ──
    console.print("\n[bold cyan]5. Key Pathology × Gender[/bold cyan]")

    key_diseases = ["Pneumonia", "Cardiomegaly", "Effusion", "Infiltration", "Mass"]
    gender_table = Table(title="Pathology by Gender", show_header=True, header_style="bold cyan")
    gender_table.add_column("Disease", width=15)
    gender_table.add_column("Male", justify="right", width=8)
    gender_table.add_column("Female", justify="right", width=8)
    gender_table.add_column("M:F Ratio", justify="right", width=10)

    for disease in key_diseases:
        mask = df["Finding Labels"].str.contains(disease)
        m_count = len(df[mask & (df["Patient Gender"] == "M")])
        f_count = len(df[mask & (df["Patient Gender"] == "F")])
        ratio = f"{m_count/f_count:.2f}" if f_count > 0 else "N/A"
        gender_table.add_row(disease, f"{m_count:,}", f"{f_count:,}", ratio)

    console.print(gender_table)

    # ── MedSymbol Relevance Summary ──
    console.print("\n[bold cyan]6. MedSymbol Relevance Assessment[/bold cyan]")

    # Diseases relevant to the thesis respiratory/thoracic focus
    respiratory = ["Pneumonia", "Infiltration", "Consolidation", "Emphysema", "Fibrosis",
                   "Atelectasis", "Pneumothorax", "Effusion", "Pleural_Thickening", "Edema"]
    resp_mask = df["Finding Labels"].apply(
        lambda x: any(d in x for d in respiratory)
    )
    console.print(f"   Respiratory/thoracic findings:  {resp_mask.sum():,} ({resp_mask.mean()*100:.1f}% of dataset)")
    console.print(f"   'No Finding' images:            {(df['Finding Labels'] == 'No Finding').sum():,}")

    no_finding = (df['Finding Labels'] == 'No Finding').sum()
    with_finding = len(df) - no_finding
    console.print(f"\n   [bold]Usable for MedSymbol training: {with_finding:,} images with pathology findings[/bold]")

    # ── Save processed summary ──
    summary = {
        "total_images": len(df),
        "unique_patients": int(df["Patient ID"].nunique()),
        "label_distribution": label_counts.to_dict(),
        "snomed_mapping": snomed_mapping,
        "icd11_mapping": icd11_mapping,
    }

    import json
    output_path = Path("data/processed/nih_cxr14_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[green]✓[/green] Summary saved to {output_path}")
    console.print("[bold green]\n✅ Exploration complete![/bold green]\n")


if __name__ == "__main__":
    main()
