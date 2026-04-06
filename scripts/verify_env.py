#!/usr/bin/env python3
"""
MedSymbol Environment Verification Script
==========================================
Run this after installing dependencies to verify everything works.

Usage:
    python scripts/verify_env.py
"""

import sys
from rich.console import Console
from rich.table import Table

console = Console()


def check_import(name: str, import_name: str = None) -> tuple[bool, str]:
    """Try importing a module and return success status + version."""
    import_name = import_name or name
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, str(e)


def check_torch_backends() -> dict:
    """Check PyTorch backend availability."""
    results = {}
    try:
        import torch

        results["CUDA available"] = torch.cuda.is_available()
        results["MPS available"] = torch.backends.mps.is_available()
        results["MPS built"] = torch.backends.mps.is_built()

        # Test MPS tensor creation
        if torch.backends.mps.is_available():
            try:
                x = torch.randn(2, 3, device="mps")
                y = x @ x.T
                results["MPS tensor ops"] = True
            except Exception as e:
                results["MPS tensor ops"] = f"FAILED: {e}"
        
        results["Default device"] = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    except ImportError:
        results["PyTorch"] = "NOT INSTALLED"
    return results


def check_transformers_models() -> dict:
    """Check that key HuggingFace models are downloadable."""
    results = {}
    try:
        from transformers import AutoConfig
        
        models = {
            "ViT-B/16": "google/vit-base-patch16-224",
            "BioBERT": "dmis-lab/biobert-v1.1",
        }
        for name, model_id in models.items():
            try:
                config = AutoConfig.from_pretrained(model_id)
                results[name] = f"✓ ({config.model_type})"
            except Exception as e:
                results[name] = f"✗ {e}"
    except ImportError:
        results["transformers"] = "NOT INSTALLED"
    return results


def check_z3() -> dict:
    """Check Z3 SMT solver functionality."""
    results = {}
    try:
        from z3 import Solver, Int, sat
        
        # Simple satisfiability test
        s = Solver()
        x = Int("x")
        s.add(x > 0, x < 10)
        result = s.check()
        results["Z3 solver"] = "✓ Working" if result == sat else f"✗ {result}"
        results["Z3 version"] = __import__("z3").get_version_string()
    except ImportError:
        results["Z3"] = "NOT INSTALLED"
    except Exception as e:
        results["Z3"] = f"ERROR: {e}"
    return results


def main():
    console.print("\n[bold blue]═══ MedSymbol Environment Verification ═══[/bold blue]\n")

    # ── Core Dependencies ──
    table = Table(title="Core Dependencies", show_header=True, header_style="bold cyan")
    table.add_column("Package", style="white", width=20)
    table.add_column("Status", width=10)
    table.add_column("Version / Info", style="dim")

    deps = [
        ("PyTorch", "torch"),
        ("TorchVision", "torchvision"),
        ("Transformers", "transformers"),
        ("TabNet", "pytorch_tabnet"),
        ("PyDICOM", "pydicom"),
        ("Pillow", "PIL"),
        ("OWLReady2", "owlready2"),
        ("Z3 Solver", "z3"),
        ("NumPy", "numpy"),
        ("Pandas", "pandas"),
        ("Scikit-learn", "sklearn"),
        ("Matplotlib", "matplotlib"),
        ("Seaborn", "seaborn"),
        ("W&B", "wandb"),
        ("Rich", "rich"),
        ("PyYAML", "yaml"),
        ("tqdm", "tqdm"),
    ]

    all_ok = True
    for name, import_name in deps:
        ok, version = check_import(name, import_name)
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        if not ok:
            all_ok = False
        table.add_row(name, status, version)

    console.print(table)

    # ── PyTorch Backends ──
    console.print("\n[bold cyan]PyTorch Backends[/bold cyan]")
    backends = check_torch_backends()
    for key, value in backends.items():
        icon = "[green]✓[/green]" if value is True else "[yellow]○[/yellow]" if value is False else "[blue]→[/blue]"
        console.print(f"  {icon} {key}: {value}")

    # ── Z3 SMT Solver ──
    console.print("\n[bold cyan]Z3 SMT Solver[/bold cyan]")
    z3_results = check_z3()
    for key, value in z3_results.items():
        console.print(f"  [blue]→[/blue] {key}: {value}")

    # ── HuggingFace Models ──
    console.print("\n[bold cyan]HuggingFace Model Access[/bold cyan]")
    console.print("  [dim](Checking config download — not downloading full weights)[/dim]")
    model_results = check_transformers_models()
    for key, value in model_results.items():
        console.print(f"  [blue]→[/blue] {key}: {value}")

    # ── Summary ──
    console.print()
    if all_ok:
        console.print("[bold green]✅ All core dependencies installed successfully![/bold green]")
        console.print("[dim]You're ready to start building MedSymbol.[/dim]\n")
    else:
        console.print("[bold red]❌ Some dependencies are missing. Install them before proceeding.[/bold red]\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
