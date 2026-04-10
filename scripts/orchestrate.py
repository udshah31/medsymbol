#!/usr/bin/env python3
"""
MedSymbol End-to-End Pipeline Orchestrator
===========================================
Simplified interface for: download → split → train → evaluate

Usage (recommended order):
    # 1. Download dataset
    python scripts/orchestrate.py --stage download --dataset nih_labels
    
    # 2. (Optional) Download full dataset with extraction
    python scripts/orchestrate.py --stage download --dataset nih_cxr14 --extract
    
    # 3. Check dataset status
    python scripts/orchestrate.py --stage status
    
    # 4. Train model
    python scripts/orchestrate.py --stage train --epochs 20 --batch_size 16
    
    # 5. Evaluate model
    python scripts/orchestrate.py --stage evaluate --model_path ./models/medsymbol.pt
    
    # OR run everything end-to-end
    python scripts/orchestrate.py --all
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

console = Console()

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_DIR / "models"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"


class PipelineOrchestrator:
    """Orchestrates the MedSymbol training pipeline."""
    
    def __init__(self):
        self.project_dir = PROJECT_DIR
        self.models_dir = MODELS_DIR
        self.experiments_dir = EXPERIMENTS_DIR
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.experiments_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def log(self, message: str, level: str = "INFO"):
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {level}: {message}"
        
        with open(self.log_file, "a") as f:
            f.write(log_line + "\n")
        
        if level == "ERROR":
            console.print(f"[red]✗[/red] {message}")
        elif level == "WARN":
            console.print(f"[yellow]⚠[/yellow] {message}")
        elif level == "SUCCESS":
            console.print(f"[green]✓[/green] {message}")
        else:
            console.print(f"[cyan]ℹ[/cyan] {message}")
    
    def run_command(self, cmd: list, desc: str = "Running command") -> bool:
        """Run shell command and track result."""
        self.log(f"Running: {' '.join(cmd)}")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[cyan]{desc}[/cyan]"),
                BarColumn(),
            ) as progress:
                task = progress.add_task("working", total=None)
                
                result = subprocess.run(
                    cmd,
                    cwd=str(self.project_dir),
                    capture_output=True,
                    text=True,
                    timeout=3600
                )
                
                progress.complete_task(task)
            
            if result.returncode == 0:
                self.log(f"Command succeeded: {desc}", "SUCCESS")
                return True
            else:
                self.log(f"Command failed: {result.stderr[:500]}", "ERROR")
                return False
        
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out: {desc}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Command error: {e}", "ERROR")
            return False
    
    def stage_download(self, dataset: str = "nih_labels", extract: bool = False):
        """Download dataset stage."""
        console.print("\n[bold blue]═══ Stage 1: Download Dataset ═══[/bold blue]")
        
        cmd = [
            sys.executable, "scripts/download_datasets.py",
            "--dataset", dataset
        ]
        
        if extract:
            cmd.append("--extract")
        
        return self.run_command(cmd, f"Downloading {dataset} dataset")
    
    def stage_status(self):
        """Show dataset status stage."""
        console.print("\n[bold blue]═══ Status: Dataset Information ═══[/bold blue]")
        
        cmd = [
            sys.executable, "scripts/download_datasets.py",
            "--status"
        ]
        
        return self.run_command(cmd, "Checking dataset status")
    
    def stage_train(self, epochs: int = 10, batch_size: int = 16, 
                    learning_rate: float = 0.0001):
        """Train model stage."""
        console.print("\n[bold blue]═══ Stage 2: Train Model ═══[/bold blue]")
        
        model_path = self.models_dir / f"medsymbol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        
        cmd = [
            sys.executable, "scripts/train_model.py",
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
            "--lr", str(learning_rate),
            "--model_path", str(model_path),
        ]
        
        success = self.run_command(cmd, f"Training for {epochs} epochs")
        
        if success:
            self.log(f"Model saved to: {model_path}", "SUCCESS")
            return str(model_path)
        
        return None
    
    def stage_evaluate(self, model_path: str, batch_size: int = 32):
        """Evaluate model stage."""
        console.print("\n[bold blue]═══ Stage 3: Evaluate Model ═══[/bold blue]")
        
        if not Path(model_path).exists():
            self.log(f"Model not found: {model_path}", "WARN")
            model_path = "./models/medsymbol.pt"
            self.log(f"Using default model path: {model_path}", "INFO")
        
        cmd = [
            sys.executable, "scripts/evaluate_model.py",
            "--model_path", model_path,
            "--batch_size", str(batch_size),
            "--output_dir", str(self.experiments_dir),
        ]
        
        return self.run_command(cmd, "Evaluating model")
    
    def run_all(self, epochs: int = 10, batch_size: int = 16):
        """Run complete pipeline."""
        console.print("\n[bold blue bold]╔════════════════════════════════════╗[/bold blue bold]")
        console.print("[bold blue bold]║   MedSymbol End-to-End Pipeline   ║[/bold blue bold]")
        console.print("[bold blue bold]╚════════════════════════════════════╝[/bold blue bold]\n")
        
        self.log("Starting complete pipeline", "INFO")
        
        # Stage 1: Check status
        if not self.stage_status():
            console.print("[yellow]⚠ Proceeding despite status check issues[/yellow]")
        
        # Stage 2: Download (labels only for quick start)
        if not self.stage_download(dataset="nih_labels"):
            self.log("Download failed. Cannot continue.", "ERROR")
            return False
        
        # Stage 3: Train
        model_path = self.stage_train(epochs=epochs, batch_size=batch_size)
        if not model_path:
            self.log("Training failed. Cannot continue.", "ERROR")
            return False
        
        # Stage 4: Evaluate
        if not self.stage_evaluate(model_path=model_path, batch_size=batch_size):
            self.log("Evaluation failed.", "WARN")
            return False
        
        console.print("\n[bold green]✓ Pipeline complete![/bold green]")
        self.log("Pipeline completed successfully", "SUCCESS")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="MedSymbol Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick start (recommended):
  python scripts/orchestrate.py --all

Step-by-step:
  python scripts/orchestrate.py --stage download --dataset nih_labels
  python scripts/orchestrate.py --stage train --epochs 10
  python scripts/orchestrate.py --stage evaluate
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run complete pipeline (download → train → evaluate)"
    )
    parser.add_argument(
        "--stage",
        choices=["download", "status", "train", "evaluate"],
        help="Run specific pipeline stage"
    )
    parser.add_argument(
        "--dataset",
        choices=["nih_cxr14", "nih_labels"],
        default="nih_labels",
        help="Dataset to download (stage=download)"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract archives after download"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model for evaluation"
    )
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator()
    
    console.print(f"[dim]Logs: {orchestrator.log_file}[/dim]\n")
    
    if args.all:
        success = orchestrator.run_all(epochs=args.epochs, batch_size=args.batch_size)
        sys.exit(0 if success else 1)
    
    elif args.stage == "download":
        success = orchestrator.stage_download(
            dataset=args.dataset,
            extract=args.extract
        )
        sys.exit(0 if success else 1)
    
    elif args.stage == "status":
        success = orchestrator.stage_status()
        sys.exit(0 if success else 1)
    
    elif args.stage == "train":
        model_path = orchestrator.stage_train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        sys.exit(0 if model_path else 1)
    
    elif args.stage == "evaluate":
        model_path = args.model_path or "./models/medsymbol.pt"
        success = orchestrator.stage_evaluate(
            model_path=model_path,
            batch_size=args.batch_size
        )
        sys.exit(0 if success else 1)
    
    else:
        # Default: show status
        orchestrator.stage_status()


if __name__ == "__main__":
    main()
