#!/usr/bin/env python3
"""
Evaluation Pipeline for MedSymbol
==================================
Generates comprehensive evaluation metrics on test set:
- AUC, Sensitivity, Specificity, F1 per disease
- Confusion matrices
- Overall performance metrics
- Integration with Weights & Biases

Usage:
    python scripts/evaluate_model.py --model_path ./models/medsymbol.pt --dataset ./data/processed
"""

import argparse
import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import MedSymbolModel
from src.utils.data_loader import NIHCXR14Dataset
from torch.utils.data import DataLoader

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("[!] Weights & Biases not available. Proceeding without experiment tracking.")


class Evaluator:
    """Comprehensive model evaluation pipeline."""
    
    def __init__(self, model: MedSymbolModel, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Storage for metrics
        self.predictions = []  # List of (logits, labels) tuples
        self.patient_data_list = []
        
    def evaluate_dataset(self, dataloader: DataLoader, desc: str = "Evaluation"):
        """
        Run inference on dataset and collect predictions.
        
        Args:
            dataloader: PyTorch DataLoader
            desc: Description for progress bar
        
        Returns:
            predictions: List of prediction dicts
        """
        predictions = []
        
        with torch.no_grad():
            for batch_idx, (inputs, labels, patient_data) in enumerate(tqdm(dataloader, desc=desc)):
                # Move to device
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        inputs[k] = inputs[k].to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(inputs)  # [batch_size, 14]
                probs = torch.softmax(logits, dim=1)  # [batch_size, 14]
                
                # Store predictions
                for i in range(logits.shape[0]):
                    pred_dict = {
                        'logits': logits[i].cpu().numpy(),
                        'probs': probs[i].cpu().numpy(),
                        'labels': labels[i].cpu().numpy(),
                        'patient_data': patient_data if isinstance(patient_data, dict) else {}
                    }
                    predictions.append(pred_dict)
        
        self.predictions = predictions
        return predictions
    
    def compute_metrics(self) -> dict:
        """
        Compute comprehensive metrics from predictions.
        
        Returns:
            metrics_dict: Dictionary of metrics
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run evaluate_dataset first.")
        
        disease_labels = NIHCXR14Dataset.DISEASE_LABELS
        num_classes = len(disease_labels)
        
        # Aggregate predictions
        all_labels = np.array([p['labels'] for p in self.predictions])      # [N, 14]
        all_probs = np.array([p['probs'] for p in self.predictions])        # [N, 14]
        all_logits = np.array([p['logits'] for p in self.predictions])      # [N, 14]
        
        # Predicted classes (argmax)
        predicted_classes = np.argmax(all_logits, axis=1)                   # [N]
        
        # Compute per-disease metrics
        metrics = {
            'overall': {},
            'per_disease': {}
        }
        
        # Overall accuracy
        accuracy = np.mean(predicted_classes == np.argmax(all_labels, axis=1))
        metrics['overall']['accuracy'] = float(accuracy)
        print(f"\n[*] Overall Accuracy: {accuracy:.4f}")
        
        # Per-disease metrics
        for disease_idx, disease_name in enumerate(disease_labels):
            disease_labels_binary = all_labels[:, disease_idx]
            disease_probs = all_probs[:, disease_idx]
            
            # Only compute metrics if we have both positive and negative samples
            if len(np.unique(disease_labels_binary)) < 2:
                metrics['per_disease'][disease_name] = {
                    'auc': None,
                    'sensitivity': None,
                    'specificity': None,
                    'f1': None,
                    'precision': None,
                    'recall': None,
                    'support': int(np.sum(disease_labels_binary))
                }
                continue
            
            try:
                # AUC-ROC
                auc = roc_auc_score(disease_labels_binary, disease_probs)
                
                # Binary predictions (threshold = 0.5)
                disease_preds = (disease_probs >= 0.5).astype(int)
                
                # Sensitivity (TP / (TP + FN)) = Recall for positive class
                tn, fp, fn, tp = confusion_matrix(disease_labels_binary, disease_preds).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                # F1, Precision
                f1 = f1_score(disease_labels_binary, disease_preds, zero_division=0)
                precision = precision_score(disease_labels_binary, disease_preds, zero_division=0)
                recall = recall_score(disease_labels_binary, disease_preds, zero_division=0)
                
                metrics['per_disease'][disease_name] = {
                    'auc': float(auc),
                    'sensitivity': float(sensitivity),
                    'specificity': float(specificity),
                    'f1': float(f1),
                    'precision': float(precision),
                    'recall': float(recall),
                    'support': int(np.sum(disease_labels_binary))
                }
                
            except Exception as e:
                print(f"[!] Error computing metrics for {disease_name}: {e}")
                metrics['per_disease'][disease_name] = {
                    'auc': None,
                    'sensitivity': None,
                    'specificity': None,
                    'f1': None,
                    'precision': None,
                    'recall': None,
                    'support': int(np.sum(disease_labels_binary))
                }
        
        # Compute and display per-disease summary
        print("\n" + "="*80)
        print("Per-Disease Metrics:")
        print("="*80)
        
        summary_data = []
        for disease_name, disease_metrics in sorted(metrics['per_disease'].items()):
            summary_data.append({
                'disease': disease_name,
                'auc': disease_metrics['auc'],
                'sensitivity': disease_metrics['sensitivity'],
                'specificity': disease_metrics['specificity'],
                'f1': disease_metrics['f1'],
                'support': disease_metrics['support']
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Compute macro-averages (excluding None values)
        valid_aucs = [m['auc'] for m in metrics['per_disease'].values() if m['auc'] is not None]
        valid_sens = [m['sensitivity'] for m in metrics['per_disease'].values() if m['sensitivity'] is not None]
        valid_spec = [m['specificity'] for m in metrics['per_disease'].values() if m['specificity'] is not None]
        valid_f1 = [m['f1'] for m in metrics['per_disease'].values() if m['f1'] is not None]
        
        if valid_aucs:
            metrics['overall']['macro_auc'] = float(np.mean(valid_aucs))
            print(f"\n[*] Macro-average AUC: {metrics['overall']['macro_auc']:.4f}")
        
        if valid_sens:
            metrics['overall']['macro_sensitivity'] = float(np.mean(valid_sens))
            print(f"[*] Macro-average Sensitivity: {metrics['overall']['macro_sensitivity']:.4f}")
        
        if valid_spec:
            metrics['overall']['macro_specificity'] = float(np.mean(valid_spec))
            print(f"[*] Macro-average Specificity: {metrics['overall']['macro_specificity']:.4f}")
        
        if valid_f1:
            metrics['overall']['macro_f1'] = float(np.mean(valid_f1))
            print(f"[*] Macro-average F1: {metrics['overall']['macro_f1']:.4f}")
        
        return metrics
    
    def generate_confusion_matrix(self, output_dir: str = './experiments'):
        """
        Generate and save confusion matrix for overall predictions.
        """
        if not self.predictions:
            raise ValueError("No predictions available.")
        
        disease_labels = NIHCXR14Dataset.DISEASE_LABELS
        all_labels = np.array([p['labels'] for p in self.predictions])
        all_logits = np.array([p['logits'] for p in self.predictions])
        
        predicted_classes = np.argmax(all_logits, axis=1)
        true_classes = np.argmax(all_labels, axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes, labels=range(len(disease_labels)))
        
        # Plot and save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=disease_labels, yticklabels=disease_labels)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix - MedSymbol Disease Classification')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n[✓] Saved confusion matrix to {output_path}")
        plt.close()
        
        return cm
    
    def save_report(self, metrics: dict, output_dir: str = './experiments'):
        """Save evaluation report as JSON."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        report_path = os.path.join(output_dir, 'evaluation_report.json')
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"[✓] Saved evaluation report to {report_path}")
        
        # Also save as CSV for easier viewing
        csv_path = os.path.join(output_dir, 'metrics_summary.csv')
        
        rows = []
        for disease_name, disease_metrics in metrics['per_disease'].items():
            row = {'disease': disease_name}
            row.update(disease_metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"[✓] Saved metrics summary to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedSymbol Model")
    parser.add_argument('--model_path', type=str, default='./models/medsymbol.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--csv_file', type=str, default='data/raw/nih_cxr14/Data_Entry_2017_v2020.csv',
                        help='Path to NIH CXR14 metadata CSV')
    parser.add_argument('--img_dir', type=str, default='data/raw/nih_cxr14',
                        help='Path to NIH CXR14 images directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Evaluation batch size')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='medsymbol',
                        help='W&B project name')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        print(f"[*] Using CUDA device")
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
        print(f"[*] Using MPS device")
    else:
        device = 'cpu'
        print(f"[*] Using CPU device")
    
    # Initialize W&B if available and enabled
    if args.wandb and HAS_WANDB:
        try:
            wandb.init(project=args.wandb_project, 
                      name="medsymbol-evaluation",
                      config=vars(args))
            print("[*] Weights & Biases initialized for evaluation")
        except Exception as e:
            print(f"[!] Failed to initialize W&B: {e}")
    
    # Load model
    print(f"\n[*] Loading model from {args.model_path}")
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model = MedSymbolModel(
            num_diagnoses=14, 
            tabular_input_dim=10, 
            history_input_dim=5
        )
        model.load_state_dict(checkpoint)
        model.to(device)
        print("[✓] Model loaded successfully")
    else:
        print(f"[!] Model not found at {args.model_path}")
        print("[*] Using randomly initialized model for demonstration")
        model = MedSymbolModel(
            num_diagnoses=14, 
            tabular_input_dim=10, 
            history_input_dim=5
        )
        model.to(device)
    
    # Create test dataset
    print(f"\n[*] Loading test dataset...")
    test_dataset = NIHCXR14Dataset(
        csv_file=args.csv_file,
        img_dir=args.img_dir,
        split='test',
        transform=None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    print(f"[✓] Test set: {len(test_dataset)} samples")
    
    # Evaluate
    print(f"\n[*] Starting evaluation...")
    evaluator = Evaluator(model, device=device)
    predictions = evaluator.evaluate_dataset(test_loader, desc="Evaluating")
    
    # Compute metrics
    print(f"\n[*] Computing metrics...")
    metrics = evaluator.compute_metrics()
    
    # Generate confusion matrix
    print(f"\n[*] Generating confusion matrix...")
    cm = evaluator.generate_confusion_matrix(args.output_dir)
    
    # Save reports
    print(f"\n[*] Saving evaluation reports...")
    evaluator.save_report(metrics, args.output_dir)
    
    # Log to W&B if available
    if HAS_WANDB:
        try:
            # Log scalar metrics
            wandb_metrics = {}
            
            # Overall metrics
            for key, value in metrics['overall'].items():
                if value is not None:
                    wandb_metrics[f'eval/overall_{key}'] = value
            
            # Per-disease metrics
            for disease_name, disease_metrics in metrics['per_disease'].items():
                for metric_name, value in disease_metrics.items():
                    if value is not None and metric_name != 'support':
                        wandb_metrics[f'eval/{disease_name}/{metric_name}'] = value
            
            wandb.log(wandb_metrics)
            
            # Log confusion matrix image
            wandb.log({"eval/confusion_matrix": wandb.Image(
                os.path.join(args.output_dir, 'confusion_matrix.png')
            )})
            
            print("[✓] Metrics logged to Weights & Biases")
        except Exception as e:
            print(f"[!] Failed to log to W&B: {e}")
    
    print(f"\n[✓] Evaluation complete!")
    print(f"[*] Results saved to {args.output_dir}")
    
    if HAS_WANDB:
        wandb.finish()


if __name__ == '__main__':
    main()
