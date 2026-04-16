"""
Advanced Evaluation & Cross-Validation
======================================

Cross-validation, external validation, and statistical testing.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_recall_curve,
    roc_curve, confusion_matrix
)
import json


class CrossValidator:
    """Sophisticated cross-validation with stratification."""
    
    @staticmethod
    def stratified_kfold_cv(X, y, model, n_splits: int = 5) -> Dict[str, List[float]]:
        """
        K-fold stratified cross-validation.
        Ensures class distribution preserved in each fold.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_results = {
            'auc': [],
            'f1': [],
            'accuracy': [],
            'precision': [],
            'recall': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train/test logic here
            y_pred = np.random.rand(len(y_test))  # Placeholder
            
            auc = roc_auc_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred.round())
            accuracy = accuracy_score(y_test, y_pred.round())
            
            fold_results['auc'].append(auc)
            fold_results['f1'].append(f1)
            fold_results['accuracy'].append(accuracy)
        
        return fold_results
    
    @staticmethod
    def temporal_cv(X, y, timestamps, model, n_splits: int = 5) -> Dict[str, List[float]]:
        """
        Time-series aware cross-validation.
        Train on past, test on future to avoid data leakage.
        """
        fold_results = {
            'auc': [],
            'f1': [],
        }
        
        sorted_idx = np.argsort(timestamps)
        fold_size = len(X) // n_splits
        
        for fold in range(n_splits):
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size
            
            train_idx = sorted_idx[:test_start]
            test_idx = sorted_idx[test_start:test_end]
            
            # Use only past data for training
            y_pred = np.random.rand(len(test_idx))  # Placeholder
            
            auc = roc_auc_score(y[test_idx], y_pred)
            fold_results['auc'].append(auc)
        
        return fold_results
    
    @staticmethod
    def nested_cv(X, y, model_fn, outer_splits: int = 5, 
                  inner_splits: int = 3) -> Dict[str, any]:
        """
        Nested cross-validation for hyperparameter tuning.
        Outer loop: evaluation, Inner loop: hyperparameter selection
        """
        outer_results = []
        
        skf_outer = StratifiedKFold(n_splits=outer_splits, shuffle=True)
        skf_inner = StratifiedKFold(n_splits=inner_splits, shuffle=True)
        
        for train_idx, test_idx in skf_outer.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner CV for hyperparameter tuning
            best_score = 0
            best_model = None
            
            for inner_train_idx, inner_val_idx in skf_inner.split(X_train, y_train):
                X_inner_train = X_train[inner_train_idx]
                X_inner_val = X_train[inner_val_idx]
                
                model = model_fn()
                # Train model...
                
                score = roc_auc_score(y_train[inner_val_idx], np.random.rand(len(inner_val_idx)))
                if score > best_score:
                    best_score = score
                    best_model = model
            
            # Evaluate on outer fold
            y_pred = np.random.rand(len(y_test))
            outer_results.append({
                'auc': roc_auc_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred.round()),
            })
        
        return {
            'mean_auc': np.mean([r['auc'] for r in outer_results]),
            'std_auc': np.std([r['auc'] for r in outer_results]),
            'fold_results': outer_results
        }


class ExternalValidation:
    """Validation on external datasets."""
    
    @staticmethod
    def evaluate_on_external_dataset(model, external_data: Dict, 
                                    external_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on external dataset from different hospital/source.
        Critical for clinical deployment.
        """
        model.eval()
        
        predictions = []
        with torch.no_grad():
            for sample in external_data:
                pred = model(sample)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        
        metrics = {
            'external_auc': roc_auc_score(external_labels, predictions),
            'external_f1': f1_score(external_labels, predictions.round()),
            'external_accuracy': accuracy_score(external_labels, predictions.round()),
        }
        
        return metrics
    
    @staticmethod
    def domain_adaptation_analysis(internal_preds: np.ndarray, 
                                   external_preds: np.ndarray) -> Dict[str, float]:
        """
        Analyze distribution shift between internal and external datasets.
        Quantifies domain adaptation challenge.
        """
        return {
            'internal_mean_confidence': internal_preds.mean(),
            'external_mean_confidence': external_preds.mean(),
            'confidence_shift': abs(internal_preds.mean() - external_preds.mean()),
            'internal_std': internal_preds.std(),
            'external_std': external_preds.std(),
            'shift_magnitude': np.sqrt(
                (internal_preds.mean() - external_preds.mean()) ** 2 +
                (internal_preds.std() - external_preds.std()) ** 2
            )
        }


class PerformanceBenchmark:
    """Comprehensive performance benchmarking."""
    
    @staticmethod
    def benchmark_by_subgroup(predictions: np.ndarray, labels: np.ndarray,
                             subgroups: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Evaluate model fairness across demographic groups.
        Essential for clinical deployment.
        """
        results = {}
        
        for group_name, group_mask in subgroups.items():
            group_preds = predictions[group_mask]
            group_labels = labels[group_mask]
            
            results[group_name] = {
                'auc': roc_auc_score(group_labels, group_preds) if len(set(group_labels)) > 1 else None,
                'f1': f1_score(group_labels, group_preds.round()) if len(set(group_labels)) > 1 else None,
                'count': len(group_preds),
                'positive_rate': group_labels.mean(),
            }
        
        return results
    
    @staticmethod
    def save_benchmark_report(results: Dict, filepath: str):
        """Save comprehensive benchmark report."""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    # Example cross-validation
    X = np.random.rand(100, 512)
    y = np.random.randint(0, 2, 100)
    
    validator = CrossValidator()
    cv_results = validator.stratified_kfold_cv(X, y, None, n_splits=5)
    
    print("Cross-validation results:")
    for metric, scores in cv_results.items():
        print(f"  {metric}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
