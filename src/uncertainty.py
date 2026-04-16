"""
Uncertainty & Severity Quantification
=====================================

Add model uncertainty estimation and disease severity grading.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np


class UncertaintyEstimator(nn.Module):
    """Bayesian uncertainty estimation with ensemble methods."""
    
    def __init__(self, num_diagnoses: int = 14, dropout_rate: float = 0.5):
        super().__init__()
        self.num_diagnoses = num_diagnoses
        self.dropout_rate = dropout_rate
        
        # MC Dropout classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(128, num_diagnoses)
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, features: torch.Tensor, n_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        MC Dropout inference for uncertainty.
        
        Args:
            features: (batch, 512)
            n_samples: Number of forward passes
        
        Returns:
            {
                'predictions': (batch, num_diagnoses),
                'uncertainty': (batch, num_diagnoses),
                'epistemic': (batch, num_diagnoses),
                'aleatoric': (batch, num_diagnoses)
            }
        """
        self.train()  # Enable dropout
        
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.classifier(features) / self.temperature
                probs = torch.softmax(logits, dim=1)
                samples.append(probs)
        
        self.eval()
        
        samples = torch.stack(samples, dim=0)  # (n_samples, batch, diagnoses)
        
        # Mean prediction
        mean_pred = samples.mean(dim=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic = samples.var(dim=0).mean(dim=-1, keepdim=True)
        
        # Aleatoric uncertainty (data uncertainty)
        aleatoric = (mean_pred * (1 - mean_pred)).mean(dim=-1, keepdim=True)
        
        # Total uncertainty
        total_uncertainty = epistemic + aleatoric
        
        return {
            'predictions': mean_pred,
            'uncertainty': total_uncertainty.squeeze(1),
            'epistemic': epistemic.squeeze(1),
            'aleatoric': aleatoric.squeeze(1),
            'samples': samples
        }


class SeverityGrader(nn.Module):
    """Disease-specific severity grading."""
    
    SEVERITY_SCALES = {
        "pneumonia": ["no_pneumonia", "mild", "moderate", "severe"],
        "pneumothorax": ["no_ptx", "small", "moderate", "large", "tension"],
        "pulmonary_edema": ["no_edema", "interstitial", "alveolar"],
        "cardiomegaly": ["normal", "mild", "moderate", "severe"],
    }
    
    def __init__(self):
        super().__init__()
        
        # Severity classifier for each disease
        self.severity_classifiers = nn.ModuleDict({
            disease: self._build_classifier(len(grades))
            for disease, grades in self.SEVERITY_SCALES.items()
        })
    
    def _build_classifier(self, num_grades: int) -> nn.Module:
        """Build severity classifier for given number of grades."""
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_grades)
        )
    
    def forward(self, features: torch.Tensor, disease: str) -> Dict[str, torch.Tensor]:
        """
        Grade disease severity.
        
        Returns:
            {
                'grades': (batch, num_grades),
                'predicted_grade': (batch,),
                'confidence': (batch,)
            }
        """
        if disease not in self.severity_classifiers:
            return {
                'grades': None,
                'predicted_grade': None,
                'confidence': None,
                'error': f'Unknown disease: {disease}'
            }
        
        classifier = self.severity_classifiers[disease]
        logits = classifier(features)
        probs = torch.softmax(logits, dim=1)
        
        predicted_grade = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1)[0]
        
        grade_names = self.SEVERITY_SCALES[disease]
        grade_names_batch = [grade_names[g.item()] for g in predicted_grade]
        
        return {
            'grades': probs,
            'predicted_grade': predicted_grade,
            'grade_names': grade_names_batch,
            'confidence': confidence,
            'all_grade_names': grade_names
        }


class CalibratedPrediction:
    """Temperature scaling and other calibration methods."""
    
    @staticmethod
    def temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Apply temperature scaling for confidence calibration."""
        return torch.softmax(logits / temperature, dim=1)
    
    @staticmethod
    def find_optimal_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Find optimal temperature using validation data."""
        temperatures = np.linspace(0.5, 5.0, 50)
        best_temp = 1.0
        best_loss = float('inf')
        
        for temp in temperatures:
            probs = torch.softmax(logits / temp, dim=1)
            nll = torch.nn.functional.nll_loss(
                torch.log(probs + 1e-8), labels
            ).item()
            
            if nll < best_loss:
                best_loss = nll
                best_temp = temp
        
        return best_temp


class RobustConfidenceEstimator:
    """Multiple methods for confidence estimation."""
    
    @staticmethod
    def margin_confidence(predictions: torch.Tensor) -> torch.Tensor:
        """Confidence based on margin between top-2 predictions."""
        sorted_probs = torch.sort(predictions, descending=True)[0]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        return margin
    
    @staticmethod
    def entropy_based_confidence(predictions: torch.Tensor) -> torch.Tensor:
        """Confidence based on entropy (lower entropy = more confident)."""
        entropy = -torch.sum(predictions * torch.log(predictions + 1e-8), dim=1)
        # Normalize to [0, 1]
        max_entropy = torch.log(torch.tensor(predictions.shape[1], dtype=torch.float))
        confidence = 1.0 - (entropy / max_entropy)
        return confidence
    
    @staticmethod
    def variability_confidence(mc_samples: torch.Tensor) -> torch.Tensor:
        """
        Confidence based on MC Dropout variability.
        Lower variability = higher confidence.
        """
        # mc_samples: (n_samples, batch, num_diagnoses)
        variance = mc_samples.var(dim=0).mean(dim=1)
        confidence = 1.0 / (1.0 + variance)
        return confidence


if __name__ == "__main__":
    # Test uncertainty estimator
    batch_size = 4
    features = torch.randn(batch_size, 512)
    
    uncertainty_model = UncertaintyEstimator(num_diagnoses=14)
    result = uncertainty_model(features)
    
    print(f"Predictions shape: {result['predictions'].shape}")
    print(f"Uncertainty shape: {result['uncertainty'].shape}")
    print(f"Epistemic uncertainty: {result['epistemic'].mean().item():.4f}")
    print(f"Aleatoric uncertainty: {result['aleatoric'].mean().item():.4f}")
    
    # Test severity grader
    severity_model = SeverityGrader()
    severity_result = severity_model(features, "pneumonia")
    print(f"\nSeverity grades: {severity_result['grade_names']}")
    print(f"Confidence: {severity_result['confidence']}")
