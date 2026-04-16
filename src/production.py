"""
Production Monitoring & Audit Trail
===================================

Real-time monitoring, data drift detection, and audit trails.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class PredictionAudit:
    """Complete audit trail for each prediction."""
    
    prediction_id: str
    timestamp: str
    patient_id: str
    input_modalities: List[str]
    model_version: str
    predictions: Dict[str, float]
    confidence: float
    uncertainty: float
    severity_grades: Dict[str, str]
    symbolic_rules_fired: List[str]
    user_id: str = None
    user_action: str = None  # approved/rejected/modified
    feedback: str = None
    input_hash: str = None  # For data integrity
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AuditLogger:
    """Maintain comprehensive audit trail."""
    
    def __init__(self, log_file: str = "predictions_audit.log"):
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.predictions = []
    
    def log_prediction(self, audit: PredictionAudit):
        """Log prediction with full audit trail."""
        self.predictions.append(audit)
        
        # Log to file
        self.logger.info(json.dumps(audit.to_dict()))
    
    def log_user_feedback(self, prediction_id: str, user_id: str, 
                         feedback: str, action: str):
        """Log user feedback on prediction."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction_id': prediction_id,
            'user_id': user_id,
            'action': action,
            'feedback': feedback
        }
        self.logger.info(f"FEEDBACK: {json.dumps(entry)}")
    
    def get_prediction_audit(self, prediction_id: str) -> PredictionAudit:
        """Retrieve audit trail for prediction."""
        for pred in self.predictions:
            if pred.prediction_id == prediction_id:
                return pred
        return None
    
    def compute_input_hash(self, input_data: Dict) -> str:
        """Compute hash of input for integrity verification."""
        input_str = json.dumps(input_data, sort_keys=True, default=str)
        return hashlib.sha256(input_str.encode()).hexdigest()


class DataDriftDetector:
    """Detect distribution shifts in production data."""
    
    def __init__(self, reference_stats: Dict[str, Dict]):
        """
        Args:
            reference_stats: Mean, std, min, max from training data
        """
        self.reference_stats = reference_stats
        self.drift_detected = False
        self.drift_metrics = {}
    
    def compute_drift(self, current_batch: List[Dict]) -> Dict[str, float]:
        """
        Compute Kolmogorov-Smirnov statistic for distribution shift.
        """
        import numpy as np
        from scipy.stats import ks_2samp
        
        drift_scores = {}
        
        for feature, ref_stats in self.reference_stats.items():
            current_values = [s.get(feature) for s in current_batch if s.get(feature) is not None]
            
            if not current_values:
                continue
            
            # Simple check: mean shift
            current_mean = np.mean(current_values)
            ref_mean = ref_stats.get('mean', 0)
            ref_std = ref_stats.get('std', 1)
            
            # Z-score drift
            z_score = abs((current_mean - ref_mean) / (ref_std + 1e-6))
            drift_scores[feature] = z_score
            
            if z_score > 2.0:  # > 2 standard deviations
                self.drift_detected = True
        
        self.drift_metrics = drift_scores
        return drift_scores


class PerformanceMonitor:
    """Real-time performance monitoring."""
    
    def __init__(self):
        self.predictions = []
        self.user_feedback = []
    
    def track_prediction(self, pred_id: str, true_label: int = None,
                        predicted_label: int = None, confidence: float = None):
        """Track prediction for monitoring."""
        self.predictions.append({
            'id': pred_id,
            'true': true_label,
            'predicted': predicted_label,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'correct': true_label == predicted_label if true_label is not None else None
        })
    
    def get_recent_accuracy(self, n_recent: int = 100) -> Dict[str, float]:
        """Compute accuracy on recent predictions."""
        recent = [p for p in self.predictions if p['correct'] is not None][-n_recent:]
        
        if not recent:
            return {'accuracy': None, 'sample_size': 0}
        
        accuracy = sum(1 for p in recent if p['correct']) / len(recent)
        avg_confidence = sum(p['confidence'] for p in recent if p['confidence']) / len(recent)
        
        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'sample_size': len(recent),
            'trend': 'improving' if accuracy > 0.85 else 'concerning'
        }
    
    def get_alerts(self) -> List[Dict]:
        """Generate alerts for concerning patterns."""
        alerts = []
        
        recent_accuracy = self.get_recent_accuracy()
        if recent_accuracy['accuracy'] is not None and recent_accuracy['accuracy'] < 0.80:
            alerts.append({
                'severity': 'high',
                'message': f"Accuracy dropped to {recent_accuracy['accuracy']:.1%}",
                'action': 'Review recent predictions and model performance'
            })
        
        # Low confidence alert
        low_conf = [p for p in self.predictions[-100:] if p['confidence'] and p['confidence'] < 0.6]
        if len(low_conf) > 30:
            alerts.append({
                'severity': 'medium',
                'message': f"{len(low_conf)}/100 predictions have low confidence",
                'action': 'Investigate model stability'
            })
        
        return alerts


class ModelVersionControl:
    """Track model versions and rollbacks."""
    
    def __init__(self):
        self.versions = {}
        self.current_version = None
    
    def register_model(self, version_id: str, model_path: str, 
                      metadata: Dict = None):
        """Register new model version."""
        self.versions[version_id] = {
            'path': model_path,
            'created': datetime.now().isoformat(),
            'metadata': metadata or {},
            'performance': None,
            'status': 'testing'
        }
    
    def promote_to_production(self, version_id: str):
        """Promote model to production."""
        if version_id in self.versions:
            self.current_version = version_id
            self.versions[version_id]['status'] = 'production'
            self.versions[version_id]['promoted_at'] = datetime.now().isoformat()
    
    def rollback(self, version_id: str):
        """Rollback to previous version."""
        if version_id in self.versions:
            self.current_version = version_id
            self.versions[version_id]['status'] = 'production'
            self.versions[version_id]['rolled_back_at'] = datetime.now().isoformat()
    
    def get_version_history(self) -> List[Dict]:
        """Get version history."""
        return [
            {
                'version': v_id,
                **v_info
            }
            for v_id, v_info in self.versions.items()
        ]


if __name__ == "__main__":
    # Example: Create audit trail
    audit = PredictionAudit(
        prediction_id="pred_001",
        timestamp=datetime.now().isoformat(),
        patient_id="patient_123",
        input_modalities=["cxr", "labs"],
        model_version="v1.0",
        predictions={"pneumonia": 0.85, "normal": 0.15},
        confidence=0.85,
        uncertainty=0.12,
        severity_grades={"pneumonia": "moderate"},
        symbolic_rules_fired=["fever_cough_rule", "age_risk_rule"],
        user_id="radiologist_001"
    )
    
    logger = AuditLogger()
    logger.log_prediction(audit)
    
    print("Audit logged successfully")
