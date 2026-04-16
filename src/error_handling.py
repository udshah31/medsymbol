"""
Robust Error Handling & Missing Data Management
==============================================

Handle edge cases, missing modalities, and graceful degradation.
"""

from typing import Dict, Optional, Any, List
import logging
import warnings


class MissingDataHandler:
    """Handle missing input data gracefully."""
    
    # Imputation strategies
    STRATEGIES = {
        'mean': 'Fill with population mean',
        'median': 'Fill with population median',
        'forward_fill': 'Use previous value',
        'learned_token': 'Use learned replacement token',
        'skip': 'Skip this modality'
    }
    
    def __init__(self, strategy: str = 'learned_token'):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy = strategy
    
    def handle_missing_modality(self, modality: str, batch_size: int, 
                                reference_stats: Dict = None) -> Any:
        """Handle missing modality with specified strategy."""
        
        if self.strategy == 'mean':
            if reference_stats and modality in reference_stats:
                mean = reference_stats[modality].get('mean')
                return [mean] * batch_size
        
        elif self.strategy == 'learned_token':
            # Return special token for missing modality
            return [f'<missing_{modality}>' for _ in range(batch_size)]
        
        elif self.strategy == 'skip':
            return None
        
        return None


class ErrorHandler:
    """Comprehensive error handling for medical AI."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_log = []
    
    def handle_inference_error(self, error: Exception, patient_id: str, 
                               error_context: Dict) -> Dict[str, str]:
        """
        Handle errors during inference gracefully.
        
        Returns safe fallback response.
        """
        error_entry = {
            'patient_id': patient_id,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': error_context
        }
        self.error_log.append(error_entry)
        self.logger.error(f"Inference error for patient {patient_id}: {error}")
        
        # Fallback response
        return {
            'status': 'error',
            'message': 'Model inference failed',
            'fallback': 'Manual radiologist review recommended',
            'error_id': f"ERR_{patient_id}_{len(self.error_log)}",
            'action': 'Contact system administrator'
        }
    
    def validate_input(self, input_data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate input data for common issues.
        
        Returns (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for NaN/Inf
        for key, value in input_data.items():
            if isinstance(value, (int, float)):
                if value != value:  # NaN check
                    errors.append(f"{key} contains NaN")
                elif abs(value) == float('inf'):
                    errors.append(f"{key} contains Infinity")
        
        # Check for empty inputs
        if not input_data:
            errors.append("Empty input data")
        
        # Check for required modalities
        if len([v for v in input_data.values() if v is not None]) == 0:
            errors.append("All modalities missing")
        
        return len(errors) == 0, errors
    
    def handle_data_quality_issue(self, issue: str, severity: str, 
                                 action: str) -> Dict[str, str]:
        """Log and handle data quality issues."""
        
        if severity == "critical":
            self.logger.critical(f"Critical data quality issue: {issue}")
            return {
                'action': 'STOP_INFERENCE',
                'message': f'Critical issue: {issue}',
                'recommendation': action
            }
        elif severity == "warning":
            self.logger.warning(f"Data quality warning: {issue}")
            return {
                'action': 'LOG_AND_CONTINUE',
                'message': f'Warning: {issue}',
                'recommendation': action
            }
        
        return {'action': 'CONTINUE'}


class ValidityChecker:
    """Check predictions for clinical validity."""
    
    @staticmethod
    def check_prediction_validity(predictions: Dict[str, float], 
                                 patient_data: Dict) -> Tuple[bool, List[str]]:
        """
        Check if predictions make clinical sense.
        
        Returns (is_valid, list_of_warnings)
        """
        warnings_list = []
        
        # Check for conflicting diagnoses
        if predictions.get('heart_failure', 0) > 0.8 and predictions.get('normal', 0) > 0.7:
            warnings_list.append("Heart failure + Normal findings conflict")
        
        # Check age-disease compatibility
        patient_age = patient_data.get('age', 50)
        if patient_age < 10 and predictions.get('heart_failure', 0) > 0.8:
            warnings_list.append("Heart failure unlikely in child")
        
        # Check probability calibration
        max_prob = max(predictions.values())
        if max_prob < 0.3 and not any(p > 0.2 for p in predictions.values()):
            warnings_list.append("All predictions have low confidence - unreliable")
        
        # Check for extreme probabilities
        for disease, prob in predictions.items():
            if prob < 0.0 or prob > 1.0:
                warnings_list.append(f"{disease} probability out of bounds: {prob}")
        
        is_valid = len(warnings_list) == 0
        return is_valid, warnings_list
    
    @staticmethod
    def sanity_check_severity(disease: str, severity: str, 
                             probability: float) -> Tuple[bool, str]:
        """Check if severity matches probability."""
        
        if severity == "severe" and probability < 0.5:
            return False, "Severe diagnosis with low confidence is suspicious"
        
        if severity == "mild" and probability > 0.95:
            return True, "OK: Mild presentation with high confidence"
        
        return True, "OK"


class GracefulDegradation:
    """Degrade gracefully when components fail."""
    
    @staticmethod
    def fallback_prediction(available_modalities: List[str]) -> Dict[str, Any]:
        """Generate fallback prediction from available data."""
        
        if 'cxr' in available_modalities and 'labs' in available_modalities:
            return {
                'mode': 'fallback_multimodal',
                'confidence': 'low',
                'message': 'Using reduced feature set'
            }
        elif 'cxr' in available_modalities:
            return {
                'mode': 'fallback_imaging_only',
                'confidence': 'very_low',
                'message': 'Using imaging only - clinical context missing'
            }
        elif 'labs' in available_modalities:
            return {
                'mode': 'fallback_labs_only',
                'confidence': 'very_low',
                'message': 'Using labs only - imaging missing'
            }
        else:
            return {
                'mode': 'no_fallback_possible',
                'confidence': 'none',
                'message': 'Insufficient data for inference',
                'action': 'Manual review required'
            }
    
    @staticmethod
    def use_ensemble_prediction(predictions_dict: Dict[str, Dict]) -> Dict[str, float]:
        """
        Use ensemble of predictions when primary model fails.
        
        Average predictions from multiple models/approaches.
        """
        import numpy as np
        
        all_diseases = set()
        for preds in predictions_dict.values():
            all_diseases.update(preds.keys())
        
        ensemble_pred = {}
        for disease in all_diseases:
            probs = [preds.get(disease, 0) for preds in predictions_dict.values()]
            ensemble_pred[disease] = np.mean(probs)
        
        return ensemble_pred


class EdgeCaseHandler:
    """Handle specific edge cases."""
    
    EDGE_CASES = {
        'pediatric': {
            'threshold': lambda age: age < 18,
            'adjustments': ['use_pediatric_norms', 'adjust_feature_ranges']
        },
        'geriatric': {
            'threshold': lambda age: age > 75,
            'adjustments': ['increased_comorbidity_weight', 'functional_status_consideration']
        },
        'pregnancy': {
            'threshold': lambda sex, preg_status: sex == 'F' and preg_status,
            'adjustments': ['avoid_certain_imaging', 'use_pregnancy_safe_protocols']
        },
        'critical': {
            'threshold': lambda severity: severity == 'critical',
            'adjustments': ['urgent_review', 'escalate_to_critical_care']
        }
    }
    
    @staticmethod
    def detect_edge_case(patient_data: Dict) -> List[str]:
        """Detect if patient falls into edge case categories."""
        
        edge_cases = []
        
        for case_name, config in EdgeCaseHandler.EDGE_CASES.items():
            threshold_fn = config.get('threshold')
            
            try:
                if case_name == 'pediatric':
                    if threshold_fn(patient_data.get('age', 50)):
                        edge_cases.append(case_name)
                elif case_name == 'geriatric':
                    if threshold_fn(patient_data.get('age', 50)):
                        edge_cases.append(case_name)
                elif case_name == 'pregnancy':
                    if threshold_fn(patient_data.get('sex'), patient_data.get('pregnant')):
                        edge_cases.append(case_name)
                elif case_name == 'critical':
                    if threshold_fn(patient_data.get('severity')):
                        edge_cases.append(case_name)
            except Exception as e:
                logging.warning(f"Error checking {case_name} edge case: {e}")
        
        return edge_cases


if __name__ == "__main__":
    # Test error handling
    handler = ErrorHandler()
    
    # Test input validation
    valid, errors = handler.validate_input({'temperature': float('nan')})
    print(f"Valid: {valid}, Errors: {errors}")
    
    # Test edge case detection
    patient = {'age': 8, 'sex': 'M', 'pregnant': False}
    edge_cases = EdgeCaseHandler.detect_edge_case(patient)
    print(f"Edge cases: {edge_cases}")
