import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, List

from .encoders import VisionEncoder, TextEncoder, TabularEncoder, HistoryEncoder, MultimodalFusion
from .symbolic import OntologyConstraintMasking, OntologyConstraints, SymbolicVerifier

class MedSymbolModel(nn.Module):
    """
    Core MedSymbol Framework integrating Neural Encoders and Symbolic Logic.
    Implements Modules 1 through 7.
    """
    def __init__(self, 
                 num_diagnoses: int, 
                 tabular_input_dim: int, 
                 history_input_dim: int,
                 tau_low: float = 0.3,
                 tau_high: float = 1.5,
                 pretrained: bool = True):
        super().__init__()
        
        # Module 1: Multimodal Encoders & Fusion
        self.vision_encoder = VisionEncoder(pretrained=pretrained)
        self.text_encoder = TextEncoder(pretrained=pretrained)
        self.tabular_encoder = TabularEncoder(input_dim=tabular_input_dim)
        self.history_encoder = HistoryEncoder(input_dim=history_input_dim)
        self.fusion = MultimodalFusion()
        
        # Module 2-3: Diagnosis Prediction Head
        # Maps 512-dim fused representation to num_diagnoses logits
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_diagnoses)
        )
        
        # Module 4 thresholds
        self.tau_low = tau_low
        self.tau_high = tau_high
        
        # Modules 5-7: Symbolic Verifiers & Masking
        self.constraints = OntologyConstraints()
        self.verifier = SymbolicVerifier(self.constraints)
        self.masking = OntologyConstraintMasking(num_diagnoses=num_diagnoses)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Differentiable forward pass for training (Modules 1-3).
        Args:
            inputs: Dict containing 'vision', 'input_ids', 'attention_mask', 
                   'tabular', and 'history' tensors.
        Returns:
            Raw logits of shape (batch_size, num_diagnoses).
        """
        modality_features = {}
        
        if 'vision' in inputs:
            modality_features['vision'] = self.vision_encoder(inputs['vision'])
        if 'input_ids' in inputs and 'attention_mask' in inputs:
            modality_features['text'] = self.text_encoder(
                inputs['input_ids'], inputs['attention_mask']
            )
        if 'tabular' in inputs:
            modality_features['tabular'] = self.tabular_encoder(inputs['tabular'])
        if 'history' in inputs:
            modality_features['history'] = self.history_encoder(inputs['history'])
            
        # Module 1e: Fusion
        fused_rep = self.fusion(modality_features)
        
        # Module 2-3: Prediction
        logits = self.classifier(fused_rep)
        return logits

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """ Calculates Shannon Entropy for the batch. """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    @torch.no_grad()
    def predict(self, 
                inputs: Dict[str, torch.Tensor], 
                patient_data: List[Dict[str, Any]],
                diagnosis_map: Dict[int, str]) -> List[Dict[str, Any]]:
        """
        Full inference pipeline including Entropy Gating and Symbolic Verification.
        Args:
            inputs: Neural network inputs.
            patient_data: List of demographic/clinical facts per patient in batch.
            diagnosis_map: Mapping from logit index to diagnosis string name.
        Returns:
            List of result dictionaries.
        """
        # Step 1: Neural Forward Pass
        logits = self.forward(inputs)
        entropy = self.compute_entropy(logits)
        
        batch_size = logits.size(0)
        results = []
        
        # Step 2: Process each patient in the batch
        for i in range(batch_size):
            h = entropy[i].item()
            patient_logits = logits[i:i+1] # Keep batch dim
            
            result = {
                "entropy": h,
                "diagnoses": None,
                "path_taken": None,
                "verification": None
            }
            
            # Module 4: Entropy-Gated Controller
            if h < self.tau_low:
                # FAST PATH: Highly confident, skip Z3 verification
                result["path_taken"] = "FAST"
                probs = F.softmax(patient_logits, dim=-1)[0]
                result["diagnoses"] = self._format_top_k(probs, diagnosis_map)
                
            elif h >= self.tau_high:
                # DEFER PATH: Too uncertain, defer to human
                result["path_taken"] = "DEFER"
                result["diagnoses"] = [] # Deferring
                
            else:
                # STANDARD PATH: Moderate certainty, invoke Symbolic Verifier (Modules 5-6)
                result["path_taken"] = "STANDARD"
                
                # Check constraints logically
                invalid_indices = []
                verification_log = {}
                
                for idx, diag_name in diagnosis_map.items():
                    # Module 5: Run Z3 Verifier for this diagnosis
                    verif_res = self.verifier.verify_patient(diag_name, patient_data[i])
                    verification_log[diag_name] = verif_res
                    
                    if not self.verifier.is_fully_consistent(verif_res):
                        invalid_indices.append(idx)
                
                # Module 6: Masking
                # Apply -inf mask to invalid indices
                masked_logits = self.masking(patient_logits, [invalid_indices])
                probs = F.softmax(masked_logits, dim=-1)[0]
                
                result["diagnoses"] = self._format_top_k(probs, diagnosis_map)
                result["verification"] = verification_log
                
            results.append(result)
            
        return results

    def _format_top_k(self, probs: torch.Tensor, diagnosis_map: Dict[int, str], k: int = 3) -> List[Dict]:
        top_probs, top_indices = torch.topk(probs, k)
        return [
            {"diagnosis": diagnosis_map[idx.item()], "probability": p.item()}
            for p, idx in zip(top_probs, top_indices)
            if p.item() > 0 # Ignore zero'd out probabilities
        ]
