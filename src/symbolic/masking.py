import torch
import torch.nn as nn
from typing import List, Dict, Tuple

class OntologyConstraintMasking(nn.Module):
    """
    Module 6: Ontology Constraint Masking
    Applies logit masking based on symbolic rule violations.
    Invalid diagnoses receive a logit mask of -inf.
    """
    def __init__(self, num_diagnoses: int):
        super().__init__()
        self.num_diagnoses = num_diagnoses

    def forward(self, logits: torch.Tensor, invalid_indices_batch: List[List[int]]) -> torch.Tensor:
        """
        Args:
            logits: Un-normalized predictions, shape (batch_size, num_diagnoses)
            invalid_indices_batch: List of lists containing indices of invalid diagnoses per instance in batch.
        Returns:
            Masked logits, where invalid diagnoses have -inf.
        """
        masked_logits = logits.clone()
        batch_size = logits.size(0)
        
        for b in range(batch_size):
            invalid_indices = invalid_indices_batch[b]
            if invalid_indices:
                # Apply mask: Set invalid logits to -inf
                masked_logits[b, invalid_indices] = float('-inf')
                
        return masked_logits
