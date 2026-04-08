import torch
import torch.nn as nn
from typing import Dict

class MultimodalFusion(nn.Module):
    """
    Multimodal Fusion Layer (Module 1e)
    Combines Vision (768), Text (768), Tabular (256), and History (128) -> Total 1920.
    Output unified representation: 512 dimensions.
    """
    def __init__(self, in_features: int = 1920, output_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        
        # In a fully implemented model, you might use self-attention or cross-attention.
        # Here we use an MLP concatenative fusion as a strong baseline layer.
        self.fusion = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
        )
        self.output_dim = output_dim

    def forward(self, 
                modality_dict: Dict[str, torch.Tensor]
               ) -> torch.Tensor:
        """
        Args:
           modality_dict: Dictionary containing the encoded modalities:
                          - 'vision': (B, 768)
                          - 'text': (B, 768)
                          - 'tabular': (B, 256)
                          - 'history': (B, 128)
        Returns:
            Tensor of shape (batch, 512)
        """
        keys = ['vision', 'text', 'tabular', 'history']
        
        # Verify all tensors exist and concatenate along features (dim=1)
        tensors_to_concat = [modality_dict[k] for k in keys if k in modality_dict]
        
        if not tensors_to_concat:
            raise ValueError("No modalities provided for fusion")
            
        merged = torch.cat(tensors_to_concat, dim=1)
        return self.fusion(merged)
