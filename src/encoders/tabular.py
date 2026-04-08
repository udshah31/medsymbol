import torch
import torch.nn as nn

class TabularEncoder(nn.Module):
    """
    Tabular Encoder (Module 1c)
    Uses MLP for laboratory values (simplified from TabNet for stability).
    Input: N numeric/categorical lab vital features.
    Output dimension: 256
    """
    def __init__(self, input_dim: int, output_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Simple MLP for tabular features
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU()
        )

    def forward(self, lab_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lab_features: Tensor of shape (batch, input_dim) 
        Returns:
            Tensor of shape (batch, 256)
        """
        return self.mlp(lab_features)
