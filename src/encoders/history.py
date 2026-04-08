import torch
import torch.nn as nn

class HistoryEncoder(nn.Module):
    """
    Patient History Encoder (Module 1d)
    Uses MLP for patient demographics (age, sex, insurance, prior diagnoses flags).
    Output dimension: 128
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        self.output_dim = output_dim

    def forward(self, demographics: torch.Tensor) -> torch.Tensor:
        """
        Args:
            demographics: Tensor of shape (batch, input_dim)
        Returns:
            Tensor of shape (batch, 128)
        """
        return self.mlp(demographics)
