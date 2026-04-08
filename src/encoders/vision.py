import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class VisionEncoder(nn.Module):
    """
    Vision Encoder (Module 1a)
    Uses ViT-B/16 to encode Chest X-ray images.
    Output dimension: 768
    """
    def __init__(self, pretrained: bool = True, output_dim: int = 768):
        super().__init__()
        if pretrained:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        else:
            config = ViTConfig(image_size=224, patch_size=16, hidden_size=768)
            self.vit = ViTModel(config)
            
        self.output_dim = output_dim
        
        # Projection head if we ever want a different output dimension
        if output_dim != 768:
            self.proj = nn.Linear(768, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Tensor of shape (batch_size, num_channels, height, width)
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        outputs = self.vit(pixel_values=pixel_values)
        # We use the pooler_output (representation of the [CLS] token)
        pooled_output = outputs.pooler_output
        return self.proj(pooled_output)
