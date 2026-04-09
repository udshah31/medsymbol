import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertModel

class TextEncoder(nn.Module):
    """
    Language Encoder (Module 1b)
    Uses BioBERT for clinical notes.
    Output dimension: 768
    """
    def __init__(self, pretrained: bool = True, output_dim: int = 768):
        super().__init__()
        
        # dmis-lab/biobert-v1.1 is standard, returning 768 features.
        model_name = "dmis-lab/biobert-v1.1"
        if pretrained:
            self.bert = AutoModel.from_pretrained(model_name)
        else:
            # Use a standard BERT-base config locally (no network required)
            config = BertConfig()
            self.bert = BertModel(config)
            
        self.output_dim = output_dim
        
        if output_dim != 768:
            self.proj = nn.Linear(768, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of token ids (batch_size, sequence_length)
            attention_mask: Tensor of mask (batch_size, sequence_length)
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        return self.proj(pooled_output)
