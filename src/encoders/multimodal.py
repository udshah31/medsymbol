"""
Multi-Modality Medical Data Encoder
===================================

Extends MedSymbol to handle:
- Multiple imaging modalities (CXR, CT, MRI)
- Clinical notes (unstructured text)
- Lab values (structured data)
- Vital signs
- Missing data handling
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import warnings


class CTEncoder(nn.Module):
    """CT scan encoder with 3D convolutions."""
    
    def __init__(self, output_dim: int = 768):
        super().__init__()
        # 3D CNN for volumetric data
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.projection = nn.Linear(128, output_dim)
    
    def forward(self, ct_volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ct_volume: (batch, depth, height, width) or (batch, 1, depth, height, width)
        
        Returns:
            features: (batch, output_dim)
        """
        if ct_volume.dim() == 4:
            ct_volume = ct_volume.unsqueeze(1)
        
        features = self.conv_layers(ct_volume)
        features = features.view(features.size(0), -1)
        return self.projection(features)


class LabValueEncoder(nn.Module):
    """Encoder for structured lab values with normalization."""
    
    # Reference ranges (min, max) for normalization
    LAB_RANGES = {
        "wbc": (4.5, 11.0),  # 1000 cells/µL
        "hemoglobin": (12.0, 17.5),  # g/dL
        "hematocrit": (36.0, 46.0),  # %
        "platelets": (150.0, 400.0),  # 1000/µL
        "glucose": (70.0, 100.0),  # mg/dL fasting
        "creatinine": (0.6, 1.2),  # mg/dL
        "bun": (7.0, 20.0),  # mg/dL
        "sodium": (136.0, 145.0),  # mEq/L
        "potassium": (3.5, 5.0),  # mEq/L
        "ph": (7.35, 7.45),
        "pco2": (35.0, 45.0),  # mmHg
        "po2": (80.0, 100.0),  # mmHg
        "lactate": (0.5, 2.0),  # mmol/L
    }
    
    def __init__(self, output_dim: int = 256, num_labs: int = 13):
        super().__init__()
        self.output_dim = output_dim
        self.num_labs = num_labs
        
        # MLPs for feature extraction
        self.value_encoder = nn.Sequential(
            nn.Linear(num_labs, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # Pathological flag indicator (per lab)
        self.abnormality_detector = nn.Sequential(
            nn.Linear(num_labs, 128),
            nn.ReLU(),
            nn.Linear(128, num_labs),
            nn.Sigmoid(),
        )
        
        self.projection = nn.Linear(256 + num_labs, output_dim)
    
    def forward(self, lab_values: torch.Tensor, lab_names: List[str] = None) -> torch.Tensor:
        """
        Args:
            lab_values: (batch, num_labs) normalized lab values
            lab_names: List of lab names for reference
        
        Returns:
            features: (batch, output_dim)
        """
        # Encode raw values
        value_features = self.value_encoder(lab_values)
        
        # Detect abnormalities (simplified: > 1.5 std dev from mean)
        abnormality_flags = self.abnormality_detector(lab_values)
        
        # Combine features
        combined = torch.cat([value_features, abnormality_flags], dim=1)
        return self.projection(combined)


class ClinicalNoteEncoder(nn.Module):
    """Encoder for unstructured clinical notes."""
    
    def __init__(self, vocab_size: int = 2000, embedding_dim: int = 128, 
                 output_dim: int = 768):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM for sequential context
        self.lstm = nn.LSTM(
            embedding_dim, 256, num_layers=2, 
            batch_first=True, bidirectional=True, dropout=0.2
        )
        
        # Attention over LSTM outputs
        self.attention = nn.MultiheadAttention(512, num_heads=8, dropout=0.1)
        
        self.projection = nn.Linear(512, output_dim)
    
    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) tokenized note
            attention_mask: (batch, seq_len) mask for padding
        
        Returns:
            features: (batch, output_dim)
        """
        embeddings = self.embedding(token_ids)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(embeddings)
        
        # Self-attention
        if attention_mask is not None:
            # Convert attention_mask to attention weights
            attn_mask = attention_mask == 0
            attn_out, _ = self.attention(
                lstm_out.transpose(0, 1),
                lstm_out.transpose(0, 1),
                lstm_out.transpose(0, 1),
                key_padding_mask=attn_mask
            )
            lstm_out = attn_out.transpose(0, 1)
        
        # Mean pooling
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1)
            features = lstm_out.sum(dim=1) / lengths.unsqueeze(1)
        else:
            features = lstm_out.mean(dim=1)
        
        return self.projection(features)


class VitalSignsEncoder(nn.Module):
    """Encoder for vital signs (heart rate, BP, temperature, O2 sat)."""
    
    VITAL_RANGES = {
        "heart_rate": (60, 100),  # bpm
        "systolic_bp": (90, 120),  # mmHg
        "diastolic_bp": (60, 80),  # mmHg
        "temperature": (36.5, 37.5),  # °C
        "o2_saturation": (95, 100),  # %
        "respiratory_rate": (12, 20),  # breaths/min
    }
    
    def __init__(self, output_dim: int = 128):
        super().__init__()
        
        num_vitals = len(self.VITAL_RANGES)
        
        self.encoder = nn.Sequential(
            nn.Linear(num_vitals, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.projection = nn.Linear(128, output_dim)
    
    def forward(self, vital_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vital_values: (batch, num_vitals) vital sign values
        
        Returns:
            features: (batch, output_dim)
        """
        encoded = self.encoder(vital_values)
        return self.projection(encoded)


class MissingModalityHandler:
    """Handle missing modalities gracefully with learned replacements."""
    
    def __init__(self, modality_dim: int = 256):
        self.modality_dim = modality_dim
        # Learnable tokens for missing modalities
        self.missing_tokens = nn.ParameterDict({
            'image': nn.Parameter(torch.randn(1, modality_dim)),
            'ct': nn.Parameter(torch.randn(1, modality_dim)),
            'labs': nn.Parameter(torch.randn(1, modality_dim)),
            'notes': nn.Parameter(torch.randn(1, modality_dim)),
            'vitals': nn.Parameter(torch.randn(1, modality_dim)),
        })
    
    def handle_missing(self, features: torch.Tensor, modality: str, 
                      batch_size: int) -> torch.Tensor:
        """Replace missing modality with learned token."""
        if modality not in self.missing_tokens:
            # Default to zero
            return torch.zeros(batch_size, self.modality_dim)
        
        token = self.missing_tokens[modality]
        return token.expand(batch_size, -1)


class MultimodalMedicalEncoder(nn.Module):
    """
    Comprehensive multi-modal encoder for medical data.
    
    Handles:
    - CXR images
    - CT scans
    - Clinical notes
    - Lab values
    - Vital signs
    - Missing data
    """
    
    def __init__(self, output_dim: int = 512, handle_missing: bool = True):
        super().__init__()
        
        self.output_dim = output_dim
        self.handle_missing = handle_missing
        
        # Individual encoders
        self.cxr_encoder = nn.Identity()  # Use existing VisionEncoder
        self.ct_encoder = CTEncoder(output_dim=256)
        self.note_encoder = ClinicalNoteEncoder(output_dim=256)
        self.lab_encoder = LabValueEncoder(output_dim=256)
        self.vital_encoder = VitalSignsEncoder(output_dim=256)
        
        # Missing modality handler
        if handle_missing:
            self.missing_handler = MissingModalityHandler(modality_dim=256)
        
        # Cross-modal attention fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1
        )
        
        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(256 * 5, output_dim),  # 5 modalities
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(self, multimodal_data: Dict[str, Optional[torch.Tensor]],
                modality_mask: Dict[str, bool] = None) -> torch.Tensor:
        """
        Args:
            multimodal_data: Dict with optional tensors:
                - 'cxr': (batch, 3, 224, 224)
                - 'ct': (batch, depth, height, width)
                - 'notes': (batch, seq_len) token ids
                - 'labs': (batch, num_labs)
                - 'vitals': (batch, num_vitals)
            modality_mask: Dict indicating which modalities are present
        
        Returns:
            fused_features: (batch, output_dim)
        """
        batch_size = next(
            (v.shape[0] for v in multimodal_data.values() if v is not None),
            1
        )
        
        features_list = []
        
        # Process each modality
        if multimodal_data.get('cxr') is not None:
            features_list.append(self.cxr_encoder(multimodal_data['cxr']))
        elif self.handle_missing:
            features_list.append(self.missing_handler.handle_missing(
                None, 'image', batch_size
            ))
        
        if multimodal_data.get('ct') is not None:
            features_list.append(self.ct_encoder(multimodal_data['ct']))
        elif self.handle_missing:
            features_list.append(self.missing_handler.handle_missing(
                None, 'ct', batch_size
            ))
        
        if multimodal_data.get('notes') is not None:
            note_features = self.note_encoder(
                multimodal_data['notes'],
                multimodal_data.get('note_attention_mask')
            )
            features_list.append(note_features)
        elif self.handle_missing:
            features_list.append(self.missing_handler.handle_missing(
                None, 'notes', batch_size
            ))
        
        if multimodal_data.get('labs') is not None:
            features_list.append(self.lab_encoder(multimodal_data['labs']))
        elif self.handle_missing:
            features_list.append(self.missing_handler.handle_missing(
                None, 'labs', batch_size
            ))
        
        if multimodal_data.get('vitals') is not None:
            features_list.append(self.vital_encoder(multimodal_data['vitals']))
        elif self.handle_missing:
            features_list.append(self.missing_handler.handle_missing(
                None, 'vitals', batch_size
            ))
        
        # Stack features
        if not features_list:
            warnings.warn("No modalities provided, returning zero features")
            return torch.zeros(batch_size, self.output_dim)
        
        # Concatenate all modality features
        stacked = torch.stack(features_list, dim=1)  # (batch, num_modalities, 256)
        
        # Apply cross-modal attention
        attn_out, _ = self.fusion(stacked, stacked, stacked)
        
        # Flatten and project
        flattened = attn_out.reshape(batch_size, -1)
        fused = self.final_projection(flattened)
        
        return fused


if __name__ == "__main__":
    # Test multi-modal encoder
    encoder = MultimodalMedicalEncoder(output_dim=512)
    
    # Sample multi-modal data
    batch_size = 2
    multimodal_data = {
        'cxr': torch.randn(batch_size, 3, 224, 224),
        'labs': torch.randn(batch_size, 13),
        'vitals': torch.randn(batch_size, 6),
        'notes': None,  # Missing modality
        'ct': None,  # Missing modality
    }
    
    output = encoder(multimodal_data)
    print(f"Output shape: {output.shape}")  # Should be (batch_size, 512)
