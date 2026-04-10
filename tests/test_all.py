#!/usr/bin/env python3
"""
Comprehensive Unit Tests for MedSymbol
======================================
Tests for all 7 modules and supporting infrastructure.

Usage:
    pytest tests/ -v
    pytest tests/ --cov=src --cov-report=html
"""

import sys
import os
import pytest
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import MedSymbolModel
from src.encoders import VisionEncoder, TextEncoder, TabularEncoder, HistoryEncoder, MultimodalFusion
from src.symbolic import OntologyConstraints, OntologyConstraintMasking, SymbolicVerifier
from src.utils.data_loader import NIHCXR14Dataset


# ============================================================================
# Module 1: Vision Encoder Tests
# ============================================================================

class TestVisionEncoder:
    """Test Module 1: Vision Encoder (ResNet50)."""
    
    def test_vision_encoder_output_shape(self):
        """Test vision encoder produces correct output shape."""
        encoder = VisionEncoder()
        batch_size = 4
        dummy_images = torch.randn(batch_size, 3, 224, 224)
        
        features = encoder(dummy_images)
        
        assert features.shape == (batch_size, 768), f"Expected shape (4, 768), got {features.shape}"
    
    def test_vision_encoder_cuda_support(self):
        """Test vision encoder works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        encoder = VisionEncoder().cuda()
        dummy_images = torch.randn(2, 3, 224, 224).cuda()
        
        features = encoder(dummy_images)
        
        assert features.device.type == 'cuda'
    
    def test_vision_encoder_batch_processing(self):
        """Test vision encoder handles different batch sizes."""
        encoder = VisionEncoder()
        
        for batch_size in [1, 2, 8, 16, 32]:
            dummy_images = torch.randn(batch_size, 3, 224, 224)
            features = encoder(dummy_images)
            assert features.shape == (batch_size, 768)


# ============================================================================
# Module 2: Text Encoder Tests
# ============================================================================

class TestTextEncoder:
    """Test Module 2: Text Encoder (BioBERT)."""
    
    def test_text_encoder_output_shape(self):
        """Test text encoder produces correct embedding."""
        encoder = TextEncoder()
        dummy_texts = ["patient has fever and cough", "normal examination"]
        
        embeddings = encoder(dummy_texts)
        
        assert embeddings.shape == (2, 768), f"Expected shape (2, 768), got {embeddings.shape}"
    
    def test_text_encoder_empty_handling(self):
        """Test text encoder handles empty/missing text."""
        encoder = TextEncoder()
        
        embeddings = encoder([""])
        
        assert embeddings.shape == (1, 768)
        assert not torch.isnan(embeddings).any()


# ============================================================================
# Module 3: Tabular Encoder Tests
# ============================================================================

class TestTabularEncoder:
    """Test Module 3: Tabular Encoder."""
    
    def test_tabular_encoder_output_shape(self):
        """Test tabular encoder output shape."""
        encoder = TabularEncoder(input_dim=10)
        batch_size = 4
        tabular_data = torch.randn(batch_size, 10)
        
        features = encoder(tabular_data)
        
        assert features.shape == (batch_size, 128)
    
    def test_tabular_encoder_different_dims(self):
        """Test tabular encoder with different input dimensions."""
        for input_dim in [5, 10, 20, 32]:
            encoder = TabularEncoder(input_dim=input_dim)
            data = torch.randn(2, input_dim)
            output = encoder(data)
            assert output.shape == (2, 128)


# ============================================================================
# Module 4: History Encoder Tests
# ============================================================================

class TestHistoryEncoder:
    """Test Module 4: History Encoder (Temporal)."""
    
    def test_history_encoder_output_shape(self):
        """Test history encoder sequence output."""
        encoder = HistoryEncoder(input_dim=5)
        batch_size = 4
        seq_length = 10
        history_seq = torch.randn(batch_size, seq_length, 5)
        
        output = encoder(history_seq)
        
        assert output.shape == (batch_size, 128)
    
    def test_history_encoder_variable_sequence_lengths(self):
        """Test history encoder with different sequence lengths."""
        encoder = HistoryEncoder(input_dim=5)
        
        for seq_len in [5, 10, 20, 50]:
            history_seq = torch.randn(2, seq_len, 5)
            output = encoder(history_seq)
            assert output.shape == (2, 128)


# ============================================================================
# Module 5: Multimodal Fusion Tests
# ============================================================================

class TestMultimodalFusion:
    """Test Module 5: Multimodal Fusion (Cross-attention)."""
    
    def test_fusion_output_shape(self):
        """Test multimodal fusion output shape."""
        fusion = MultimodalFusion()
        
        # Create dummy multimodal inputs
        vision_features = torch.randn(4, 512)
        text_embedding = torch.randn(4, 768)
        tabular_features = torch.randn(4, 128)
        history_features = torch.randn(4, 128)
        
        fused = fusion(vision_features, text_embedding, tabular_features, history_features)
        
        assert fused.shape == (4, 512)
    
    def test_fusion_attention_weights(self):
        """Test fusion produces valid attention weights."""
        fusion = MultimodalFusion()
        
        inputs = [torch.randn(2, 512), torch.randn(2, 768), 
                 torch.randn(2, 128), torch.randn(2, 128)]
        
        output = fusion(*inputs)
        
        # Output should be valid (no NaNs or Infs)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# ============================================================================
# Module 6: Symbolic Masking Tests
# ============================================================================

class TestSymbolicMasking:
    """Test Module 6: Symbolic Constraint Masking."""
    
    def test_masking_reduces_logits(self):
        """Test masking applies constraints to logits."""
        masking = OntologyConstraintMasking(num_diagnoses=14)
        
        logits = torch.randn(4, 14)
        age_values = torch.tensor([30.0, 50.0, 70.0, 20.0], dtype=torch.float32)
        
        masked_logits = masking.apply_age_mask(logits, age_values)
        
        assert masked_logits.shape == logits.shape
    
    def test_masking_preserves_output_shape(self):
        """Test masking preserves output shape."""
        masking = OntologyConstraintMasking(num_diagnoses=14)
        logits = torch.randn(8, 14)
        
        # Test with different age constraints
        for age in [10, 30, 50, 70, 90]:
            age_tensor = torch.full((8,), float(age))
            masked = masking.apply_age_mask(logits, age_tensor)
            assert masked.shape == (8, 14)


# ============================================================================
# Module 7: Proof Generator Tests
# ============================================================================

class TestProofGenerator:
    """Test Module 7: Symbolic Proof Generator."""
    
    def test_verifier_constraint_checking(self):
        """Test symbolic verifier constraint checks."""
        constraints = OntologyConstraints()
        verifier = SymbolicVerifier(constraints)
        
        patient_data = {
            "age": 45,
            "sex": "M",
            "comorbidities": ["hypertension"]
        }
        
        results = verifier.verify_patient("Cardiomegaly", patient_data)
        
        assert isinstance(results, dict)
        assert "age_validity" in results


# ============================================================================
# Ontology Constraints Tests
# ============================================================================

class TestOntologyConstraints:
    """Test constraint logic and medical rules."""
    
    def test_age_constraint_validation(self):
        """Test age constraint checking."""
        constraints = OntologyConstraints()
        
        # Pneumothorax: typically 15-40 years
        assert constraints.check_age_constraint("Pneumothorax", 30) == True
        assert constraints.check_age_constraint("Pneumothorax", 10) == False
        assert constraints.check_age_constraint("Pneumothorax", 50) == False
    
    def test_sex_constraint_validation(self):
        """Test sex-based constraints."""
        constraints = OntologyConstraints()
        
        # Pneumothorax more common in males
        assert constraints.check_sex_constraint("Pneumothorax", "M") == True
    
    def test_comorbidity_risk_calculation(self):
        """Test comorbidity risk scoring."""
        constraints = OntologyConstraints()
        
        # Pneumonia with immunocompromise
        risk = constraints.calculate_comorbidity_risk("Pneumonia", 
                                                     ["immunocompromised"], 
                                                     age=50)
        assert risk > 1.0
    
    def test_symptom_matching(self):
        """Test symptom pattern matching."""
        constraints = OntologyConstraints()
        
        # Pneumonia with fever and cough
        compatible, score = constraints.match_symptoms("Pneumonia", 
                                                      ["fever", "cough", "dyspnea"])
        assert compatible == True
        assert score > 0.5
    
    def test_risk_stratification(self):
        """Test risk classification."""
        constraints = OntologyConstraints()
        
        patient_data = {
            "age": 70,
            "sex": "M",
            "comorbidities": ["heart_failure"],
            "symptoms": ["dyspnea"]
        }
        
        risk = constraints.get_risk_stratification("Cardiomegaly", patient_data)
        assert risk in ["low", "intermediate", "high"]


# ============================================================================
# Full Model Integration Tests
# ============================================================================

class TestMedSymbolModel:
    """Test complete MedSymbol model."""
    
    def test_model_initialization(self):
        """Test model instantiation."""
        model = MedSymbolModel(
            num_diagnoses=14,
            tabular_input_dim=10,
            history_input_dim=5
        )
        
        assert model is not None
        assert model.classifier is not None
    
    def test_model_forward_pass(self):
        """Test full forward pass through model."""
        model = MedSymbolModel(
            num_diagnoses=14,
            tabular_input_dim=10,
            history_input_dim=5
        )
        model.eval()
        
        # Create dummy inputs using correct keys for forward method
        sample_input = {
            'vision': torch.randn(2, 3, 224, 224),
            'input_ids': torch.randint(0, 1000, (2, 128)),
            'attention_mask': torch.ones(2, 128, dtype=torch.long),
            'tabular': torch.randn(2, 10),
            'history': torch.randn(2, 5, 128)
        }
        
        with torch.no_grad():
            logits = model(sample_input)
        
        assert logits.shape == (2, 14)
        assert not torch.isnan(logits).any()
    
    def test_model_to_device(self):
        """Test model device placement."""
        model = MedSymbolModel(num_diagnoses=14, tabular_input_dim=10, history_input_dim=5)
        
        # Test CPU
        model.cpu()
        assert next(model.parameters()).device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model.cuda()
            assert next(model.parameters()).device.type == 'cuda'


# ============================================================================
# Data Loading Tests
# ============================================================================

class TestDataLoader:
    """Test dataset loading and preprocessing."""
    
    def test_dataset_initialization(self):
        """Test dataset creation."""
        dataset = NIHCXR14Dataset(
            csv_file="dummy.csv",
            img_dir="./data/raw/nih_cxr14",
            split='train'
        )
        
        assert len(dataset) > 0
    
    def test_dataset_getitem(self):
        """Test single sample retrieval."""
        dataset = NIHCXR14Dataset(
            csv_file="dummy.csv",
            img_dir="./data/raw/nih_cxr14",
            split='train'
        )
        
        sample = dataset[0]
        
        assert 'image' in sample
        assert 'labels' in sample


# ============================================================================
# Performance Benchmarking Tests
# ============================================================================

class TestPerformanceBenchmarks:
    """Benchmark model inference speed."""
    
    @pytest.mark.benchmark
    def test_model_inference_speed(self, benchmark):
        """Benchmark single inference."""
        model = MedSymbolModel(num_diagnoses=14, tabular_input_dim=10, history_input_dim=5)
        model.eval()
        
        sample_input = {
            'vision': torch.randn(1, 3, 224, 224),
            'input_ids': torch.randint(0, 1000, (1, 128)),
            'attention_mask': torch.ones(1, 128, dtype=torch.long),
            'tabular': torch.randn(1, 10),
            'history': torch.randn(1, 5, 128)
        }
        
        def infer():
            with torch.no_grad():
                return model(sample_input)
        
        benchmark(infer)
    
    @pytest.mark.benchmark
    def test_batch_processing_speed(self, benchmark):
        """Benchmark batch inference."""
        model = MedSymbolModel(num_diagnoses=14, tabular_input_dim=10, history_input_dim=5)
        model.eval()
        
        sample_input = {
            'vision': torch.randn(32, 3, 224, 224),
            'input_ids': torch.randint(0, 1000, (32, 128)),
            'attention_mask': torch.ones(32, 128, dtype=torch.long),
            'tabular': torch.randn(32, 10),
            'history': torch.randn(32, 5, 128)
        }
        
        def infer():
            with torch.no_grad():
                return model(sample_input)
        
        benchmark(infer)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
