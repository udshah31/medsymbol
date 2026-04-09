"""
Unit tests for OntologyConstraintMasking (src/symbolic/masking.py).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from symbolic.masking import OntologyConstraintMasking


@pytest.fixture
def masking():
    return OntologyConstraintMasking(num_diagnoses=5)


class TestOntologyConstraintMasking:
    def test_no_invalid_indices(self, masking):
        """With no invalid indices, logits should be unchanged."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        masked = masking(logits, [[]])
        assert torch.allclose(logits, masked)

    def test_single_invalid_index(self, masking):
        """A single invalid index should be set to -inf."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        masked = masking(logits, [[2]])
        assert torch.isinf(masked[0, 2]) and masked[0, 2] < 0
        # Other indices unchanged
        assert masked[0, 0].item() == 1.0
        assert masked[0, 1].item() == 2.0
        assert masked[0, 3].item() == 4.0
        assert masked[0, 4].item() == 5.0

    def test_multiple_invalid_indices(self, masking):
        """Multiple invalid indices should all be -inf."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        masked = masking(logits, [[0, 3, 4]])
        assert torch.isinf(masked[0, 0]) and masked[0, 0] < 0
        assert torch.isinf(masked[0, 3]) and masked[0, 3] < 0
        assert torch.isinf(masked[0, 4]) and masked[0, 4] < 0
        # Valid indices unchanged
        assert masked[0, 1].item() == 2.0
        assert masked[0, 2].item() == 3.0

    def test_batch_masking(self, masking):
        """Each item in a batch receives its own set of invalid indices."""
        logits = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ])
        masked = masking(logits, [[1], [3]])
        # First batch item: index 1 masked
        assert torch.isinf(masked[0, 1]) and masked[0, 1] < 0
        assert masked[0, 0].item() == 1.0
        # Second batch item: index 3 masked
        assert torch.isinf(masked[1, 3]) and masked[1, 3] < 0
        assert masked[1, 4].item() == 1.0

    def test_original_logits_not_modified(self, masking):
        """Masking should not modify the original logits tensor."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        original = logits.clone()
        masking(logits, [[2]])
        assert torch.allclose(logits, original)

    def test_all_masked_gives_neg_inf_softmax_nan(self, masking):
        """If all logits are -inf, softmax should produce NaN or undefined."""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        masked = masking(logits, [[0, 1, 2, 3, 4]])
        assert all(torch.isinf(masked[0, i]) and masked[0, i] < 0 for i in range(5))
