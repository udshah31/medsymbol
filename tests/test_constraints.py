"""
Unit tests for OntologyConstraints (src/symbolic/constraints.py).
These tests cover age and sex constraint checks without heavy dependencies.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from symbolic.constraints import OntologyConstraints


@pytest.fixture
def constraints():
    return OntologyConstraints()


class TestAgeConstraints:
    def test_no_rule_returns_valid(self, constraints):
        """Diagnoses with no rules should always pass."""
        assert constraints.check_age_constraint("unknown_disease", 50) is True

    def test_acute_bronchiolitis_valid_age(self, constraints):
        """acute_bronchiolitis is constrained to age 0-2."""
        assert constraints.check_age_constraint("acute_bronchiolitis", 0) is True
        assert constraints.check_age_constraint("acute_bronchiolitis", 1) is True
        assert constraints.check_age_constraint("acute_bronchiolitis", 2) is True

    def test_acute_bronchiolitis_invalid_age(self, constraints):
        """Adults should fail acute_bronchiolitis age constraint."""
        assert constraints.check_age_constraint("acute_bronchiolitis", 3) is False
        assert constraints.check_age_constraint("acute_bronchiolitis", 30) is False

    def test_myocardial_infarction_valid_age(self, constraints):
        """myocardial_infarction requires age >= 18."""
        assert constraints.check_age_constraint("myocardial_infarction", 18) is True
        assert constraints.check_age_constraint("myocardial_infarction", 65) is True

    def test_myocardial_infarction_invalid_age(self, constraints):
        """Children should fail myocardial_infarction age constraint."""
        assert constraints.check_age_constraint("myocardial_infarction", 17) is False
        assert constraints.check_age_constraint("myocardial_infarction", 0) is False

    def test_preeclampsia_has_no_age_constraint(self, constraints):
        """preeclampsia rule has sex constraint only; any age should pass age check."""
        assert constraints.check_age_constraint("preeclampsia", 25) is True
        assert constraints.check_age_constraint("preeclampsia", 80) is True


class TestSexConstraints:
    def test_no_rule_returns_valid(self, constraints):
        """Diagnoses with no rules should always pass."""
        assert constraints.check_sex_constraint("unknown_disease", "female") is True
        assert constraints.check_sex_constraint("unknown_disease", "male") is True

    def test_preeclampsia_female(self, constraints):
        """preeclampsia is female-only."""
        assert constraints.check_sex_constraint("preeclampsia", "female") is True
        assert constraints.check_sex_constraint("preeclampsia", "Female") is True

    def test_preeclampsia_male(self, constraints):
        """Males should fail preeclampsia sex constraint."""
        assert constraints.check_sex_constraint("preeclampsia", "male") is False

    def test_no_sex_constraint_diagnosis(self, constraints):
        """Diagnoses without a sex rule pass for any sex."""
        assert constraints.check_sex_constraint("acute_bronchiolitis", "male") is True
        assert constraints.check_sex_constraint("acute_bronchiolitis", "female") is True
        assert constraints.check_sex_constraint("myocardial_infarction", "male") is True
        assert constraints.check_sex_constraint("myocardial_infarction", "female") is True
