"""
Unit tests for SymbolicVerifier and ProofCertificateGenerator
(src/symbolic/verifier.py and src/symbolic/proofs.py).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from symbolic.constraints import OntologyConstraints
from symbolic.verifier import SymbolicVerifier
from symbolic.proofs import ProofCertificateGenerator


@pytest.fixture
def verifier():
    return SymbolicVerifier(OntologyConstraints())


@pytest.fixture
def proof_gen():
    return ProofCertificateGenerator()


class TestSymbolicVerifier:
    def test_verify_unknown_diagnosis_no_age(self, verifier):
        """An unknown diagnosis with no patient age data should return FAIL for age_validity."""
        results = verifier.verify_patient("unknown_disease", {})
        # age_validity starts as FAIL when no age is provided
        assert results["age_validity"] == "FAIL"

    def test_verify_valid_age_for_myocardial_infarction(self, verifier):
        """Adult patient (age 50) with myocardial_infarction should pass age validity."""
        results = verifier.verify_patient("myocardial_infarction", {"age": 50})
        assert results["age_validity"] == "PASS"

    def test_verify_invalid_age_for_myocardial_infarction(self, verifier):
        """Child patient (age 10) with myocardial_infarction should fail age validity."""
        results = verifier.verify_patient("myocardial_infarction", {"age": 10})
        assert results["age_validity"] == "FAIL"

    def test_verify_valid_age_for_acute_bronchiolitis(self, verifier):
        """Infant (age 1) with acute_bronchiolitis should pass age validity."""
        results = verifier.verify_patient("acute_bronchiolitis", {"age": 1})
        assert results["age_validity"] == "PASS"

    def test_verify_invalid_age_for_acute_bronchiolitis(self, verifier):
        """Adult (age 30) with acute_bronchiolitis should fail age validity."""
        results = verifier.verify_patient("acute_bronchiolitis", {"age": 30})
        assert results["age_validity"] == "FAIL"

    def test_verify_result_keys(self, verifier):
        """Verification results must include all five check keys."""
        results = verifier.verify_patient("myocardial_infarction", {"age": 50})
        expected_keys = {
            "age_validity",
            "symptom_consistency",
            "lab_consistency",
            "contraindications",
            "hierarchical_consistency",
        }
        assert set(results.keys()) == expected_keys

    def test_is_fully_consistent_all_pass(self, verifier):
        """All PASS values should yield True from is_fully_consistent."""
        all_pass = {k: "PASS" for k in ["age_validity", "symptom_consistency",
                                          "lab_consistency", "contraindications",
                                          "hierarchical_consistency"]}
        assert verifier.is_fully_consistent(all_pass) is True

    def test_is_fully_consistent_one_fail(self, verifier):
        """A single FAIL should yield False from is_fully_consistent."""
        mixed = {
            "age_validity": "FAIL",
            "symptom_consistency": "PASS",
            "lab_consistency": "PASS",
            "contraindications": "PASS",
            "hierarchical_consistency": "PASS",
        }
        assert verifier.is_fully_consistent(mixed) is False

    def test_placeholder_checks_default_pass(self, verifier):
        """Non-age checks default to PASS as placeholders."""
        results = verifier.verify_patient("myocardial_infarction", {"age": 50})
        assert results["symptom_consistency"] == "PASS"
        assert results["lab_consistency"] == "PASS"
        assert results["contraindications"] == "PASS"
        assert results["hierarchical_consistency"] == "PASS"


class TestProofCertificateGenerator:
    def test_generate_certificate_sat(self, proof_gen):
        """SAT result should produce z3_result SAT."""
        reasons = {"age_validity": "PASS"}
        cert = proof_gen.generate_certificate("p001", "pneumonia", True, reasons)
        assert cert["z3_result"] == "SAT"
        assert cert["diagnosis"] == "pneumonia"
        assert cert["id"] == "p001"
        assert cert["format"] == "SMT-LIB"
        assert cert["verification_checks"] == reasons

    def test_generate_certificate_unsat(self, proof_gen):
        """UNSAT result should produce z3_result UNSAT."""
        reasons = {"age_validity": "FAIL"}
        cert = proof_gen.generate_certificate("p002", "preeclampsia", False, reasons)
        assert cert["z3_result"] == "UNSAT"

    def test_certificate_verification_url(self, proof_gen):
        """Certificate should include a verification URL with the identifier."""
        cert = proof_gen.generate_certificate("abc123", "pneumonia", True, {})
        assert "abc123" in cert["verification_url"]
