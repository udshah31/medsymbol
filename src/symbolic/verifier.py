import json
from z3 import Solver, Int, String, Bool, Real, And, Or, Not, Implies, sat, unsat
from .constraints import OntologyConstraints

class SymbolicVerifier:
    """
    Module 5: Symbolic Verifier
    5 consistency checks using Z3 SMT solver:
    1. Age validity (pediatric/adult/geriatric constraints)
    2. Symptom consistency (expected vs actual)
    3. Lab value consistency (WBC, O2 sat, etc.)
    4. Contraindication check (drug interactions, immunocompromised)
    5. Hierarchical consistency (SNOMED-CT relationships)
    """
    def __init__(self, constraints: OntologyConstraints):
        self.constraints = constraints
        self.solver = Solver()

    def _build_age_constraints(self, diagnosis: str, age_val: int) -> bool:
        """
        Creates Z3 assertions for age constraints and solves.
        Returns: True if satisfiable (valid path).
        """
        age = Int('age')
        self.solver.push()  # Add a new scope level
        
        # Add patient fact
        self.solver.add(age == age_val)
        
        # Get ontology rule constraints
        rule = self.constraints.rules.get(diagnosis, {})
        min_age = rule.get("age_min", 0)
        max_age = rule.get("age_max", 150)
        
        # Add rule assertions
        self.solver.add(age >= min_age)
        self.solver.add(age <= max_age)
        
        # Check satisfiability
        is_sat = self.solver.check() == sat
        self.solver.pop()  # Remove assertions
        
        return is_sat
    
    def _check_symptom_consistency(self, diagnosis: str, symptoms: list) -> bool:
        """
        Check symptom consistency using Z3 string matching.
        Returns: True if symptoms support diagnosis.
        """
        if not symptoms:
            return True  # No symptoms reported is not a failure
        
        # Use constraint checker for symptom validation
        return self.constraints.check_symptom_consistency(diagnosis, symptoms)
    
    def _check_lab_consistency(self, diagnosis: str, lab_values: dict) -> bool:
        """
        Check laboratory value consistency using Z3 real constraints.
        Returns: True if lab values consistent with diagnosis.
        """
        self.solver.push()
        
        # Create Z3 real variables for each lab value
        z3_labs = {}
        for lab_name, value in lab_values.items():
            z3_labs[lab_name] = Real(lab_name)
            self.solver.add(z3_labs[lab_name] == float(value))
        
        # Apply range constraints from ontology
        for lab_name in z3_labs:
            if lab_name in self.constraints.lab_ranges:
                min_val, max_val = self.constraints.lab_ranges[lab_name]
                # Allow some flexibility (±30%) for pathological states
                self.solver.add(z3_labs[lab_name] >= min_val * 0.7)
                self.solver.add(z3_labs[lab_name] <= max_val * 1.3)
        
        is_sat = self.solver.check() == sat
        self.solver.pop()
        
        return is_sat
    
    def _check_contraindications(self, diagnosis: str, patient_data: dict) -> bool:
        """
        Check for contraindications using Z3 boolean logic.
        Returns: True if no contraindications found.
        """
        comorbidities = patient_data.get("comorbidities", [])
        return self.constraints.check_contraindications(diagnosis, comorbidities)
    
    def _check_hierarchical_consistency(self, diagnosis: str, other_diagnoses: list) -> bool:
        """
        Check hierarchical consistency with other diagnoses.
        Returns: True if diagnosis hierarchy is consistent.
        """
        for other_dx in other_diagnoses:
            if not self.constraints.check_hierarchical_consistency(diagnosis, other_dx):
                return False
        
        return True

    def verify_patient(self, diagnosis: str, patient_data: dict, 
                      lab_values: dict = None, symptoms: list = None,
                      other_diagnoses: list = None) -> dict:
        """
        Runs 5-check symbolic verifier for a specific diagnosis.
        
        Args:
            diagnosis: Disease diagnosis to verify
            patient_data: Dict with 'age', 'sex', 'comorbidities'
            lab_values: Dict mapping lab names to values
            symptoms: List of reported symptoms
            other_diagnoses: List of other diagnoses to check consistency with
        
        Returns: Dictionary of PASS/FAIL statuses for each check
        """
        results = {
            "age_validity": "FAIL",
            "symptom_consistency": "PASS",
            "lab_consistency": "PASS",
            "contraindications": "PASS",
            "hierarchical_consistency": "PASS"
        }
        
        # Check 1: Age validity
        if "age" in patient_data:
            if self._build_age_constraints(diagnosis, patient_data["age"]):
                results["age_validity"] = "PASS"
        else:
            results["age_validity"] = "PASS"  # No age data, pass
        
        # Check 2: Symptom consistency
        if symptoms:
            if self._check_symptom_consistency(diagnosis, symptoms):
                results["symptom_consistency"] = "PASS"
            else:
                results["symptom_consistency"] = "FAIL"
        
        # Check 3: Lab consistency
        if lab_values:
            if self._check_lab_consistency(diagnosis, lab_values):
                results["lab_consistency"] = "PASS"
            else:
                results["lab_consistency"] = "FAIL"
        
        # Check 4: Contraindications
        if self._check_contraindications(diagnosis, patient_data):
            results["contraindications"] = "PASS"
        else:
            results["contraindications"] = "FAIL"
        
        # Check 5: Hierarchical consistency
        if other_diagnoses:
            if self._check_hierarchical_consistency(diagnosis, other_diagnoses):
                results["hierarchical_consistency"] = "PASS"
            else:
                results["hierarchical_consistency"] = "FAIL"
        
        return results

    def is_fully_consistent(self, verification_results: dict) -> bool:
        """
        Returns True if all checks passed.
        """
        return all(res == "PASS" for res in verification_results.values())
    
    def verify_with_risk_stratification(self, diagnosis: str, patient_data: dict,
                                        lab_values: dict = None, symptoms: list = None,
                                        other_diagnoses: list = None) -> dict:
        """
        Enhanced verification with risk stratification and confidence scores.
        
        Returns: Extended results with:
        - All 5 verification checks (PASS/FAIL)
        - Risk stratification (low/intermediate/high)
        - Confidence score (0-1)
        - Detailed explanations
        """
        # Run standard 5 checks
        results = self.verify_patient(diagnosis, patient_data, lab_values, symptoms, other_diagnoses)
        
        # Add sex-based check
        sex = patient_data.get("sex")
        if sex:
            if self.constraints.check_sex_constraint(diagnosis, sex):
                results["sex_consistency"] = "PASS"
            else:
                results["sex_consistency"] = "WARN"  # Warning, not fail
        
        # Add comorbidity risk scoring
        comorbidities = patient_data.get("comorbidities", [])
        age = patient_data.get("age")
        comorbidity_multiplier = self.constraints.calculate_comorbidity_risk(
            diagnosis, comorbidities, age
        )
        results["comorbidity_risk_multiplier"] = comorbidity_multiplier
        
        # Add symptom matching details
        if symptoms:
            sym_compatible, sym_score = self.constraints.match_symptoms(diagnosis, symptoms)
            results["symptom_match_score"] = sym_score
            results["symptom_compatible"] = sym_compatible
        
        # Add advanced lab checking
        if lab_values:
            lab_pass, lab_details = self.constraints.check_lab_consistency_advanced(
                diagnosis, lab_values
            )
            results["lab_details"] = lab_details
        
        # Risk stratification
        results["risk_stratification"] = self.constraints.get_risk_stratification(
            diagnosis, patient_data
        )
        
        # Calculate overall confidence (0-1)
        pass_count = sum(1 for v in results.values() 
                        if isinstance(v, str) and v == "PASS")
        check_count = sum(1 for v in results.values() 
                         if isinstance(v, str) and v in ["PASS", "FAIL", "WARN"])
        
        if check_count > 0:
            results["confidence_score"] = pass_count / check_count
        else:
            results["confidence_score"] = 0.5
        
        results["fully_consistent"] = self.is_fully_consistent(results)
        
        return results
