from typing import Dict, Any, List

class OntologyConstraints:
    """
    Simulates Age/Sex/Symptom rules from SNOMED-CT/ICD-11.
    Provides logic parameters for specific diagnosis constraints.
    """
    def __init__(self):
        # A simplified dictionary simulating ontology rules parsed from JSON.
        # Key: Diagnosis index or code. Value: dict of constraints.
        self.rules = {
            "acute_bronchiolitis": {
                "age_min": 0,
                "age_max": 2, # Typically affects children < 2 years
            },
            "myocardial_infarction": {
                "age_min": 18, # Very rare below 18
            },
            "preeclampsia": {
                "sex": "female",
            }
        }

    def check_age_constraint(self, diagnosis: str, age: int) -> bool:
        """
        Check if diagnosis is valid based on standard age bounds.
        Returns: True if valid, False if valid (contraindicated).
        """
        constraint = self.rules.get(diagnosis)
        if not constraint:
            return True
        
        min_age = constraint.get("age_min", 0)
        max_age = constraint.get("age_max", 150)
        
        # True if age is within bounds
        return min_age <= age <= max_age

    def check_sex_constraint(self, diagnosis: str, sex: str) -> bool:
        """
        Check if diagnosis is valid for a particular biological sex.
        Returns: True if valid, False if valid (contraindicated).
        """
        constraint = self.rules.get(diagnosis)
        if not constraint:
            return True
            
        allowed_sex = constraint.get("sex")
        if not allowed_sex:
            return True
            
        return allowed_sex.lower() == sex.lower()
