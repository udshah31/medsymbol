from typing import Dict, Any, List

class OntologyConstraints:
    """
    Simulates Age/Sex/Symptom rules from SNOMED-CT/ICD-11.
    Provides logic parameters for specific diagnosis constraints.
    
    NIH ChestX-ray14 Disease Cards:
    - Age ranges, sex restrictions, symptom patterns, contraindication flags
    """
    def __init__(self):
        # Comprehensive rules for all 14 NIH CXR14 disease labels
        # Based on medical literature (age peaks, comorbidities, demographics)
        self.rules = {
            "Atelectasis": {
                "age_min": 0,
                "age_max": 150,
                "primary_symptom": "diminished_breath_sounds",
                "risk_factors": ["smoking", "obesity", "surgery"],
                "contraindications": [],
            },
            "Cardiomegaly": {
                "age_min": 30,
                "age_max": 150,
                "primary_symptom": "dyspnea",
                "risk_factors": ["hypertension", "diabetes", "smoking"],
                "contraindications": [],
            },
            "Effusion": {
                "age_min": 0,
                "age_max": 150,
                "primary_symptom": "chest_pain",
                "risk_factors": ["malignancy", "infection", "heart_failure"],
                "contraindications": [],
            },
            "Infiltration": {
                "age_min": 0,
                "age_max": 150,
                "primary_symptom": "cough",
                "risk_factors": ["infection", "aspiration", "malignancy"],
                "contraindications": [],
            },
            "Mass": {
                "age_min": 40,
                "age_max": 150,
                "primary_symptom": "chest_pain",
                "risk_factors": ["smoking", "asbestos_exposure"],
                "contraindications": [],
            },
            "Nodule": {
                "age_min": 30,
                "age_max": 150,
                "primary_symptom": "incidental_finding",
                "risk_factors": ["smoking", "prior_malignancy"],
                "contraindications": [],
            },
            "Pneumonia": {
                "age_min": 0,
                "age_max": 150,
                "primary_symptom": "fever_cough",
                "risk_factors": ["immunocompromised", "aspiration_risk"],
                "contraindications": [],
            },
            "Pneumothorax": {
                "age_min": 15,
                "age_max": 40,  # Peaks in young men
                "sex": "M",  # Primary spontaneous PTX more common in males
                "primary_symptom": "acute_chest_pain",
                "risk_factors": ["smoking", "tall_thin_build"],
                "contraindications": [],
            },
            "Consolidation": {
                "age_min": 0,
                "age_max": 150,
                "primary_symptom": "cough_fever",
                "risk_factors": ["infection", "aspiration"],
                "contraindications": [],
            },
            "Edema": {
                "age_min": 30,
                "age_max": 150,
                "primary_symptom": "dyspnea",
                "risk_factors": ["heart_failure", "renal_failure"],
                "contraindications": [],
            },
            "Emphysema": {
                "age_min": 40,
                "age_max": 150,
                "primary_symptom": "dyspnea_on_exertion",
                "risk_factors": ["smoking", "alpha_1_deficiency"],
                "contraindications": [],
            },
            "Fibrosis": {
                "age_min": 40,
                "age_max": 150,
                "primary_symptom": "dyspnea",
                "risk_factors": ["idiopathic", "occupational_exposure"],
                "contraindications": [],
            },
            "Pleural_Thickening": {
                "age_min": 40,
                "age_max": 150,
                "primary_symptom": "chronic_pleural_effusion",
                "risk_factors": ["asbestos_exposure", "recurrent_infection"],
                "contraindications": [],
            },
            "Hernia": {
                "age_min": 0,
                "age_max": 150,
                "primary_symptom": "incidental_finding",
                "risk_factors": ["obesity", "prior_surgery"],
                "contraindications": [],
            },
            "No Finding": {
                "age_min": 0,
                "age_max": 150,
                "primary_symptom": "asymptomatic",
                "risk_factors": [],
                "contraindications": [],
            },
        }
        
        # Lab value normal ranges for consistency checking
        self.lab_ranges = {
            "WBC": (4.5, 11.0),           # K/uL
            "Hemoglobin": (13.5, 17.5),   # g/dL (males) / (12.0, 15.5) females
            "O2_saturation": (95, 100),   # %
            "CRP": (0, 10),               # mg/L
            "Creatinine": (0.7, 1.3),    # mg/dL
            "BNP": (0, 100),              # pg/mL
            "Troponin": (0, 0.04),        # ng/mL
            "D_dimer": (0, 0.5),          # ug/mL FEU
            "Lactate": (0.5, 2.0),        # mmol/L
            "pH": (7.35, 7.45),           # arterial
        }

    def check_age_constraint(self, diagnosis: str, age: int) -> bool:
        """
        Check if diagnosis is valid based on standard age bounds.
        Returns: True if valid, False if contraindicated.
        """
        rule = self.rules.get(diagnosis)
        if not rule:
            return True
        
        min_age = rule.get("age_min", 0)
        max_age = rule.get("age_max", 150)
        
        # True if age is within bounds
        return min_age <= age <= max_age

    def check_sex_constraint(self, diagnosis: str, sex: str) -> bool:
        """
        Check if diagnosis is valid for a particular biological sex.
        Returns: True if valid, False if contraindicated.
        """
        rule = self.rules.get(diagnosis)
        if not rule:
            return True
            
        allowed_sex = rule.get("sex")
        if not allowed_sex:
            return True
            
        return allowed_sex.lower() == sex.lower()
    
    def check_symptom_consistency(self, diagnosis: str, reported_symptoms: List[str]) -> bool:
        """
        Check if reported symptoms are consistent with diagnosis.
        Returns: True if symptoms match expected pattern.
        """
        rule = self.rules.get(diagnosis)
        if not rule:
            return True
        
        primary_symptom = rule.get("primary_symptom", "")
        if not primary_symptom:
            return True
        
        # Simple substring matching for symptom consistency
        reported_lower = ' '.join([s.lower() for s in reported_symptoms])
        return primary_symptom.replace('_', ' ') in reported_lower or len(reported_symptoms) == 0
    
    def check_lab_consistency(self, diagnosis: str, lab_values: Dict[str, float]) -> bool:
        """
        Check if lab values are consistent with diagnosis.
        Returns: True if labs are within expected ranges for diagnosis.
        
        Args:
            diagnosis: Disease name
            lab_values: Dict mapping lab name to value
        """
        # Most diseases don't have strict lab dependencies in CXR context
        # This is a placeholder for more sophisticated checking
        for lab_name, value in lab_values.items():
            if lab_name in self.lab_ranges:
                min_val, max_val = self.lab_ranges[lab_name]
                # Flag extreme outliers
                if value < min_val * 0.5 or value > max_val * 2.0:
                    return False
        
        return True
    
    def check_contraindications(self, diagnosis: str, comorbidities: List[str]) -> bool:
        """
        Check for contraindications based on comorbidities.
        Returns: True if no contraindications, False if contraindicated.
        """
        rule = self.rules.get(diagnosis)
        if not rule:
            return True
        
        contraindications = rule.get("contraindications", [])
        for condition in comorbidities:
            if condition.lower() in [c.lower() for c in contraindications]:
                return False
        
        return True
    
    def check_hierarchical_consistency(self, diagnosis1: str, diagnosis2: str) -> bool:
        """
        Check if two diagnoses can co-occur based on SNOMED-CT hierarchy.
        Returns: True if compatible, False if mutually exclusive.
        """
        # Simplified check: some findings are mutually exclusive
        mutually_exclusive = [
            {"Pneumothorax", "Consolidation"},  # Usually distinct
        ]
        
        pair = {diagnosis1, diagnosis2}
        for exclusive_set in mutually_exclusive:
            if pair == exclusive_set:
                return False
        
        # Most comorbidities are possible
        return True
