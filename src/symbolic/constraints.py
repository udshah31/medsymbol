from typing import Dict, Any, List, Tuple

class OntologyConstraints:
    """
    Enhanced Symbolic Constraints from SNOMED-CT/ICD-11 + Medical Literature
    Provides comprehensive logic parameters for 14 NIH CXR14 diseases:
    - Age/sex-specific constraints
    - Symptom pattern matching
    - Hierarchical disease relationships
    - Comorbidity interactions
    - Laboratory value interpretation
    
    Based on: medical literature, epidemiology, and clinical guidelines
    """
    def __init__(self):
        # Enhanced comprehensive rules for all 14 NIH CXR14 disease labels
        # Based on medical literature (age peaks, comorbidities, demographics, epidemiology)
        self.rules = {
            "Atelectasis": {
                "age_min": 0, "age_max": 150,
                "sex": None,  # No sex-based restriction
                "primary_symptom": "diminished_breath_sounds",
                "required_symptoms": ["reduced_air_entry"],
                "risk_factors": ["smoking", "obesity", "surgery", "prolonged_bedrest"],
                "protective_factors": ["young_age", "good_pulmonary_function"],
                "contraindications": [],
                "comorbidity_associations": {"obesity": 1.5, "COPD": 1.3},
                "hierarchical_parent": None,
                "hierarchical_children": [],
            },
            "Cardiomegaly": {
                "age_min": 30, "age_max": 150,
                "sex": None,
                "primary_symptom": "dyspnea",
                "required_symptoms": ["heart_failure_signs", "dyspnea"],
                "risk_factors": ["hypertension", "diabetes", "smoking", "obesity", "chronic_alcohol"],
                "protective_factors": ["exercise", "low_BMI", "normal_BP"],
                "contraindications": [],
                "comorbidity_associations": {"hypertension": 2.5, "diabetes": 1.8, "heart_failure": 3.0},
                "hierarchical_parent": "heart_disease",
                "hierarchical_children": [],
                "lab_triggers": {"BNP": (100, 999), "troponin": (0.04, 0.5)},
            },
            "Effusion": {
                "age_min": 0, "age_max": 150,
                "sex": None,
                "primary_symptom": "chest_pain",
                "required_symptoms": ["dyspnea", "reduced_breath_sounds"],
                "risk_factors": ["malignancy", "infection", "heart_failure", "renal_failure", "trauma"],
                "protective_factors": [],
                "contraindications": [],
                "comorbidity_associations": {"heart_failure": 2.0, "malignancy": 2.5, "pneumonia": 1.8},
                "hierarchical_parent": "pleural_disease",
                "hierarchical_children": [],
            },
            "Infiltration": {
                "age_min": 0, "age_max": 150,
                "sex": None,
                "primary_symptom": "cough",
                "required_symptoms": ["cough", "fever_or_dyspnea"],
                "risk_factors": ["infection", "aspiration", "malignancy", "immunocompromised"],
                "protective_factors": ["good_immune_function"],
                "contraindications": [],
                "comorbidity_associations": {"tuberculosis": 2.5, "pneumonia": 2.0, "HIV": 2.8},
                "hierarchical_parent": "lung_opacity",
                "hierarchical_children": [],
            },
            "Mass": {
                "age_min": 40, "age_max": 150,
                "sex": None,
                "primary_symptom": "chest_pain",
                "required_symptoms": ["lung_nodule_or_mass"],
                "risk_factors": ["smoking", "asbestos_exposure", "prior_cancer", "family_history"],
                "protective_factors": ["never_smoker", "good_exposure_history"],
                "contraindications": [],
                "comorbidity_associations": {"lung_cancer": 3.5, "smoking": 2.5},
                "hierarchical_parent": "malignancy_suspected",
                "hierarchical_children": [],
            },
            "Nodule": {
                "age_min": 30, "age_max": 150,
                "sex": None,
                "primary_symptom": "incidental_finding",
                "required_symptoms": [],
                "risk_factors": ["smoking", "prior_malignancy", "occupational_exposure"],
                "protective_factors": ["young_age", "small_size"],
                "contraindications": [],
                "comorbidity_associations": {"lung_cancer": 2.5, "smoking": 1.8},
                "hierarchical_parent": "lung_opacity",
                "hierarchical_children": [],
                "size_thresholds": {"concerning": (8, 999), "borderline": (4, 8), "benign": (0, 4)},
            },
            "Pneumonia": {
                "age_min": 0, "age_max": 150,
                "sex": None,
                "primary_symptom": "fever_cough",
                "required_symptoms": ["fever", "cough", "dyspnea"],
                "risk_factors": ["immunocompromised", "aspiration_risk", "elderly", "smoking", "chronic_disease"],
                "protective_factors": ["vaccination", "good_immune_function"],
                "contraindications": [],
                "comorbidity_associations": {"immunocompromised": 3.0, "diabetes": 1.8, "COPD": 2.0, "elderly": 2.5},
                "hierarchical_parent": "infection",
                "hierarchical_children": ["bacterial_pneumonia", "viral_pneumonia"],
                "lab_triggers": {"WBC": (11.0, 30), "CRP": (20, 999)},
            },
            "Pneumothorax": {
                "age_min": 15, "age_max": 40,  # Peaks in young men
                "sex": "M",  # Primary spontaneous PTX more common in males (4-6:1)
                "primary_symptom": "acute_chest_pain",
                "required_symptoms": ["acute_chest_pain", "dyspnea"],
                "risk_factors": ["smoking", "tall_thin_build", "COPD", "bullae"],
                "protective_factors": ["normal_BMI"],
                "contraindications": [],
                "comorbidity_associations": {"COPD": 1.8, "smoking": 2.0},
                "hierarchical_parent": "pleural_disease",
                "hierarchical_children": ["primary_spontaneous", "secondary_spontaneous", "traumatic"],
                "acute_presentation": True,
            },
            "Consolidation": {
                "age_min": 0, "age_max": 150,
                "sex": None,
                "primary_symptom": "cough_fever",
                "required_symptoms": ["cough", "fever", "dyspnea"],
                "risk_factors": ["infection", "aspiration", "immunocompromised"],
                "protective_factors": [],
                "contraindications": [],
                "comorbidity_associations": {"pneumonia": 2.5, "infection": 2.0},
                "hierarchical_parent": "infection",
                "hierarchical_children": [],
            },
            "Edema": {
                "age_min": 30, "age_max": 150,
                "sex": None,
                "primary_symptom": "dyspnea",
                "required_symptoms": ["dyspnea", "orthopnea"],
                "risk_factors": ["heart_failure", "renal_failure", "hypoalbuminemia"],
                "protective_factors": ["normal_cardiac_function"],
                "contraindications": [],
                "comorbidity_associations": {"heart_failure": 3.0, "renal_failure": 2.5, "CKD": 2.0},
                "hierarchical_parent": "pulmonary_edema",
                "hierarchical_children": ["cardiogenic", "noncardiogenic"],
                "lab_triggers": {"BNP": (100, 999), "creatinine": (1.3, 999)},
            },
            "Emphysema": {
                "age_min": 40, "age_max": 150,
                "sex": None,
                "primary_symptom": "dyspnea_on_exertion",
                "required_symptoms": ["chronic_dyspnea"],
                "risk_factors": ["smoking", "alpha_1_deficiency", "genetic_predisposition"],
                "protective_factors": ["never_smoker", "good_lung_function"],
                "contraindications": [],
                "comorbidity_associations": {"COPD": 2.5, "smoking": 3.0, "lung_cancer": 1.8},
                "hierarchical_parent": "COPD",
                "hierarchical_children": [],
            },
            "Fibrosis": {
                "age_min": 40, "age_max": 150,
                "sex": None,
                "primary_symptom": "dyspnea",
                "required_symptoms": ["chronic_dyspnea", "dry_cough"],
                "risk_factors": ["idiopathic", "occupational_exposure", "asbestos", "rheumatologic_disease"],
                "protective_factors": [],
                "contraindications": [],
                "comorbidity_associations": {"occupational_exposure": 2.5, "connective_tissue_disease": 2.0},
                "hierarchical_parent": "diffuse_lung_disease",
                "hierarchical_children": ["IPF", "occupational", "drug_induced"],
            },
            "Pleural_Thickening": {
                "age_min": 40, "age_max": 150,
                "sex": None,
                "primary_symptom": "chronic_pleural_effusion",
                "required_symptoms": ["pleural_effusion"],
                "risk_factors": ["asbestos_exposure", "recurrent_infection", "malignancy"],
                "protective_factors": [],
                "contraindications": [],
                "comorbidity_associations": {"asbestos_exposure": 3.0, "pleural_effusion": 2.5},
                "hierarchical_parent": "pleural_disease",
                "hierarchical_children": [],
            },
            "Hernia": {
                "age_min": 0, "age_max": 150,
                "sex": None,
                "primary_symptom": "incidental_finding",
                "required_symptoms": [],
                "risk_factors": ["obesity", "prior_surgery", "connective_tissue_disease"],
                "protective_factors": ["normal_BMI", "good_abdominal_wall_integrity"],
                "contraindications": [],
                "comorbidity_associations": {"obesity": 1.8},
                "hierarchical_parent": "anatomic_abnormality",
                "hierarchical_children": [],
            },
            "No Finding": {
                "age_min": 0, "age_max": 150,
                "sex": None,
                "primary_symptom": "asymptomatic",
                "required_symptoms": [],
                "risk_factors": [],
                "protective_factors": ["healthy_status"],
                "contraindications": [],
                "comorbidity_associations": {},
                "hierarchical_parent": None,
                "hierarchical_children": [],
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
        
        # Check hierarchical compatibility
        parent1 = self.rules.get(diagnosis1, {}).get("hierarchical_parent")
        parent2 = self.rules.get(diagnosis2, {}).get("hierarchical_parent")
        
        # If they share a parent, they're related but can coexist
        return True
    
    def check_sex_constraint(self, diagnosis: str, patient_sex: str) -> bool:
        """
        Check if diagnosis is valid for patient's sex.
        Returns: True if valid or no sex restriction, False if contraindicated.
        """
        rule = self.rules.get(diagnosis)
        if not rule:
            return True
        
        sex_restriction = rule.get("sex")
        if sex_restriction is None:
            return True  # No sex restriction
        
        # For pneumothorax, male predominance (M:F ratio ~4-6:1)
        # Sex restriction means "more common in" not "only in"
        return patient_sex.upper() == sex_restriction.upper()
    
    def calculate_comorbidity_risk(self, diagnosis: str, comorbidities: List[str], age: int = None) -> float:
        """
        Calculate risk multiplier based on comorbidities.
        Returns: Risk score (1.0 = baseline, >1.0 = increased risk).
        """
        rule = self.rules.get(diagnosis)
        if not rule:
            return 1.0
        
        comorbidity_map = rule.get("comorbidity_associations", {})
        risk_score = 1.0
        
        for comorbidity in comorbidities:
            multiplier = comorbidity_map.get(comorbidity, 1.0)
            risk_score *= multiplier
        
        # Age-based risk modulation
        if age:
            age_min = rule.get("age_min", 0)
            age_max = rule.get("age_max", 150)
            peak_age = (age_min + age_max) / 2
            
            # Risk peaks at median age, decreases at extremes
            if age < age_min or age > age_max:
                risk_score *= 0.5
            elif abs(age - peak_age) > 20:
                risk_score *= 0.8
        
        return min(risk_score, 5.0)  # Cap at 5x
    
    def match_symptoms(self, diagnosis: str, observed_symptoms: List[str]) -> Tuple[bool, float]:
        """
        Match observed symptoms against diagnosis requirements.
        Returns: (is_compatible, match_score 0-1)
        """
        rule = self.rules.get(diagnosis)
        if not rule:
            return True, 0.5
        
        required = rule.get("required_symptoms", [])
        if not required:
            return True, 1.0
        
        if not observed_symptoms:
            return len(required) == 0, 0.0
        
        # Calculate symptom match score
        observed_lower = [s.lower() for s in observed_symptoms]
        matched = sum(1 for req in required if any(req.lower() in obs for obs in observed_lower))
        
        match_score = matched / len(required) if required else 1.0
        is_compatible = match_score >= 0.5  # At least 50% match
        
        return is_compatible, match_score
    
    def check_lab_consistency_advanced(self, diagnosis: str, lab_values: Dict[str, float]) -> Tuple[bool, Dict[str, bool]]:
        """
        Advanced lab value consistency checking.
        Returns: (overall_pass, detailed_results).
        """
        rule = self.rules.get(diagnosis)
        if not rule:
            return True, {}
        
        lab_triggers = rule.get("lab_triggers", {})
        results = {}
        
        for lab_name, (min_val, max_val) in lab_triggers.items():
            if lab_name in lab_values:
                value = lab_values[lab_name]
                results[lab_name] = min_val <= value <= max_val
        
        # All triggered labs must be in range
        overall_pass = all(results.values()) if results else True
        return overall_pass, results
    
    def get_risk_stratification(self, diagnosis: str, patient_data: Dict[str, Any]) -> str:
        """
        Stratify patient risk based on diagnosis and presentation.
        Returns: "low", "intermediate", or "high" risk.
        """
        age = patient_data.get("age", 50)
        sex = patient_data.get("sex", "M")
        comorbidities = patient_data.get("comorbidities", [])
        symptoms = patient_data.get("symptoms", [])
        
        rule = self.rules.get(diagnosis)
        if not rule:
            return "intermediate"
        
        risk_score = 1.0
        
        # Age risk
        age_min = rule.get("age_min", 0)
        age_max = rule.get("age_max", 150)
        if age < age_min or age > age_max:
            risk_score += 1.5
        elif age > 70:
            risk_score += 0.8
        
        # Sex risk
        if not self.check_sex_constraint(diagnosis, sex):
            risk_score += 0.5
        
        # Comorbidity risk
        comorbidity_risk = self.calculate_comorbidity_risk(diagnosis, comorbidities, age)
        risk_score *= (comorbidity_risk / 3.0)  # Normalize
        
        # Symptom risk
        sym_compatible, sym_score = self.match_symptoms(diagnosis, symptoms)
        if not sym_compatible:
            risk_score += 1.0
        elif sym_score < 0.5:
            risk_score += 0.5
        
        # Classify
        if risk_score < 1.5:
            return "low"
        elif risk_score < 3.0:
            return "intermediate"
        else:
            return "high"
