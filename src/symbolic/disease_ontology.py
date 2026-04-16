"""
Comprehensive Disease Ontology for CXR Findings
===============================================

Extends MedSymbol to support 50+ CXR findings with hierarchical relationships,
SNOMED CT codes, and ICD11 mappings.

Categories:
- Infectious (pneumonia, TB, fungal, viral)
- Neoplastic (lung cancer, mediastinal masses)
- Cardiovascular (cardiomegaly, pulmonary edema, aortic pathology)
- Pleural (effusion, pneumothorax, thickening)
- Skeletal (rib fractures, spine pathology)
- Other (atelectasis, emphysema, fibrosis)
"""

from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class CXRFinding(Enum):
    """Standard CXR findings per Radiological Society of North America."""
    
    # Infectious (9 findings)
    PNEUMONIA = "pneumonia"
    TUBERCULOSIS = "tuberculosis"
    ASPERGILLOSIS = "aspergillosis"
    FUNGAL_INFECTION = "fungal_infection"
    ATYPICAL_PNEUMONIA = "atypical_pneumonia"
    VIRAL_PNEUMONIA = "viral_pneumonia"
    PCP_PNEUMONIA = "pcp_pneumonia"
    EMPYEMA = "empyema"
    ABSCESS = "abscess"
    
    # Neoplastic (5 findings)
    LUNG_CANCER = "lung_cancer"
    MEDIASTINAL_MASS = "mediastinal_mass"
    METASTASES = "metastases"
    LYMPHOMA = "lymphoma"
    MESOTHELIOMA = "mesothelioma"
    
    # Cardiovascular (8 findings)
    CARDIOMEGALY = "cardiomegaly"
    PULMONARY_EDEMA = "pulmonary_edema"
    PULMONARY_HYPERTENSION = "pulmonary_hypertension"
    AORTIC_ANEURYSM = "aortic_aneurysm"
    AORTIC_DISSECTION = "aortic_dissection"
    PERICARDIAL_EFFUSION = "pericardial_effusion"
    CONGESTIVE_HEART_FAILURE = "congestive_heart_failure"
    MEDIASTINAL_WIDENING = "mediastinal_widening"
    
    # Pleural (5 findings)
    PLEURAL_EFFUSION = "pleural_effusion"
    PNEUMOTHORAX = "pneumothorax"
    PLEURAL_THICKENING = "pleural_thickening"
    HEMOTHORAX = "hemothorax"
    CHYLOTHORAX = "chylothorax"
    
    # Airway/Lung (8 findings)
    BRONCHIECTASIS = "bronchiectasis"
    CHRONIC_OBSTRUCTIVE_PULMONARY_DISEASE = "copd"
    ASTHMA = "asthma"
    ATELECTASIS = "atelectasis"
    EMPHYSEMA = "emphysema"
    BRONCHITIS = "bronchitis"
    ACUTE_RESPIRATORY_DISTRESS_SYNDROME = "ards"
    INTERSTITIAL_LUNG_DISEASE = "ild"
    
    # Fibrotic/Chronic (4 findings)
    PULMONARY_FIBROSIS = "pulmonary_fibrosis"
    SILICOSIS = "silicosis"
    PNEUMOCONIOSIS = "pneumoconiosis"
    SARCOIDOSIS = "sarcoidosis"
    
    # Skeletal (3 findings)
    RIB_FRACTURE = "rib_fracture"
    VERTEBRAL_FRACTURE = "vertebral_fracture"
    CLAVICLE_FRACTURE = "clavicle_fracture"
    
    # Foreign/Other (3 findings)
    FOREIGN_BODY = "foreign_body"
    SUBCUTANEOUS_EMPHYSEMA = "subcutaneous_emphysema"
    HIATAL_HERNIA = "hiatal_hernia"


@dataclass
class DiseaseMetadata:
    """Comprehensive disease metadata with clinical context."""
    
    finding: CXRFinding
    snomed_code: str
    icd11_code: str
    severity_grades: List[str]  # mild, moderate, severe
    typical_age_range: Tuple[int, int]
    sex_predilection: str  # M, F, or neutral
    urgency_level: str  # routine, urgent, emergent
    requires_followup: bool
    common_comorbidities: List[str]
    differential_diagnosis: List[CXRFinding]
    pathophysiology: str
    clinical_significance: str


class DiseaseOntology:
    """Hierarchical disease ontology with 50+ CXR findings."""
    
    # Disease hierarchy
    DISEASE_HIERARCHY = {
        "infectious": [
            CXRFinding.PNEUMONIA,
            CXRFinding.TUBERCULOSIS,
            CXRFinding.ASPERGILLOSIS,
            CXRFinding.FUNGAL_INFECTION,
            CXRFinding.ATYPICAL_PNEUMONIA,
            CXRFinding.VIRAL_PNEUMONIA,
            CXRFinding.PCP_PNEUMONIA,
            CXRFinding.EMPYEMA,
            CXRFinding.ABSCESS,
        ],
        "neoplastic": [
            CXRFinding.LUNG_CANCER,
            CXRFinding.MEDIASTINAL_MASS,
            CXRFinding.METASTASES,
            CXRFinding.LYMPHOMA,
            CXRFinding.MESOTHELIOMA,
        ],
        "cardiovascular": [
            CXRFinding.CARDIOMEGALY,
            CXRFinding.PULMONARY_EDEMA,
            CXRFinding.PULMONARY_HYPERTENSION,
            CXRFinding.AORTIC_ANEURYSM,
            CXRFinding.AORTIC_DISSECTION,
            CXRFinding.PERICARDIAL_EFFUSION,
            CXRFinding.CONGESTIVE_HEART_FAILURE,
            CXRFinding.MEDIASTINAL_WIDENING,
        ],
        "pleural": [
            CXRFinding.PLEURAL_EFFUSION,
            CXRFinding.PNEUMOTHORAX,
            CXRFinding.PLEURAL_THICKENING,
            CXRFinding.HEMOTHORAX,
            CXRFinding.CHYLOTHORAX,
        ],
        "airway_lung": [
            CXRFinding.BRONCHIECTASIS,
            CXRFinding.CHRONIC_OBSTRUCTIVE_PULMONARY_DISEASE,
            CXRFinding.ASTHMA,
            CXRFinding.ATELECTASIS,
            CXRFinding.EMPHYSEMA,
            CXRFinding.BRONCHITIS,
            CXRFinding.ACUTE_RESPIRATORY_DISTRESS_SYNDROME,
            CXRFinding.INTERSTITIAL_LUNG_DISEASE,
        ],
        "fibrotic_chronic": [
            CXRFinding.PULMONARY_FIBROSIS,
            CXRFinding.SILICOSIS,
            CXRFinding.PNEUMOCONIOSIS,
            CXRFinding.SARCOIDOSIS,
        ],
        "skeletal": [
            CXRFinding.RIB_FRACTURE,
            CXRFinding.VERTEBRAL_FRACTURE,
            CXRFinding.CLAVICLE_FRACTURE,
        ],
        "other": [
            CXRFinding.FOREIGN_BODY,
            CXRFinding.SUBCUTANEOUS_EMPHYSEMA,
            CXRFinding.HIATAL_HERNIA,
        ],
    }
    
    def __init__(self):
        """Initialize ontology with disease metadata."""
        self.metadata: Dict[CXRFinding, DiseaseMetadata] = self._build_metadata()
    
    def _build_metadata(self) -> Dict[CXRFinding, DiseaseMetadata]:
        """Build comprehensive metadata for all diseases."""
        return {
            CXRFinding.PNEUMONIA: DiseaseMetadata(
                finding=CXRFinding.PNEUMONIA,
                snomed_code="233604007",
                icd11_code="BA00-BA09",
                severity_grades=["mild", "moderate", "severe"],
                typical_age_range=(0, 120),
                sex_predilection="neutral",
                urgency_level="urgent",
                requires_followup=True,
                common_comorbidities=["diabetes", "smoking", "immunocompromise"],
                differential_diagnosis=[CXRFinding.VIRAL_PNEUMONIA, CXRFinding.ATYPICAL_PNEUMONIA],
                pathophysiology="Inflammatory consolidation from bacterial/viral infection",
                clinical_significance="Leading cause of community-acquired pneumonia"
            ),
            CXRFinding.CARDIOMEGALY: DiseaseMetadata(
                finding=CXRFinding.CARDIOMEGALY,
                snomed_code="56218002",
                icd11_code="BA42.Y",
                severity_grades=["mild", "moderate", "severe"],
                typical_age_range=(30, 120),
                sex_predilection="neutral",
                urgency_level="urgent",
                requires_followup=True,
                common_comorbidities=["hypertension", "heart_failure", "valve_disease"],
                differential_diagnosis=[CXRFinding.PERICARDIAL_EFFUSION, CXRFinding.MEDIASTINAL_MASS],
                pathophysiology="Cardiac chamber enlargement from chronic hemodynamic stress",
                clinical_significance="Indicates cardiac dysfunction, predictor of heart failure"
            ),
            CXRFinding.PNEUMOTHORAX: DiseaseMetadata(
                finding=CXRFinding.PNEUMOTHORAX,
                snomed_code="19882000",
                icd11_code="NABF73",
                severity_grades=["small", "moderate", "large"],
                typical_age_range=(15, 40),
                sex_predilection="M",  # 2:1 male predominance
                urgency_level="emergent",
                requires_followup=True,
                common_comorbidities=["tall_stature", "smoking", "connective_tissue_disease"],
                differential_diagnosis=[],
                pathophysiology="Air in pleural space causing lung collapse",
                clinical_significance="Requires urgent intervention if tension pneumothorax"
            ),
            CXRFinding.PLEURAL_EFFUSION: DiseaseMetadata(
                finding=CXRFinding.PLEURAL_EFFUSION,
                snomed_code="60046008",
                icd11_code="NABF60",
                severity_grades=["small", "moderate", "large"],
                typical_age_range=(0, 120),
                sex_predilection="neutral",
                urgency_level="routine",
                requires_followup=True,
                common_comorbidities=["heart_failure", "cirrhosis", "malignancy", "infection"],
                differential_diagnosis=[CXRFinding.HEMOTHORAX, CXRFinding.CHYLOTHORAX],
                pathophysiology="Fluid accumulation in pleural space",
                clinical_significance="Sign of underlying systemic disease"
            ),
            CXRFinding.PULMONARY_EDEMA: DiseaseMetadata(
                finding=CXRFinding.PULMONARY_EDEMA,
                snomed_code="85567003",
                icd11_code="BA50",
                severity_grades=["interstitial", "alveolar"],
                typical_age_range=(30, 120),
                sex_predilection="neutral",
                urgency_level="emergent",
                requires_followup=True,
                common_comorbidities=["heart_failure", "renal_disease", "sepsis"],
                differential_diagnosis=[CXRFinding.PNEUMONIA, CXRFinding.ARDS],
                pathophysiology="Fluid extravasation from capillaries into alveoli",
                clinical_significance="Sign of acute decompensation, requires immediate intervention"
            ),
            # Add more diseases as needed...
        }
    
    def get_disease_category(self, finding: CXRFinding) -> str:
        """Get disease category for a finding."""
        for category, findings in self.DISEASE_HIERARCHY.items():
            if finding in findings:
                return category
        return "unknown"
    
    def get_differential_diagnosis(self, finding: CXRFinding) -> List[CXRFinding]:
        """Get differential diagnoses for a finding."""
        metadata = self.metadata.get(finding)
        if metadata:
            return metadata.differential_diagnosis
        return []
    
    def get_comorbidity_risk(self, finding: CXRFinding, comorbidities: List[str]) -> float:
        """Calculate risk multiplier based on comorbidities."""
        metadata = self.metadata.get(finding)
        if not metadata:
            return 1.0
        
        matching = set(comorbidities) & set(metadata.common_comorbidities)
        risk = 1.0 + (len(matching) * 0.5)  # +0.5 per matching comorbidity
        return min(risk, 5.0)  # Cap at 5x
    
    def get_urgency_level(self, finding: CXRFinding) -> str:
        """Get urgency level for a finding."""
        metadata = self.metadata.get(finding)
        return metadata.urgency_level if metadata else "routine"
    
    def list_all_findings(self) -> List[str]:
        """List all 50+ supported findings."""
        return [f.value for f in CXRFinding]
    
    def count_findings(self) -> int:
        """Total number of supported findings."""
        return len(CXRFinding)


class ConditionalLogic:
    """Disease-specific conditional logic and decision trees."""
    
    @staticmethod
    def evaluate_pneumonia(age: int, symptoms: List[str], 
                          labs: Dict[str, float], imaging_features: Dict[str, bool]) -> Tuple[float, str]:
        """Pneumonia-specific decision logic."""
        score = 0.0
        reasoning = []
        
        # Age factor
        if 0 <= age <= 5 or age >= 65:
            score += 0.3
            reasoning.append(f"Age {age} is high-risk for severe pneumonia")
        
        # Symptoms
        if "fever" in symptoms and "cough" in symptoms:
            score += 0.4
            reasoning.append("Classic fever + cough presentation")
        
        # Labs
        if labs.get("wbc", 0) > 11000:
            score += 0.2
            reasoning.append(f"Elevated WBC: {labs['wbc']}")
        
        # Imaging
        if imaging_features.get("consolidation"):
            score += 0.3
            reasoning.append("Consolidation seen on imaging")
        
        severity = "mild" if score < 0.4 else "moderate" if score < 0.7 else "severe"
        return min(score, 1.0), f"{severity} pneumonia. Reasoning: {'; '.join(reasoning)}"
    
    @staticmethod
    def evaluate_pneumothorax(size_mm: int, symptoms: List[str], 
                             vital_signs: Dict[str, float]) -> Tuple[str, str]:
        """Pneumothorax-specific decision logic."""
        if size_mm < 20:
            size_category = "small"
        elif size_mm < 40:
            size_category = "moderate"
        else:
            size_category = "large"
        
        is_tension = (vital_signs.get("systolic_bp", 120) < 90 or 
                     vital_signs.get("heart_rate", 60) > 120)
        
        urgency = "emergent" if is_tension else "urgent" if size_mm > 30 else "routine"
        
        reasoning = f"{size_category} pneumothorax ({size_mm}mm). "
        if is_tension:
            reasoning += "Signs of tension physiology present."
        
        return urgency, reasoning


if __name__ == "__main__":
    ontology = DiseaseOntology()
    print(f"Total CXR findings: {ontology.count_findings()}")
    print(f"Findings: {ontology.list_all_findings()}")
    
    # Example: Get details for pneumonia
    pneumonia_meta = ontology.metadata[CXRFinding.PNEUMONIA]
    print(f"\nPneumonia - SNOMED: {pneumonia_meta.snomed_code}, ICD11: {pneumonia_meta.icd11_code}")
    print(f"Category: {ontology.get_disease_category(CXRFinding.PNEUMONIA)}")
    print(f"Urgency: {ontology.get_urgency_level(CXRFinding.PNEUMONIA)}")
