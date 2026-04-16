"""
Clinical Explanations & Recommendations
=======================================

Interpretable outputs and clinical decision support.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ClinicalExplanation:
    """Structured clinical explanation for prediction."""
    
    disease: str
    predicted_probability: float
    severity: str
    confidence: str  # high/moderate/low
    key_findings: List[str]
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    differential_diagnoses: List[Tuple[str, float]]  # (disease, probability)
    clinical_significance: str
    recommendations: List[str]
    limitations: List[str]
    follow_up: str = None
    disclaimer: str = None


class ClinicalRecommender:
    """Generate clinical recommendations from predictions."""
    
    DISEASE_PROTOCOLS = {
        "pneumonia": {
            "mild": {
                "recommendations": [
                    "Outpatient management with oral antibiotics",
                    "Follow-up CXR in 6-8 weeks",
                    "Patient education on infection prevention"
                ],
                "antibiotic_class": "beta-lactams (amoxicillin) or macrolides"
            },
            "moderate": {
                "recommendations": [
                    "Consider hospitalization for observation",
                    "IV antibiotics (broad-spectrum coverage)",
                    "Oxygen support if SpO2 < 92%",
                    "Reassess in 24-48 hours"
                ],
                "antibiotic_class": "broad-spectrum beta-lactams or fluoroquinolones"
            },
            "severe": {
                "recommendations": [
                    "URGENT: Hospital admission to ICU",
                    "IV broad-spectrum antibiotics",
                    "Aggressive supportive care",
                    "Consider mechanical ventilation",
                    "Infectious disease consultation"
                ],
                "antibiotic_class": "broad-spectrum beta-lactams + macrolides"
            }
        },
        "pneumothorax": {
            "small": {
                "recommendations": [
                    "Observation and serial imaging (24-48 hours)",
                    "Bed rest, supplemental O2",
                    "No invasive intervention needed",
                    "Follow-up CXR in 1 week"
                ],
                "treatment": "Conservative management"
            },
            "large": {
                "recommendations": [
                    "URGENT: Consider chest tube placement",
                    "Oxygen therapy (15L for reabsorption)",
                    "Respiratory support",
                    "Cardiothoracic surgery consultation"
                ],
                "treatment": "Interventional (chest tube, aspiration, or VATS)"
            }
        },
        "pulmonary_edema": {
            "recommendations": [
                "EMERGENT: Consider ICU admission",
                "Diuretics (furosemide IV)",
                "Oxygen/CPAP support",
                "Cardiac workup (troponin, BNP, echo)",
                "Cardiology consultation",
                "Treat underlying cause (heart failure, renal disease)"
            ]
        },
        "normal": {
            "recommendations": [
                "No acute findings",
                "Routine follow-up as per primary care",
                "Outpatient management appropriate"
            ]
        }
    }
    
    @staticmethod
    def generate_recommendations(disease: str, severity: str) -> List[str]:
        """Get protocol-based recommendations."""
        protocol = ClinicalRecommender.DISEASE_PROTOCOLS.get(disease, {})
        
        if isinstance(protocol, dict) and severity in protocol:
            return protocol[severity].get("recommendations", [])
        elif isinstance(protocol, dict) and "recommendations" in protocol:
            return protocol["recommendations"]
        
        return ["Consult clinical guidelines and attending physician"]
    
    @staticmethod
    def generate_explanation(disease: str, probability: float, 
                            findings: List[str], severity: str = None) -> ClinicalExplanation:
        """Generate comprehensive clinical explanation."""
        
        # Confidence interpretation
        if probability > 0.9:
            confidence = "high"
        elif probability > 0.7:
            confidence = "moderate"
        else:
            confidence = "low"
        
        # Recommendations
        recommendations = ClinicalRecommender.generate_recommendations(disease, severity or "unknown")
        
        explanation = ClinicalExplanation(
            disease=disease,
            predicted_probability=probability,
            severity=severity or "unknown",
            confidence=confidence,
            key_findings=findings,
            supporting_evidence=[
                f"Model confidence: {probability:.1%}",
                f"Symbolic rules matched: {len(findings)}",
            ],
            contradicting_evidence=[],
            differential_diagnoses=[],
            clinical_significance=f"{disease} requires prompt evaluation and management",
            recommendations=recommendations,
            limitations=[
                "This is an AI-assisted diagnosis, not a replacement for clinical judgment",
                "Radiologist review is mandatory for clinical decision-making",
                "Consider clinical context, patient history, and physical examination"
            ],
            disclaimer=(
                "⚠️ DISCLAIMER: This model is for research/educational purposes only. "
                "Not approved for clinical use. Always consult qualified healthcare professionals."
            )
        )
        
        return explanation


class RiskStratification:
    """Stratify patients into risk categories."""
    
    RISK_CATEGORIES = {
        "low_risk": {
            "criteria": ["stable_vitals", "mild_symptoms", "normal_labs"],
            "action": "Outpatient management",
            "follow_up": "48-72 hours"
        },
        "intermediate_risk": {
            "criteria": ["borderline_vitals", "moderate_symptoms", "abnormal_labs"],
            "action": "Close observation or short-term hospitalization",
            "follow_up": "24 hours"
        },
        "high_risk": {
            "criteria": ["unstable_vitals", "severe_symptoms", "critical_labs"],
            "action": "ICU admission, intensive monitoring",
            "follow_up": "Continuous"
        }
    }
    
    @staticmethod
    def stratify_risk(patient_data: Dict) -> Dict[str, str]:
        """
        Stratify patient into risk category.
        """
        score = 0
        
        # Simple risk scoring
        if patient_data.get("heart_rate", 60) > 120 or patient_data.get("heart_rate", 60) < 50:
            score += 2
        if patient_data.get("systolic_bp", 120) < 90 or patient_data.get("systolic_bp", 120) > 140:
            score += 2
        if patient_data.get("temperature", 37) > 38.5 or patient_data.get("temperature", 37) < 36:
            score += 1
        if patient_data.get("o2_sat", 98) < 92:
            score += 3
        
        if score >= 6:
            risk_level = "high_risk"
        elif score >= 3:
            risk_level = "intermediate_risk"
        else:
            risk_level = "low_risk"
        
        return {
            "risk_level": risk_level,
            "risk_score": score,
            **RiskStratification.RISK_CATEGORIES[risk_level]
        }


class DecisionSupportFormatter:
    """Format recommendations for clinical display."""
    
    @staticmethod
    def format_report(explanation: ClinicalExplanation) -> str:
        """Generate human-readable clinical report."""
        
        report = f"""
╔═════════════════════════════════════════════════════════════╗
║              MEDSYMBOL CLINICAL SUPPORT REPORT              ║
╚═════════════════════════════════════════════════════════════╝

PRIMARY FINDING: {explanation.disease.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Probability:        {explanation.predicted_probability:.1%}
Confidence:         {explanation.confidence.upper()}
Severity:           {explanation.severity}

KEY FINDINGS:
"""
        for finding in explanation.key_findings:
            report += f"  • {finding}\n"
        
        report += f"""
SUPPORTING EVIDENCE:
"""
        for evidence in explanation.supporting_evidence:
            report += f"  ✓ {evidence}\n"
        
        report += f"""
DIFFERENTIAL DIAGNOSIS:
"""
        for dx, prob in explanation.differential_diagnoses[:3]:
            report += f"  • {dx}: {prob:.1%}\n"
        
        report += f"""
RECOMMENDED ACTIONS:
"""
        for i, rec in enumerate(explanation.recommendations, 1):
            report += f"  {i}. {rec}\n"
        
        report += f"""
IMPORTANT LIMITATIONS:
"""
        for limitation in explanation.limitations:
            report += f"  ⚠️  {limitation}\n"
        
        report += f"""
CLINICAL SIGNIFICANCE:
  {explanation.clinical_significance}

{explanation.disclaimer}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated by MedSymbol v1.0 | For Research Use Only
"""
        
        return report


if __name__ == "__main__":
    # Example: Generate clinical explanation
    explanation = ClinicalRecommender.generate_explanation(
        disease="pneumonia",
        probability=0.87,
        findings=["consolidation_RLL", "fever", "productive_cough"],
        severity="moderate"
    )
    
    # Format for display
    formatter = DecisionSupportFormatter()
    report = formatter.format_report(explanation)
    print(report)
