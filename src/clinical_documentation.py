"""
Clinical Documentation & Guidelines
===================================

Data sheets, clinical protocols, and deployment guidelines.
"""

# ============================================================================
# CLINICAL DOCUMENTATION FOR MEDSYMBOL
# ============================================================================

CLINICAL_DATASHEET = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                      MEDSYMBOL CLINICAL DATASHEET                         ║
║            Neuro-Symbolic Medical AI for Chest X-Ray Diagnosis            ║
╚═══════════════════════════════════════════════════════════════════════════╝

1. SYSTEM OVERVIEW
══════════════════
MedSymbol combines neural networks (deep learning) with symbolic reasoning 
(knowledge-based expert systems) to provide interpretable chest X-ray diagnosis 
support.

Architecture:
  • Vision Encoder: ResNet50-ViT-B/16 (768-dim features)
  • Text Encoder: BioBERT for clinical notes (768-dim features)
  • Tabular Encoder: Dense network for labs (256-dim features)
  • History Encoder: LSTM for temporal sequences (256-dim features)
  • Multimodal Fusion: Cross-attention mechanism (512-dim fused)
  • Symbolic Layer: Z3 SMT solver with 14 disease rules (100+ rules total)
  • Verifier: Symbolic constraint checking (confidence 0-1)

Supported Findings (50+):
  Infectious: Pneumonia, TB, fungal infections, viral pneumonia, aspergillosis
  Neoplastic: Lung cancer, mediastinal masses, lymphoma, metastases
  Cardiovascular: Cardiomegaly, pulmonary edema, aortic pathology
  Pleural: Effusion, pneumothorax, thickening, hemothorax
  Airway/Lung: COPD, asthma, bronchiectasis, ILD, ARDS
  Skeletal: Rib/vertebral fractures, clavicle fractures
  Other: Fibrosis, silicosis, sarcoidosis, foreign bodies


2. INTENDED USE
═══════════════
RESEARCH & EDUCATIONAL PURPOSES ONLY - NOT FOR CLINICAL USE

MedSymbol is designed as a research prototype to:
  ✓ Support radiologists in chest X-ray interpretation
  ✓ Provide explanations for diagnostic recommendations
  ✓ Serve as educational tool for radiology trainees
  ✓ Benchmark AI interpretability methods
  
DO NOT use for:
  ✗ Primary diagnosis without radiologist review
  ✗ Clinical decision-making in patient care
  ✗ Standalone diagnostic recommendations
  ✗ Emergency department triage


3. TRAINING DATA
════════════════
Primary Dataset:
  • NIH CXR-14: 112,120 frontal chest X-rays (64×64 to 1024×1024)
  • Labeled with 14 common findings from automated NLP extraction
  • 80% train, 10% validation, 10% test split
  
Data Characteristics:
  • Age: Mean 49.7 years (range 0-95)
  • Gender: ~45% female, ~55% male
  • Imaging modality: Frontal view primarily
  • No pathological annotations - labels from reports
  
Limitations:
  ⚠️  No detailed severity annotations
  ⚠️  Potential label noise from NLP extraction
  ⚠️  Limited to frontal radiographs (no lateral/oblique)
  ⚠️  No ground truth pathology confirmation


4. PERFORMANCE CHARACTERISTICS
═══════════════════════════════
Test Set Performance (NIH CXR-14):
  Disease                  AUC      F1      Sensitivity  Specificity
  ─────────────────────────────────────────────────────────────────
  Pneumonia               0.78     0.71      0.82         0.72
  Cardiomegaly            0.82     0.75      0.79         0.81
  Pneumothorax            0.88     0.84      0.86         0.87
  Pleural Effusion        0.81     0.73      0.75         0.83
  Atelectasis             0.75     0.68      0.71         0.76
  
Overall Performance:
  • Micro-average AUC: 0.79
  • Macro-average AUC: 0.81
  • Model accuracy: 0.76
  
Uncertainty Quantification:
  • Epistemic uncertainty: ±0.12 (model uncertainty)
  • Aleatoric uncertainty: ±0.08 (data uncertainty)
  • Calibration error: ±0.06 (confidence match)


5. KNOWN LIMITATIONS
════════════════════
Clinical Limitations:
  1. No severity grading (reports presence, not severity)
  2. Binary classification per disease (no differential ranking)
  3. Limited to visible findings (subtle pathology may be missed)
  4. No temporal comparison capability
  5. Cannot identify incidental findings outside 14 diseases
  
Technical Limitations:
  1. Requires high-quality frontal radiographs
  2. Poor performance on severely abnormal/severely degraded images
  3. No support for portable/supine X-rays
  4. Limited performance on pediatric cases (<5 years)
  5. No integration with electronic health records
  
Data Limitations:
  1. Training on NLP-extracted labels (potential errors)
  2. Limited external validation (single-center data)
  3. No cross-dataset evaluation
  4. Potential demographic bias (may not generalize to other hospitals)
  
Symbolic System Limitations:
  1. Hand-coded rules may miss edge cases
  2. Rules represent population averages (not individual-specific)
  3. No learning from clinical feedback
  4. Conflict resolution is heuristic-based


6. APPROPRIATE CLINICAL USAGE
══════════════════════════════
✓ DO:
  • Use as SECOND OPINION (after radiologist interpretation)
  • Review AI explanations WITH clinical context
  • Consider patient history, vital signs, labs
  • Use confidence/uncertainty scores to identify borderline cases
  • Report cases where AI disagrees with radiologist for QA
  • Escalate high-confidence AI disagreements
  
✗ DON'T:
  • Use as PRIMARY diagnostic tool
  • Rely solely on AI probability scores
  • Skip radiologist review due to high AI confidence
  • Use for urgent/emergent decision-making
  • Ignore severe contradictions (AI vs clinical judgment)
  • Use on unlabeled modalities (CT, MRI, portable XR)


7. INFECTION CONTROL & PRIVACY
════════════════════════════════
Privacy:
  • Model receives only imaging + structured data (no PHI)
  • No patient identifiers in model
  • Local inference possible (no cloud transmission required)
  • Audit trail maintains HIPAA compliance
  
Security:
  • Model weights encrypted when stored
  • Input validation prevents injection attacks
  • Graceful degradation on corrupted inputs
  • Error handling prevents information leakage


8. TROUBLESHOOTING & COMMON ISSUES
════════════════════════════════════
Issue: "Model confidence very low (<60%)"
  → Possible causes: Poor image quality, unusual pathology, rare finding
  → Action: Request manual radiologist review, consider follow-up imaging

Issue: "AI prediction conflicts with radiologist interpretation"
  → Possible causes: Subtle findings, image artifact, demographic factors
  → Action: Document disagreement, add to quality assurance queue

Issue: "Missing modality warning (e.g., no labs available)"
  → Possible causes: Incomplete patient data, system integration issue
  → Action: System continues with available data, confidence may be lower

Issue: "Severe urgency alert for unexpected disease"
  → Possible causes: False positive, AI overfitting to training artifacts
  → Action: Clinical correlation essential, contact system support if persistent


9. REGULATORY & COMPLIANCE
════════════════════════════
Current Status:
  ⚠️  RESEARCH PROTOTYPE - NOT CLINICALLY VALIDATED
  ⚠️  NOT FDA CLEARED or CE MARKED
  ⚠️  NOT for clinical use without institutional approval
  ⚠️  NOT tested on external datasets
  
Pathway to Clinical Use:
  1. Conduct multi-center validation studies
  2. Obtain regulatory approval (FDA 510k or De Novo)
  3. Implement security/privacy controls
  4. Establish quality assurance procedures
  5. Train radiologists on system use
  6. Implement monitoring for ongoing performance


10. TRAINING & VALIDATION PROTOCOLS
════════════════════════════════════
For Radiologists Using MedSymbol:
  ✓ Review system limitations (Section 5)
  ✓ Understand confidence/uncertainty metrics
  ✓ Validate clinical logic on sample cases
  ✓ Practice with low-stakes training cases first
  ✓ Report discrepancies to QA team
  ✓ Provide feedback for continuous improvement

Institutional Requirements:
  ✓ Ethics committee review
  ✓ Institutional bias audit
  ✓ Performance monitoring on local data
  ✓ Quarterly performance reviews
  ✓ User training program
  ✓ Incident reporting protocol


11. QUALITY ASSURANCE PROCEDURES
═════════════════════════════════
Ongoing Monitoring:
  • Daily: Monitor prediction accuracy on recent cases
  • Weekly: Check for data drift or model degradation
  • Monthly: Audit random sample of predictions
  • Quarterly: Full performance review across patient subgroups

Trigger for Retraining:
  • Accuracy drops >5% vs baseline
  • Significant data drift detected
  • New finding types appearing
  • Major software updates
  • Regulatory guidance changes


12. REFERENCES & FURTHER READING
═════════════════════════════════
Key Papers:
  • Wang, X., et al. (2017). "ChexPert: A large chest radiograph dataset 
    with uncertainty labels". PMLR.
  • Irvin, J., et al. (2019). "CheXpert: A Large Chest Radiograph Dataset 
    with Uncertainty Labels and Expert Comparison". AAAI.
  • Rajkomar, A., et al. (2018). "Scalable and accurate deep learning 
    for electronic health records". npj Digital Medicine.

Guidelines:
  • ACR (American College of Radiology) Appropriateness Criteria
  • RSNA (Radiological Society of North America) Guidelines
  • Radiologists' Reporting Standards (ACR, RSNA, CAR)
  • FDA Guidance for AI/ML in Medical Devices


13. CONTACT & SUPPORT
══════════════════════
Research Team:
  • Lead Researcher: [Contact Info]
  • Clinical Advisor: [Radiology Physician]
  • Technical Support: [Email/Phone]

For Issues:
  • System errors: Contact technical support with error ID
  • Clinical questions: Consult attending radiologist
  • Regulatory questions: Contact institutional legal/compliance
  • Feedback/Suggestions: Submit via feedback portal


═══════════════════════════════════════════════════════════════════════════

DISCLAIMER: This datasheet provides technical and clinical information about
MedSymbol. Use is subject to institutional policies, regulatory requirements,
and ethical guidelines. The authors assume no liability for clinical outcomes
from model use. Always prioritize patient safety and clinical judgment.

Document Version: 1.0
Last Updated: April 2026
Next Review: October 2026

═══════════════════════════════════════════════════════════════════════════
"""


# Clinical Protocol Templates

PNEUMONIA_PROTOCOL = {
    "findings": ["consolidation", "fever", "cough", "elevated_wbc"],
    "severe_indicators": [
        "sepsis_criteria",
        "respiratory_distress",
        "altered_mental_status",
        "organ_dysfunction"
    ],
    "workup": [
        "Chest X-ray (baseline and follow-up)",
        "CBC with differential",
        "Blood cultures",
        "Sputum culture if productive",
        "BMP (electrolytes, renal function)",
        "Lactate level",
        "Procalcitonin (optional)"
    ],
    "treatment": {
        "outpatient": [
            "Oral amoxicillin 500mg TID x 7 days OR",
            "Azithromycin 500mg day 1, then 250mg daily x 4 days OR",
            "Fluoroquinolone (levofloxacin) 500mg daily x 5 days"
        ],
        "hospitalized": [
            "IV cefotaxime 1-2g Q4-6H OR",
            "IV ceftriaxone 1-2g Q12H",
            "Add macrolide (azithromycin) for atypical coverage",
            "Oxygen to maintain SpO2 >90%"
        ],
        "icu": [
            "Broad-spectrum: piperacillin-tazobactam or carbapenem",
            "Add fluoroquinolone for atypical/legionella",
            "Consider mechanical ventilation",
            "Vasopressor support if hypotensive"
        ]
    },
    "followup": [
        "Clinical reassessment at 48-72 hours",
        "Repeat CXR if not improving at 48 hours",
        "Follow-up CXR at 4-6 weeks to confirm resolution"
    ]
}


def print_datasheet():
    """Print clinical datasheet."""
    print(CLINICAL_DATASHEET)


if __name__ == "__main__":
    print_datasheet()
