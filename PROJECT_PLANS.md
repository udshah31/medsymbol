# MedSymbol Project Plans

## 🎯 Primary Goal
Create a **neuro-symbolic framework that combines deep learning accuracy with formally verifiable reasoning** for medical diagnosis—specifically with chest X-ray interpretation.

---

## 📋 Project Phases

### **Phase 1: Proof of Concept (Current)**
- ✅ Architecture designed (7 modules)
- ✅ Configuration framework ready
- ✅ NIH ChestX-ray14 dataset downloaded
- ✅ Ontology mappings defined (SNOMED-CT, ICD-11)
- ⏳ **TODO:** Implement core encoder/symbolic modules

### **Phase 2: Model Development**

**Module 1: Multimodal Neural Encoder**
- Vision Encoder: ViT-B/16 for X-ray images (768-dim output)
- Language Encoder: BioBERT for clinical notes (768-dim output)
- Tabular Encoder: TabNet for lab values (256-dim output)
- History Encoder: MLP for patient demographics (128-dim output)
- Fusion layer: Combine 4 modalities (1920-dim) → 512-dim unified patient representation

**Module 2-3: Diagnosis Prediction**
- Concept extraction network: Multi-label classification to 500+ SNOMED codes
- Diagnosis prediction head: FC layers with entropy calculation
- Softmax probability distribution over N_diagnoses

**Module 4: Entropy-Gated Controller**
- Calculate Shannon entropy: H = -Σ(p * log(p))
- Route based on thresholds:
  - H < 0.3: FAST PATH (lightweight checks, ~10ms)
  - 0.3 ≤ H < 1.5: STANDARD PATH (full verification, ~200ms)
  - H ≥ 1.5: DEFER PATH (human review)

**Modules 5-7: Symbolic Verification**
- **Module 5: Symbolic Verifier** — 5 consistency checks:
  1. Age validity (pediatric/adult/geriatric constraints)
  2. Symptom consistency (expected vs actual)
  3. Lab value consistency (WBC, O2 sat, etc.)
  4. Contraindication check (drug interactions, immunocompromised, etc.)
  5. Hierarchical consistency (SNOMED-CT relationships)
  
- **Module 6: Ontology Constraint Masking** — Apply logit masking:
  - Invalid diagnoses → logit = -∞
  - Valid diagnoses → logit unchanged
  - Results in constrained softmax distribution
  
- **Module 7: Proof Certificate Generator** — SMT-LIB format:
  - Formal logic proof in Z3-verifiable format
  - Declares patient variables, assertions, clinical rules
  - Check satisfiability: SAT = consistent, UNSAT = contradiction

### **Phase 3: Dataset Expansion**
Current: NIH ChestX-ray14 (prototyping)
Target progression:
1. **CheXpert** (Stanford) - 224K images (medium-scale benchmark)
2. **MIMIC-CXR** (PhysioNet) - 450K images (production-scale, requires credentialed access)
3. **MIMIC-IV** - Integrate tabular clinical data with imaging

### **Phase 4: Evaluation & Validation**

**Target Performance Metrics:**
- AUC-ROC > 0.90
- Sensitivity > 0.85
- Specificity > 0.85
- Ontology consistency > 0.99
- Mean latency < 500ms
- P95 latency < 1000ms
- Proof validity 100% (all generated proofs must be valid)

**Pathway Testing:**
- Fast Path validation (~10ms for high-confidence cases)
- Standard Path verification (full 5-check ontology validation)
- Deferred Path handling (human expert escalation, top-3 differentials)

**Clinical Guidelines Integration:**
- CAP Diagnosis (IDSA/ATS 2019): 15 rules
- Pneumonia Severity (CURB-65, PSI): 8 rules
- Differential Diagnosis (UpToDate): 25 rules
- Contraindications (FDA drug labels): 50 rules

### **Phase 5: Clinical Integration & FDA Compliance**
- Machine-verifiable proof certificates for audit trails
- Integration with hospital PACS systems
- Clinical decision support validation studies
- FDA documentation package for potential 510(k) submission
- Formal verification reports demonstrating consistency

---

## 🔬 Technical Implementation Plan

### **Currently Scaffolded (Need Implementation):**
```
src/encoders/
  ├── __init__.py              [Empty - needs implementations]
  ├── vision.py                [ViT-B/16 encoder]
  ├── text.py                  [BioBERT encoder]
  ├── tabular.py               [TabNet encoder]
  ├── history.py               [MLP history encoder]
  └── fusion.py                [1920->512 dim fusion layer]

src/symbolic/
  ├── __init__.py              [Empty - needs implementations]
  ├── verifier.py              [5-check verification logic]
  ├── masking.py               [Ontology constraint masking]
  ├── proofs.py                [SMT-LIB proof generation]
  └── constraints.py           [Age/sex/symptom rules from SNOMED-CT]

src/utils/
  ├── data_loader.py           [Load NIH/CheXpert/MIMIC-CXR data]
  ├── metrics.py               [AUC, sensitivity, specificity, etc.]
  └── visualization.py         [Plot ROC curves, confusion matrices]
```

### **Scripts to Implement:**
- `scripts/train_model.py` — Main training loop with Module 1-3
- `scripts/evaluate_model.py` — Benchmark against test sets
- `scripts/inference.py` — Single patient inference pipeline
- `scripts/generate_proof.py` — Standalone proof certificate generator
- `scripts/test_thresholds.py` — Optimize entropy thresholds (τ_low, τ_high)

---

## 📊 Expected Outputs Per Diagnosis

**Complete Inference Output:**
```json
{
  "diagnosis": "Community-Acquired Pneumonia (J18.9)",
  "confidence": {
    "neural_score": 0.89,
    "symbolic_verification": "VERIFIED",
    "combined_confidence": "HIGH"
  },
  "evidence": {
    "extracted_concepts": [
      {"code": "233604007", "label": "Pneumonia", "probability": 0.89},
      {"code": "386661006", "label": "Fever", "probability": 0.92},
      {"code": "281647001", "label": "Lung opacity", "probability": 0.94}
    ],
    "verification_checks": {
      "age_validity": "PASS",
      "symptom_consistency": "PASS",
      "lab_consistency": "PASS",
      "contraindications": "PASS",
      "hierarchical_consistency": "PASS"
    }
  },
  "explanation": "The diagnosis of pneumonia is supported by: chest X-ray consolidation (0.94), fever and cough symptoms, elevated WBC (14200/uL). Diagnosis is consistent with SNOMED-CT hierarchy and CPG-PNEUMONIA-001 criteria.",
  "proof_certificate": {
    "id": "MS-2026-04-06-00142",
    "format": "SMT-LIB",
    "z3_result": "SAT",
    "verification_url": "medsymbol.verify('MS-2026-04-06-00142')"
  },
  "differentials": [
    {"diagnosis": "Viral pneumonia (J12.9)", "confidence": 0.15, "note": "Consider if no improvement"},
    {"diagnosis": "Acute bronchitis (J20.9)", "confidence": 0.08, "note": "Less likely given X-ray"}
  ],
  "processing": {
    "entropy": 0.73,
    "path_taken": "STANDARD_PATH",
    "latency_ms": 187
  }
}
```

**Audit Trail (For FDA/Compliance):**
- Full history of checks performed
- Which ontology constraints were applied
- Why diagnoses were masked (age, sex, symptom constraints)
- Latency breakdown (neural inference vs symbolic verification)
- Proof certificate hash for tamper detection

---

## 🎓 Key Innovation Areas

1. **Age-appropriate diagnosis filtering** 
   - Automatically masks invalid diagnoses for pediatric/adult/elderly patients
   - Example: Myocardial infarction invalid for 8-year-old (masked)

2. **Entropy-gated routing** 
   - Fast path for confident cases (high probability, low entropy)
   - Full verification for uncertain cases
   - Human review for highly uncertain cases

3. **Formal verification** 
   - Z3 SMT proofs for FDA/regulatory compliance
   - Machine-checkable proof certificates
   - 100% proof validity target

4. **Explainability** 
   - Every mask decision traceable to ontology rules
   - Clinical guideline citations
   - Differential diagnosis ranking with reasoning

5. **Multi-modal fusion** 
   - Combines images (ViT), text (BioBERT), labs (TabNet), history (MLP)
   - Unified 512-dim representation
   - All modalities equally weighted in fusion

---

## 🚀 Success Criteria

### **Immediate (Next 2-4 weeks)**
- [ ] Implement all 4 sub-encoders + fusion layer
- [ ] Implement concept extraction head
- [ ] Implement entropy calculator
- [ ] Test on NIH ChestX-ray14 validation set

### **Short-term (Weeks 4-8)**
- [ ] Implement 5-check symbolic verifier
- [ ] Implement ontology constraint masking
- [ ] Integrate Z3 SMT solver
- [ ] Implement proof certificate generator
- [ ] Achieve AUC > 0.85 on NIH CXR14

### **Medium-term (Weeks 8-16)**
- [ ] Encrypt to CheXpert dataset
- [ ] Achieve AUC > 0.90 on CheXpert
- [ ] Optimize entropy thresholds via validation data
- [ ] Benchmark latency: FAST < 20ms, STANDARD < 250ms
- [ ] Generate comprehensive evaluation report

### **Long-term (Weeks 16+)**
- [ ] Obtain MIMIC-CXR credentialed access
- [ ] Scale to MIMIC-CXR (~450K images)
- [ ] FDA pre-submission meeting (Q-submission)
- [ ] Clinical validation study protocol
- [ ] Publication in peer-reviewed venue

---

## 📈 Future Directions (Post-MVP)

1. **Multi-pathology support** 
   - Extend beyond pneumonia/fever/cough
   - Cover all 14 CheXpert labels

2. **Longitudinal reasoning** 
   - Track diagnosis changes over time
   - Progressive disease vs acute event detection

3. **Drug interaction checking** 
   - Integrate with treatment recommendations
   - Check contraindications via Z3

4. **Active learning** 
   - Model learns from human expert corrections
   - Uncertainty sampling for expert annotation

5. **Federated learning** 
   - Multi-site training while preserving privacy
   - Decentralized proof generation

6. **Real-time alerts** 
   - Integration with hospital EHR systems
   - Push notifications for critical findings

---

## 📁 Project Structure Reference

```
medsymbol/
├── configs/
│   └── default.yaml                 # Model hyperparameters & thresholds
├── data/
│   ├── raw/nih_cxr14/              # NIH dataset (224K images)
│   ├── processed/                   # Preprocessed data splits
│   └── ontology/icd11_medsymbol_codes.json
├── notebooks/                       # Exploration & analysis
├── scripts/
│   ├── train_model.py              # Main training loop
│   ├── evaluate_model.py            # Metrics & benchmarks
│   ├── inference.py                 # Single patient inference
│   └── verify_env.py                # Environment validation
├── src/
│   ├── encoders/                    # Module 1: Neural encoders
│   ├── symbolic/                    # Modules 5-7: Verification
│   └── utils/                       # Data loading, metrics
├── tests/                           # Unit tests
├── experiments/                     # Results & logs
├── ARCHITECTURE_DIAGRAM.md          # Visual system overview
├── INFERENCE_WORKFLOW_DIAGRAM.md    # 3-path routing logic
├── ONTOLOGY_MASKING_DIAGRAM.md      # Age/sex filtering
├── DIAGRAMS_README.md               # Diagram navigation
└── PROJECT_PLANS.md                 # This file
```

---

## 🔗 Key Dependencies

**Core ML Stack:**
- torch >= 2.0 (neural networks)
- transformers >= 4.30 (ViT, BioBERT)
- pytorch-tabnet >= 4.0 (TabNet encoder)

**Symbolic Reasoning:**
- z3-solver >= 4.12 (SMT verification)
- owlready2 >= 0.40 (SNOMED-CT/ICD-11 parsing)

**Data & Utilities:**
- pandas >= 2.0 (data manipulation)
- scikit-learn >= 1.3 (metrics)
- matplotlib, seaborn (visualization)
- wandb >= 0.15 (experiment tracking)

---

## 📞 Contact & Questions

For questions about:
- **Architecture:** See ARCHITECTURE_DIAGRAM.md
- **Inference Logic:** See INFERENCE_WORKFLOW_DIAGRAM.md
- **Ontology Constraints:** See ONTOLOGY_MASKING_DIAGRAM.md
- **Technical Details:** See MedSymbol_Technical_Details.docx
- **Configuration:** See configs/default.yaml

---

**Last Updated:** April 8, 2026  
**Status:** Phase 1 (Proof of Concept)  
**Next Milestone:** Complete Module 1-7 implementation
