# MedSymbol Architecture Diagrams

## Overview

This directory contains three comprehensive diagrams explaining the MedSymbol neuro-symbolic framework:

### STATS  Diagram 1: [Architecture & Data Flow](ARCHITECTURE_DIAGRAM.md)
**File:** `ARCHITECTURE_DIAGRAM.md`

Shows the complete system from input to output:
- 7 integrated modules 
- Multi-input fusion (images + text + tables + history)
- Data transformations at each stage
- Processing paths through entropy-gated controller

**Key Insight:** 4 independent input modalities fuse ---> 512-dim unified patient representation ---> splits into concept extraction + diagnosis prediction

---

### FAST  Diagram 2: [3-Path Inference Workflow](INFERENCE_WORKFLOW_DIAGRAM.md)
**File:** `INFERENCE_WORKFLOW_DIAGRAM.md`

The entropy-gated decision logic:
- **FAST PATH** (H < 0.3): High confidence ---> light checks (~10ms)
- **STANDARD PATH** (0.3 ≤ H < 1.5): Moderate confidence ---> full verification (~200ms)
- **DEFER PATH** (H ≥ 1.5): High uncertainty ---> human review

Includes entropy threshold examples and clinical workflow integration.

**Key Insight:** System automatically routes cases based on confidence level, balancing speed vs verification depth

---

### 🛡️ Diagram 3: [Ontology Constraint Masking](ONTOLOGY_MASKING_DIAGRAM.md)
**File:** `ONTOLOGY_MASKING_DIAGRAM.md`

The age/sex/symptom filtering mechanism that answers "Does this project have age limitations?":

**5 Constraint Checks:**
1. Age validity (pediatric vs adult vs geriatric)
2. Sex constraints (e.g., prostate diseases only for males)
3. Symptom consistency (e.g., MI requires chest pain)
4. Lab value consistency (e.g., infection requires elevated WBC)
5. Hierarchical consistency with SNOMED-CT

**Real Example:** 8-year-old with fever+cough
- OK  Pneumonia (0.56) - valid for age & symptoms
- OK  Bronchitis (0.28) - valid for age & symptoms  
- NO  Myocardial Infarction - MASKED (age<30 required)
- NO  Prostate Cancer - MASKED (males only, age 40+)

**Key Insight:** Invalid diagnoses logits ---> -infinity, softmax ---> 0, so only medically valid options appear

---

## How to View These Diagrams

### Option 1: Online (Best)
Copy the mermaid code blocks into:
- [Mermaid Live Editor](https://mermaid.live)
- GitHub (renders automatically)
- Notion/Obsidian/any markdown renderer with Mermaid support

### Option 2: VS Code
- Install "Markdown Preview Mermaid Support" extension
- Open the `.md` files in preview mode

### Option 3: Convert to Images
```bash
# Install mmdc (Mermaid CLI)
npm install -g @mermaid-js/mermaid-cli

# Convert to PNG/SVG
mmdc -i ARCHITECTURE_DIAGRAM.md -o architecture.png
mmdc -i INFERENCE_WORKFLOW_DIAGRAM.md -o workflow.png
mmdc -i ONTOLOGY_MASKING_DIAGRAM.md -o masking.png
```

---

## Quick Summary Table

| Aspect | Details |
|--------|---------|
| **Input Modalities** | X-ray (224×224) + Clinical Notes (512 tokens) + Lab Results (50 features) + History (20 features) |
| **Neural Processing** | ViT-B/16 + BioBERT + TabNet + MLP ---> 512-dim patient representation |
| **Concept Extraction** | 500+ SNOMED codes with probabilities |
| **Uncertainty Metric** | Shannon Entropy of diagnosis probabilities |
| **Decision Threshold** | tau_low=0.3, tau_high=1.5 (configurable) |
| **Fast Path Latency** | ~10ms (age + sex + basic checks) |
| **Standard Path Latency** | ~200ms (full 5-check verification + proof) |
| **Deferred Path** | Human review (no latency bound) |
| **Constraint Sources** | SNOMED-CT (350K+), ICD-11 (55K+), CPG (500 rules) |
| **Proof Format** | SMT-LIB (Z3 solver verifiable) |
| **Age Limitations** | OK  YES - Explicitly enforced per diagnosis |

---

## Document References

- **Main Documentation:** See `../README.md` for project overview
- **Technical Details:** See `MedSymbol_Technical_Details.docx` (extracted from chat)
- **Configuration:** See `../configs/default.yaml` for model hyperparameters
- **Ontology Data:** See `../data/ontology/icd11_medsymbol_codes.json`

---

## Navigation

```
medsymbol/
├── ARCHITECTURE_DIAGRAM.md          ← Start here for overall system
├── INFERENCE_WORKFLOW_DIAGRAM.md    ← Understand 3-path routing
├── ONTOLOGY_MASKING_DIAGRAM.md      ← Deep dive on constraints
├── README.md                        ← Project overview
├── configs/
│   └── default.yaml                 ← Model configuration
├── data/
│   └── ontology/
│       └── icd11_medsymbol_codes.json
└── src/
    ├── encoders/                    ← Module 1 (neural)
    └── symbolic/                    ← Modules 5-7 (verification)
```

---

**Generated:** April 8, 2026  
**Format:** Mermaid diagram markup + detailed explanations
