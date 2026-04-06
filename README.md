# MedSymbol

**A Neuro-Symbolic Framework for Verifiably Consistent Medical Diagnosis**

MedSymbol bridges the gap between deep learning diagnostic accuracy and formally verifiable reasoning in medical AI. Instead of just explaining model outputs, MedSymbol generates machine-checkable proof certificates proving each diagnosis is logically consistent with established medical ontologies (SNOMED-CT, ICD-11).

## Architecture

```
Input (X-ray, Notes, Labs, History)
    → Module 1: Multimodal Neural Encoder (ViT + BioBERT + TabNet + MLP → 512-dim)
    → Module 2: Concept Extraction Network (→ SNOMED codes)
    → Module 3: Diagnosis Prediction Head (+ entropy calculation)
    → Module 4: Entropy-Gated Controller
        ├── Fast Path (H < τ_low):    light check (~10ms)
        ├── Standard Path:            full verification (~200ms)
        │   → Module 5: Symbolic Verification Engine
        │   → Module 6: Ontology-Constrained Output
        │   → Module 7: Proof Certificate Generator
        └── Defer Path (H ≥ τ_high):  human review
    → Final Output: Diagnosis + Confidence + Proof Certificate
```

## Project Structure

```
medsymbol/
├── configs/           # Training & experiment configs
├── data/
│   ├── raw/           # Original datasets (gitignored)
│   ├── processed/     # Preprocessed data (gitignored)
│   └── ontology/      # SNOMED-CT, ICD-11, clinical rules
├── src/
│   ├── encoders/      # Module 1: ViT, BioBERT, TabNet, Fusion
│   ├── symbolic/      # Modules 5-7: Verification, Constraints, Proofs
│   └── utils/         # Data loading, metrics, visualization
├── notebooks/         # Exploration & analysis
├── experiments/       # Experiment results & configs
├── scripts/           # Training & evaluation scripts
└── tests/             # Unit tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Datasets

- **Primary:** MIMIC-CXR + MIMIC-IV (PhysioNet, credentialed access)
- **Benchmark:** CheXpert (Stanford AIMI)
- **Prototyping:** NIH ChestX-ray14 (open access)

## Ontologies

- **SNOMED-CT:** Via UMLS (NLM account)
- **ICD-11:** WHO API (free)
