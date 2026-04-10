# MedSymbol Quick Start Guide

## 5-Minute Setup

```bash
# 1. Clone & install
git clone https://github.com/udshah31/medsymbol.git
cd medsymbol
pip install -e .

# 2. Check dataset status
python scripts/orchestrate.py --stage status

# 3. Download metadata only (quick)
python scripts/orchestrate.py --stage download --dataset nih_labels

# 4. Train model (locally or on GPU)
python scripts/orchestrate.py --stage train --epochs 10 --batch_size 16

# 5. Evaluate
python scripts/orchestrate.py --stage evaluate
```

---

## Stage-by-Stage Guide

### Stage 1: Download Dataset

**Option A: Metadata Only (10 MB, ~1 min)**
```bash
python scripts/orchestrate.py --stage download --dataset nih_labels
```
Shows disease distribution and statistics. Good for exploring before full download.

**Option B: Full Dataset (42 GB, ~4-8 hours)**
```bash
python scripts/orchestrate.py --stage download --dataset nih_cxr14
```
Downloads all archives but doesn't extract (saves disk space).

**Option C: Full Dataset + Extract (150 GB total, ~8-12 hours)**
```bash
python scripts/orchestrate.py --stage download --dataset nih_cxr14 --extract
```
Downloads and extracts. Requires 150GB temporary disk space.

**To extract later:**
```bash
python scripts/download_datasets.py --extract-only
```

---

### Stage 2: Check Dataset Status

```bash
python scripts/orchestrate.py --stage status
```

Shows:
- ✓/✗ Labels CSV status
- ✓/✗ BBox CSV status  
- ◐ Downloaded archives (X/12)
- ✓/✗ Extracted images count
- Disease distribution table
- Demographics (age, sex breakdown)

---

### Stage 3: Train Model

**Quick training (10 epochs, ~5 min on T4 GPU):**
```bash
python scripts/orchestrate.py --stage train --epochs 10 --batch_size 16
```

**Parameters:**
- `--epochs` — Number of training epochs (default: 10)
- `--batch_size` — Batch size (default: 16, reduce if OOM)
- `--lr` — Learning rate (default: 0.0001)

**Output:**
- Model checkpoint: `models/medsymbol_YYYYMMDD_HHMMSS.pt`
- Metrics logged to Weights & Biases automatically

**Remote (Google Colab):**
```bash
%cd /content/medsymbol
!python scripts/orchestrate.py --stage train --epochs 20 --batch_size 32
```

---

### Stage 4: Evaluate Model

```bash
python scripts/orchestrate.py --stage evaluate --model_path ./models/medsymbol.pt
```

**Output:**
- `evaluation_report.json` — Full metrics per disease
- `metrics_summary.csv` — Easy-to-read CSV table
- `confusion_matrix.png` — Heatmap visualization
- W&B dashboard with interactive charts

**Metrics computed:**
- Overall accuracy
- Per-disease: AUC, Sensitivity, Specificity, F1, Precision, Recall
- Confusion matrix

---

## Complete End-to-End Pipeline

Run everything in one command:
```bash
python scripts/orchestrate.py --all --epochs 20 --batch_size 32
```

This runs:
1. ✓ Check dataset status
2. ✓ Download NIH labels
3. ✓ Train for 20 epochs
4. ✓ Evaluate and generate reports

---

## Using on Google Colab

```python
# Cell 1: Clone & install
!git clone https://github.com/udshah31/medsymbol.git
%cd medsymbol
!pip install -e .

# Cell 2: Check status
!python scripts/orchestrate.py --stage status

# Cell 3: Download dataset
!python scripts/orchestrate.py --stage download --dataset nih_labels

# Cell 4: Train model (uses T4 GPU automatically)
!python scripts/orchestrate.py --stage train --epochs 20 --batch_size 32

# Cell 5: Evaluate
!python scripts/orchestrate.py --stage evaluate

# Cell 6: View results
from IPython.display import Image, display
import json

# Show confusion matrix
display(Image('experiments/confusion_matrix.png'))

# Show metrics
with open('experiments/metrics_summary.csv') as f:
    print(f.read())
```

---

## Integration with Weights & Biases

Automatic tracking of:
- Training loss per epoch
- Learning rate
- Hyperparameters
- All evaluation metrics
- Confusion matrix image

**View dashboard:**
```
https://wandb.ai/[your-username]/medsymbol
```

To disable W&B:
```python
import os
os.environ['WANDB_DISABLED'] = 'true'
```

---

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```bash
python scripts/orchestrate.py --stage train --batch_size 8
```

Enable gradient accumulation in training (already enabled with AMP).

### Slow Download

NIH servers can be slow. Continue interrupted downloads:
```bash
python scripts/orchestrate.py --stage download --dataset nih_cxr14
# Interrupt (Ctrl+C) and run again - resumes from last file
```

### GPU Not Detected

Check CUDA:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Or use CPU:
```bash
# In train_model.py, change device = 'cuda' to device = 'cpu'
```

### No Metrics/W&B Error

Script continues without W&B. To fix:
```bash
pip install wandb
wandb login
```

---

## Architecture Overview

**Modules Implemented:**
1. ✓ Vision Encoder (ViT-B/16) → 768-dim image embeddings
2. ✓ Text Encoder (BioBERT) → 768-dim clinical note embeddings
3. ✓ Tabular Encoder (TabNet) → 256-dim lab value features
4. ✓ History Encoder (MLP) → 128-dim demographic features
5. ✓ Multimodal Fusion → 512-dim unified patient representation
6. ✓ Symbolic Verifier (Z3 SMT) → 5-check consistency verification
7. ✓ Proof Generator (SMT-LIB) → FDA-auditable proof certificates

**Training Loop:**
- Automatic Mixed Precision (AMP) — 50% memory savings
- Gradient Accumulation — effective larger batch sizes
- Binary Cross-Entropy with Logits Loss
- AdamW optimizer
- Weights & Biases tracking

**Inference Pipeline:**
- Entropy gating with 3 routing paths:
  - **FAST** (high confidence) — Skip symbolic verification
  - **STANDARD** (medium) — Run full symbolic checks
  - **DEFER** (low confidence) — Defer to human clinician

---

## Project Structure

```
medsymbol/
├── scripts/
│   ├── orchestrate.py          ← Main entry point (NEW)
│   ├── download_datasets.py    ← Dataset download + analysis  
│   ├── train_model.py          ← Training loop with W&B
│   ├── evaluate_model.py       ← Comprehensive metrics
│   └── integration_test.py     ← Component validation
├── src/
│   ├── model.py                ← MedSymbolModel (Modules 1-7)
│   ├── encoders/               ← Vision, Text, Tabular, History, Fusion
│   ├── symbolic/               ← Verifier, Masking, Proof Generator
│   └── utils/                  ← Data loader, preprocessing
├── data/
│   ├── raw/
│   │   └── nih_cxr14/          ← Downloaded datasets
│   └── processed/              ← Preprocessed splits
├── models/                      ← Saved checkpoints
├── experiments/                 ← Evaluation results & logs
└── pyproject.toml              ← Dependencies
```

---

## Next Steps

1. **Real Data Training** — Download full NIH dataset (~42GB)
2. **Multi-GPU Training** — Use DistributedDataParallel for faster training
3. **CheXpert/MIMIC Integration** — Scale to 450K+ images
4. **Clinical Validation** — Evaluate on additional datasets
5. **FDA Submission** — Formal verification documentation

---

## Performance Targets

**Model Metrics (Project Goals for Phase 2):**
- AUC > 0.90
- Sensitivity > 0.85
- Specificity > 0.85
- Ontology consistency > 0.99
- Inference latency < 500ms

**Current Status (Dummy Data):**
- ✓ Converges successfully (10 epochs, loss 0.67→0.05)
- ✓ All 7 modules execute end-to-end
- ✓ Framework ready for real NIH data

---

## Citation

If you use MedSymbol in research:

```bibtex
@software{medsymbol2026,
  title={MedSymbol: Neuro-Symbolic Medical Diagnosis},
  author={Shah, Uday},
  year={2026},
  url={https://github.com/udshah31/medsymbol}
}
```

---

**Questions?** Check `PROJECT_PLANS.md` for roadmap and architecture details.
