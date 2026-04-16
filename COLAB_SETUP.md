# Running MedSymbol in Google Colab

Complete setup guide for training and inference in Colab with GPU support.

## 📋 Prerequisites

- Google account with Colab access
- Optional: Kaggle account (for NIH CXR-14 dataset)
- Optional: PhysioNet credentials (for MIMIC datasets)

---

## 🚀 Quick Start (Copy-Paste into Colab)

Create a new Colab notebook and run these cells in order:

### Cell 1: Mount Google Drive & Clone Repository

```python
# Mount Google Drive for data persistence
from google.colab import drive
drive.mount('/content/drive')

# Clone MedSymbol repository
!cd /content && git clone https://github.com/udshah31/medsymbol.git
!cd /content/medsymbol && git pull origin master

# Check Colab environment
!python --version
!nvidia-smi  # Check GPU availability
```

### Cell 2: Install Dependencies

```python
# Install PyTorch with CUDA support (pre-installed in Colab)
# Colab comes with torch, torchvision, transformers pre-installed
# Just upgrade to latest versions if needed

!pip install -q --upgrade torch torchvision transformers

# Install MedSymbol dependencies
%cd /content/medsymbol
!pip install -q -e .

# Additional Colab-specific packages
!pip install -q kaggle  # For dataset download
!pip install -q google-colab-transfer  # Optional: faster file transfer
```

### Cell 3: Verify Environment

```python
import sys
sys.path.insert(0, '/content/medsymbol')

# Run verification script
%cd /content/medsymbol
!python scripts/verify_env.py

# Quick import check
from src.symbolic.disease_ontology import DiseaseOntology
from src.encoders.vision import VisionEncoder
from src.error_handling import ErrorHandler
print("✓ All imports successful!")
```

### Cell 4: Download & Prepare Data

#### Option A: NIH CXR-14 (Open Access, Recommended)

```python
import os
from pathlib import Path

# Create data directory in Google Drive
DATA_DIR = '/content/drive/MyDrive/medsymbol_data'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(f'{DATA_DIR}/raw/nih_cxr14', exist_ok=True)

# Download NIH dataset metadata (required CSV)
!cd {DATA_DIR}/raw/nih_cxr14 && wget -q https://nihcc.box.com/shared/static/itsreall706qrcf160r3l7uc3llrzekk -O Data_Entry_2017.csv

print("✓ NIH CXR-14 metadata downloaded")
print(f"✓ Data directory: {DATA_DIR}")

# Note: Full image download (112GB) - use Kaggle if needed
# Instructions below...
```

#### Option B: Download via Kaggle (NIH Full Dataset)

```python
# Upload your Kaggle credentials
from google.colab import files

print("Upload your Kaggle API key (kaggle.json)")
print("Get it from: https://www.kaggle.com/settings/account")
files.upload()

# Setup Kaggle
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download NIH CXR-14 dataset
DATA_DIR = '/content/drive/MyDrive/medsymbol_data'
os.makedirs(f'{DATA_DIR}/raw', exist_ok=True)

!kaggle datasets download -d nih-chest-xrays/chest-xray-14 -p {DATA_DIR}/raw --unzip

print("✓ Dataset downloaded to Google Drive (persistent)")
```

#### Option C: Use Preprocessed Dataset (Fastest)

```python
# If you have preprocessed data in Drive, symlink it
DATA_DIR = '/content/drive/MyDrive/medsymbol_data'
!ln -s {DATA_DIR}/processed /content/medsymbol/data/processed

# Or download from cloud storage if available
!gsutil -m cp gs://your-bucket/processed_data.tar.gz /tmp/
!cd /tmp && tar -xzf processed_data.tar.gz -C /content/medsymbol/data/
```

### Cell 5: Test Basic Functionality

```python
import torch
import numpy as np

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Test VisionEncoder
from src.encoders.vision import VisionEncoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vision_encoder = VisionEncoder().to(device)

# Create dummy input (batch=1, channels=1, height=224, width=224)
dummy_xray = torch.randn(1, 1, 224, 224).to(device)
features = vision_encoder(dummy_xray)

print(f"✓ Vision encoder output shape: {features.shape}")
print(f"✓ Device: {device}")
```

### Cell 6: Run Tests

```python
import subprocess

%cd /content/medsymbol

# Run full test suite
result = subprocess.run(
    ["python", "-m", "pytest", "tests/test_all.py", "-v", "--tb=short"],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)
```

### Cell 7: Train/Run Inference

```python
import json
from pathlib import Path

# Load config
config_path = '/content/medsymbol/configs/default.yaml'

# Example: Run inference on sample data
from src.clinical_support import ClinicalRecommender, DecisionSupportFormatter

# Generate example explanation
explanation = ClinicalRecommender.generate_explanation(
    disease="pneumonia",
    probability=0.87,
    findings=["consolidation_RLL", "fever", "elevated_wbc"],
    severity="moderate"
)

# Format for display
formatter = DecisionSupportFormatter()
report = formatter.format_report(explanation)
print(report)
```

---

## 🔧 Advanced Configuration

### Use Colab's TPU (Optional)

```python
# In Colab: Edit notebook → Runtime → Change runtime type → TPU
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(f"TPU device: {device}")

# Models can then train on TPU
```

### Enable Wandb for Experiment Tracking

```python
import wandb

# Authenticate with Wandb
!wandb login

# In your training code:
wandb.init(project="medsymbol", name="colab-experiment")
wandb.log({"loss": 0.5, "accuracy": 0.92})
```

### Save Checkpoints to Google Drive

```python
import shutil

def save_checkpoint(model, name):
    """Save model to Google Drive"""
    save_dir = '/content/drive/MyDrive/medsymbol_checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    path = f'{save_dir}/{name}.pt'
    torch.save(model.state_dict(), path)
    print(f"✓ Saved to {path}")

def load_checkpoint(name):
    """Load model from Google Drive"""
    path = f'/content/drive/MyDrive/medsymbol_checkpoints/{name}.pt'
    model = VisionEncoder()  # Initialize model
    model.load_state_dict(torch.load(path))
    return model
```

---

## 📊 Performance Tips

### 1. Mixed Precision Training (Faster, Less Memory)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    loss = model(batch)
    
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Gradient Checkpointing (More Memory-Efficient)

```python
model.gradient_checkpointing_enable()  # For transformers
```

### 3. Distributed Data Parallel (Multiple GPUs - if available)

```python
# Colab usually has 1 GPU, but if multiple:
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### 4. Optimize Batch Size

```python
# Start small, increase until OOM
BATCH_SIZE = 32  # Adjust based on GPU memory

# Monitor memory
!nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"

```python
# Clear cache
import torch
torch.cuda.empty_cache()

# Reduce batch size
BATCH_SIZE = 16  # Instead of 32

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
```

### "ModuleNotFoundError: No module named 'src'"

```python
import sys
sys.path.insert(0, '/content/medsymbol')
```

### "FileNotFoundError: data/raw/..."

```python
# Symlink Drive data
import os
DATA_DIR = '/content/drive/MyDrive/medsymbol_data'
os.symlink(DATA_DIR, '/content/medsymbol/data')
```

### Slow Data Loading

```python
# Use num_workers=0 for Colab (Windows/WSL limitation)
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=0  # Change to 2 if on Linux Colab
)
```

### Git Authentication Issues

```python
# If git clone fails, use HTTPS and cache credentials
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"

# Or generate GitHub token and use:
!git clone https://your_token@github.com/udshah31/medsymbol.git
```

---

## 📁 Directory Structure in Colab

```
/content/
├── medsymbol/           # Cloned repo
│   ├── src/
│   ├── configs/
│   ├── tests/
│   ├── scripts/
│   └── data/ → symlink to Drive
│
/content/drive/MyDrive/
├── medsymbol_data/      # Large datasets (persistent)
│   ├── raw/
│   │   └── nih_cxr14/
│   ├── processed/
│   └── ontology/
│
├── medsymbol_checkpoints/  # Model checkpoints
└── medsymbol_experiments/  # Results/logs
```

---

## 🚄 Complete Training Example

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-4

# Dummy data (replace with real dataset)
X = torch.randn(1000, 1, 224, 224)
y = torch.randint(0, 14, (1000,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Model
from src.encoders.vision import VisionEncoder

model = VisionEncoder().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

# Training loop
from tqdm import tqdm

for epoch in range(EPOCHS):
    total_loss = 0
    
    for batch_idx, (x, y_true) in enumerate(tqdm(loader)):
        x, y_true = x.to(device), y_true.to(device)
        
        # Forward
        logits = model(x)
        loss = criterion(logits, y_true)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    if (epoch + 1) % 1 == 0:
        torch.save(model.state_dict(), 
                   f'/content/drive/MyDrive/medsymbol_checkpoints/epoch_{epoch+1}.pt')
        print(f"✓ Checkpoint saved")

print("✓ Training complete!")
```

---

## 🔐 Security Best Practices

1. **Never commit credentials** to Git
2. **Use Secrets** for API keys in Colab
3. **Revoke Kaggle tokens** after use
4. **Don't share Colab notebooks** with sensitive data
5. **Use separate Drive folders** for data, checkpoints, experiments

---

## 📞 Support

- GitHub Issues: https://github.com/udshah31/medsymbol/issues
- Email: [contact info]
- Check logs: `!cat /content/medsymbol/logs/*.log`

---

**Happy Training! 🎉**
