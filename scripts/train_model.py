import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MedSymbolModel
from src.utils.data_loader import NIHCXR14Dataset

# Define collate_fn at module level to make it picklable
def collate_fn(batch):
    inputs_list, labels_list, patient_data_list = zip(*batch)
    
    # Batch tensors
    batched_inputs = {
        'vision': torch.stack([x['vision'] for x in inputs_list]),
        'history': torch.stack([x['history'] for x in inputs_list]),
        'input_ids': torch.stack([x['input_ids'] for x in inputs_list]),
        'attention_mask': torch.stack([x['attention_mask'] for x in inputs_list]),
        'tabular': torch.stack([x['tabular'] for x in inputs_list]),
    }
    
    batched_labels = torch.stack(labels_list)
    return batched_inputs, batched_labels, patient_data_list

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train(args):
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    # Dataset & DataLoader
    csv_file = "data/raw/nih_cxr14/Data_Entry_2017.csv"
    img_dir = "data/raw/nih_cxr14/images"
    
    print("[*] Initializing Dataset...")
    dataset = NIHCXR14Dataset(csv_file=csv_file, img_dir=img_dir, split='train')
    
    # Use the module-level collate_fn and disable multiprocessing workers
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # Initialize Core Model
    print("[*] Initializing MedSymbol Model...")
    model = MedSymbolModel(
        num_diagnoses=len(dataset.DISEASE_LABELS), 
        tabular_input_dim=10, 
        history_input_dim=5,
        tau_low=0.3,
        tau_high=1.5,
        pretrained=not args.no_pretrained
    ).to(device)

    # Optimizer & Loss for Multilabel classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training Loop
    print("[*] Starting Training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, labels, patient_metadata in pbar:
            # Move inputs to device
            inputs_device = {k: v.to(device) for k, v in inputs.items()}
            labels_device = labels.to(device)

            # Forward
            optimizer.zero_grad()
            logits = model(inputs_device) # Differentiable block (Modules 1-3)
            
            # Loss & Backward
            loss = criterion(logits, labels_device)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

        # Optional: Run validation and log metrics here
        
        # Save Checkpoint
        os.makedirs("experiments", exist_ok=True)
        torch.save(model.state_dict(), f"experiments/medsymbol_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MedSymbol Model")
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--no_pretrained', action='store_true', help='Use randomly initialized weights instead of pretrained (useful for offline/smoke tests)')
    
    args = parser.parse_args()
    train(args)
