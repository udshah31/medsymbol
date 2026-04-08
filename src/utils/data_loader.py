import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class NIHCXR14Dataset(Dataset):
    """
    Dataset loader for NIH ChestX-ray14 dataset.
    Handles Multimodal data:
    - Vision: X-ray images (ViT-B/16 expects 224x224)
    - History: Extracted from demographics (Age, Sex, etc.)
    - Text/Tabular: Mocked for NIH (as the raw dataset only has images/demographics)
    """
    
    DISEASE_LABELS = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
        'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    def __init__(self, csv_file: str, img_dir: str, split: str = 'train', transform=None):
        self.img_dir = img_dir
        self.split = split
        
        # Load CSV
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
            # Basic preprocessing/filtering could happen here
        else:
            # Fallback for scaffolding/testing if dataset isn't fully extracted
            print(f"Warning: {csv_file} not found. Generating dummy metadata.")
            self.df = pd.DataFrame({
                'Image Index': [f'test_{i:04d}.png' for i in range(100)],
                'Finding Labels': ['Pneumonia' if i%2==0 else 'No Finding' for i in range(100)],
                'Patient Age': np.random.randint(10, 90, 100),
                'Patient Sex': ['M' if i%2==0 else 'F' for i in range(100)]
            })

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            # Repeat grayscale to 3 channels for ViT
            transforms.Grayscale(num_output_channels=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def _get_labels(self, label_str: str) -> torch.Tensor:
        """Convert 'Pneumonia|Effusion' to a multi-hot tensor."""
        labels = torch.zeros(len(self.DISEASE_LABELS))
        if label_str != 'No Finding':
            for i, d in enumerate(self.DISEASE_LABELS):
                if d in label_str:
                    labels[i] = 1.0
        return labels

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row.get('Image Index', f'test_{idx:04d}.png') if 'Image Index' in self.df.columns else f'test_{idx:04d}.png'
        img_path = os.path.join(self.img_dir, img_name)
        
        # --- 1. Vision ---
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        else:
            # Dummy tensor if image is missing
            image = torch.randn(3, 224, 224)

        # --- 2. History (Age normalized (max 120), Sex one-hot) ---
        age_val = row.get('Patient Age', int(idx % 80 + 18)) if 'Patient Age' in self.df.columns else int(idx % 80 + 18)
        sex_val = row.get('Patient Sex', 'M' if idx % 2 == 0 else 'F') if 'Patient Sex' in self.df.columns else ('M' if idx % 2 == 0 else 'F')
        
        age = min(age_val, 120) / 120.0
        sex_m = 1.0 if sex_val == 'M' else 0.0
        sex_f = 1.0 if sex_val == 'F' else 0.0
        # Dummy filling the rest of the 5-dim history vector
        history = torch.tensor([age, sex_m, sex_f, 0.0, 0.0], dtype=torch.float32)

        # --- 3. Text (Mocked empty BioBERT inputs for now) ---
        input_ids = torch.randint(0, 2000, (128,), dtype=torch.long)
        attention_mask = torch.ones(128, dtype=torch.long)

        # --- 4. Tabular (Mocked missing lab values) ---
        tabular = torch.zeros(10, dtype=torch.float32)

        inputs = {
            'vision': image,
            'history': history,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tabular': tabular
        }
        
        label_str = row.get('Finding Labels', 'No Finding') if 'Finding Labels' in self.df.columns else 'No Finding'
        labels = self._get_labels(label_str)
        
        # Extra metadata for symbolic verifier
        patient_data = {
            'age': int(age_val),
            'sex': sex_val,
            'identifier': str(img_name)
        }

        return inputs, labels, patient_data
