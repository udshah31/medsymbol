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
    
    Dataset Structure:
    - data/raw/nih_cxr14/
        - Data_Entry_2017_v2020.csv (metadata)
        - images/
            - 00000001_000.png
            - 00000002_001.png
            - ... (112,120 total)
    """
    
    DISEASE_LABELS = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
        'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]

    def __init__(self, csv_file: str, img_dir: str, split: str = 'train', transform=None, seed: int = 42):
        """
        Args:
            csv_file: Path to Data_Entry_2017_v2020.csv
            img_dir: Path to directory containing images/
            split: 'train' (70%), 'val' (15%), or 'test' (15%)
            transform: Optional torchvision transforms
            seed: Random seed for reproducible splits
        """
        self.img_dir = img_dir
        self.split = split
        self.seed = seed
        self.transform = transform
        
        # Load CSV
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
            print(f"[*] Loaded {len(self.df)} samples from {csv_file}")
            
            # Stratified split by most common finding
            self._apply_split()
            print(f"[*] Split '{split}': {len(self.df)} samples")
        else:
            # Fallback for scaffolding/testing if dataset isn't fully extracted
            print(f"Warning: {csv_file} not found. Generating dummy metadata.")
            self.df = pd.DataFrame({
                'Image Index': [f'test_{i:04d}.png' for i in range(100)],
                'Finding Labels': ['Pneumonia' if i%2==0 else 'No Finding' for i in range(100)],
                'Patient Age': np.random.randint(10, 90, 100),
                'Patient Sex': ['M' if i%2==0 else 'F' for i in range(100)]
            })
            self._apply_split()
    
    def _apply_split(self):
        """Apply train/val/test stratified split."""
        np.random.seed(self.seed)
        
        # Create stratification key based on primary finding
        def get_primary_finding(labels_str):
            if pd.isna(labels_str) or labels_str == 'No Finding':
                return 'No Finding'
            return labels_str.split('|')[0]  # First label is primary
        
        self.df['primary_finding'] = self.df.get('Finding Labels', 'No Finding').apply(get_primary_finding)
        
        # Perform stratified split per finding
        splits = []
        for finding in self.df['primary_finding'].unique():
            subset = self.df[self.df['primary_finding'] == finding]
            n = len(subset)
            indices = np.random.permutation(n)
            
            train_end = int(0.7 * n)
            val_end = int(0.85 * n)
            
            subset_train = subset.iloc[indices[:train_end]]
            subset_val = subset.iloc[indices[train_end:val_end]]
            subset_test = subset.iloc[indices[val_end:]]
            
            if self.split == 'train':
                splits.append(subset_train)
            elif self.split == 'val':
                splits.append(subset_val)
            elif self.split == 'test':
                splits.append(subset_test)
        
        self.df = pd.concat(splits, ignore_index=True) if splits else self.df

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
        
        # Get image filename - handles both flat and structured directories
        if 'Image Index' in self.df.columns:
            img_name = row['Image Index']
        else:
            img_name = f'test_{idx:04d}.png'
        
        # Try multiple possible paths for image location
        possible_paths = [
            os.path.join(self.img_dir, img_name),
            os.path.join(self.img_dir, 'images', img_name),
        ]
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        # --- 1. Vision ---
        if img_path:
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except Exception as e:
                print(f"[!] Failed to load {img_path}: {e}")
                image = torch.randn(3, 224, 224)
        else:
            # Dummy tensor if image is missing
            image = torch.randn(3, 224, 224)

        # --- 2. History (Age normalized (max 120), Sex one-hot) ---
        age_val = int(row.get('Patient Age', int(idx % 80 + 18)))
        sex_val = str(row.get('Patient Sex', 'M' if idx % 2 == 0 else 'F')).strip().upper()
        
        # Ensure valid sex
        if sex_val not in ['M', 'F']:
            sex_val = 'M' if idx % 2 == 0 else 'F'
        
        age = min(max(age_val, 0), 120) / 120.0  # Clamp to [0, 120]
        sex_m = 1.0 if sex_val == 'M' else 0.0
        sex_f = 1.0 if sex_val == 'F' else 0.0
        
        # 5-dim history: [age_norm, sex_M, sex_F, bmi_proxy, smoking_proxy]
        bmi_proxy = np.random.randn() * 0.1  # Dummy BMI proxy
        smoking_proxy = np.random.rand()  # Dummy smoking proxy
        history = torch.tensor([age, sex_m, sex_f, bmi_proxy, smoking_proxy], dtype=torch.float32)

        # --- 3. Text (Mocked BioBERT inputs) ---
        # In production, would extract text from radiology reports
        input_ids = torch.randint(0, 2000, (128,), dtype=torch.long)
        attention_mask = torch.ones(128, dtype=torch.long)

        # --- 4. Tabular (Mocked lab values) ---
        # In production, would include: WBC, O2_sat, RBC, CRP, etc.
        # Normalized to [0, 1]
        tabular = torch.rand(10, dtype=torch.float32)  # 10 common lab features

        inputs = {
            'vision': image,
            'history': history,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tabular': tabular
        }
        
        # Extract and parse labels
        label_str = str(row.get('Finding Labels', 'No Finding'))
        labels = self._get_labels(label_str)
        
        # Extra metadata for symbolic verifier
        patient_data = {
            'age': age_val,
            'sex': sex_val,
            'identifier': str(img_name),
            'findings': label_str,
        }

        return inputs, labels, patient_data
