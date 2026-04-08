import torch
import argparse
from src.model import MedSymbolModel
from src.utils.data_loader import NIHCXR14Dataset
from torch.utils.data import DataLoader

def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"[*] Using device: {device}")

    # Load Dataset
    csv_file = "data/raw/nih_cxr14/Data_Entry_2017.csv"
    img_dir = "data/raw/nih_cxr14/images"
    
    dataset = NIHCXR14Dataset(csv_file=csv_file, img_dir=img_dir, split='val')
    
    def collate_fn(batch):
        inputs_list, labels_list, patient_data_list = zip(*batch)
        batched_inputs = {
            'vision': torch.stack([x['vision'] for x in inputs_list]),
            'history': torch.stack([x['history'] for x in inputs_list]),
            'input_ids': torch.stack([x['input_ids'] for x in inputs_list]),
            'attention_mask': torch.stack([x['attention_mask'] for x in inputs_list]),
            'tabular': torch.stack([x['tabular'] for x in inputs_list]),
        }
        batched_labels = torch.stack(labels_list)
        return batched_inputs, batched_labels, patient_data_list

    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize Model
    model = MedSymbolModel(
        num_diagnoses=len(dataset.DISEASE_LABELS),
        tabular_input_dim=10,
        history_input_dim=5,
        tau_low=0.3,
        tau_high=1.5
    ).to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"[*] Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model.eval()

    print("\n[*] Running Inference Pipeline (Modules 1-7)...")
    print("=" * 80)

    with torch.no_grad():
        for batch_idx, (inputs, labels, patient_data) in enumerate(val_loader):
            if batch_idx >= 1:  # Just run on first batch for demo
                break

            inputs_device = {k: v.to(device) for k, v in inputs.items()}

            # Run full inference pipeline
            results = model.predict(
                inputs=inputs_device,
                patient_data=patient_data,
                diagnosis_map={i: d for i, d in enumerate(dataset.DISEASE_LABELS)}
            )

            # Display results
            for i, result in enumerate(results):
                print(f"\nPatient {i+1}:")
                print(f"  Patient ID: {patient_data[i]['identifier']}")
                print(f"  Age: {patient_data[i]['age']}, Sex: {patient_data[i]['sex']}")
                print(f"  Entropy: {result['entropy']:.4f}")
                print(f"  Routing Path: {result['path_taken']}")
                
                if result['diagnoses']:
                    print(f"  Top Diagnoses:")
                    for diag in result['diagnoses']:
                        print(f"    - {diag['diagnosis']}: {diag['probability']:.4f}")
                
                if result['verification']:
                    print(f"  Symbolic Verification Results:")
                    for diagnosis, checks in result['verification'].items():
                        print(f"    {diagnosis}:")
                        for check, status in checks.items():
                            print(f"      - {check}: {status}")

    print("\n" + "=" * 80)
    print("[✓] Inference Complete - All Modules (1-7) Executed Successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with Full MedSymbol Pipeline")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    
    args = parser.parse_args()
    inference(args)
