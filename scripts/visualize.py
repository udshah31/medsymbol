#!/usr/bin/env python3
"""
Interpretability & Visualization Module
========================================
Provides SHAP explanations, attention visualization, and proof formatting.

Includes:
- SHAP force plots for feature importance
- Attention heatmaps from multimodal fusion
- Proof generator logical chain visualization
- Feature importance ranking
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple
import json
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
from datetime import datetime

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ============================================================================
# SHAP Explainability
# ============================================================================

class SHAPExplainer:
    """
    Compute SHAP values for model predictions.
    Requires: pip install shap
    """
    
    @staticmethod
    def has_shap() -> bool:
        """Check if SHAP is available."""
        return HAS_SHAP
    
    @staticmethod
    def compute_shap_values(model, input_features: np.ndarray, background_samples: int = 100) -> dict:
        """
        Compute SHAP values for given inputs.
        
        Args:
            model: Neural network model
            input_features: Feature matrix (N, D)
            background_samples: Number of background samples
        
        Returns: Dictionary with SHAP values and base value
        """
        if not HAS_SHAP:
            raise ImportError("SHAP not installed. Run: pip install shap")
        
        try:
            # Use kernel SHAP (model-agnostic)
            explainer = shap.KernelExplainer(
                lambda x: model.predict(x).numpy(),
                background_data=input_features[:background_samples]
            )
            
            shap_values = explainer.shap_values(input_features)
            
            return {
                "shap_values": shap_values,
                "base_value": explainer.expected_value,
                "features": input_features
            }
        except Exception as e:
            print(f"[!] SHAP computation failed: {e}")
            return None
    
    @staticmethod
    def plot_force(shap_values: dict, prediction_index: int = 0, 
                   feature_names: List[str] = None, output_path: str = None) -> None:
        """
        Create SHAP force plot visualization.
        
        Args:
            shap_values: Dict from compute_shap_values()
            prediction_index: Which prediction to visualize
            feature_names: Names of features
            output_path: Path to save plot
        """
        if not HAS_SHAP or shap_values is None:
            print("[!] Cannot create SHAP force plot")
            return
        
        try:
            plt.figure(figsize=(12, 6))
            shap.force_plot(
                shap_values["base_value"],
                shap_values["shap_values"][prediction_index],
                shap_values["features"][prediction_index],
                feature_names=feature_names,
                matplotlib=True,
            )
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                print(f"[✓] SHAP force plot saved to {output_path}")
            
            plt.close()
        except Exception as e:
            print(f"[!] SHAP force plot failed: {e}")


# ============================================================================
# Attention Visualization
# ============================================================================

class AttentionVisualizer:
    """Visualize multimodal fusion attention weights and flows."""
    
    @staticmethod
    def visualize_attention_heatmap(attention_weights: torch.Tensor,
                                    modality_names: List[str] = None,
                                    output_path: str = None) -> None:
        """
        Create attention weight heatmap.
        
        Args:
            attention_weights: (seq_len, seq_len) or (batch, seq, seq) tensor
            modality_names: Names of modalities
            output_path: Path to save plot
        """
        if attention_weights.dim() > 2:
            # Average over batch
            attention_weights = attention_weights.mean(dim=0)
        
        # Convert to numpy
        attn_np = attention_weights.detach().cpu().numpy()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_np, annot=True, fmt='.2f', cmap='viridis', 
                   xticklabels=modality_names or range(attn_np.shape[1]),
                   yticklabels=modality_names or range(attn_np.shape[0]))
        plt.title("Multimodal Fusion Attention Weights")
        plt.xlabel("Values (What)")
        plt.ylabel("Queries (Who)")
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[✓] Attention heatmap saved to {output_path}")
        
        plt.close()
    
    @staticmethod
    def visualize_attention_on_image(image: np.ndarray,
                                     attention_map: np.ndarray,
                                     output_path: str = None) -> None:
        """
        Overlay attention heatmap on image.
        
        Args:
            image: Input image (H, W) or (H, W, 3)
            attention_map: Attention weights (h, w)
            output_path: Path to save plot
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(image, cmap='gray')
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Attention map
        ax2.imshow(attention_map, cmap='hot')
        ax2.set_title("Attention Map")
        ax2.axis('off')
        
        # Overlay
        ax3.imshow(image, cmap='gray', alpha=0.6)
        ax3.imshow(attention_map, cmap='hot', alpha=0.4)
        ax3.set_title("Attention Overlay")
        ax3.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[✓] Attention overlay saved to {output_path}")
        
        plt.close()


# ============================================================================
# Feature Importance
# ============================================================================

class FeatureImportanceVisualizer:
    """Visualize feature importance and contributions."""
    
    @staticmethod
    def plot_feature_importance(importances: Dict[str, float],
                                top_k: int = 15,
                                output_path: str = None) -> None:
        """
        Create feature importance bar chart.
        
        Args:
            importances: Dict mapping feature names to importance scores
            top_k: Number of top features to display
            output_path: Path to save plot
        """
        # Sort and get top-k
        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_k]
        features, scores = zip(*sorted_items)
        
        # Create bar chart
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(features)), scores, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel("Importance Score")
        plt.title(f"Top {top_k} Most Important Features")
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(score, i, f" {score:.3f}", va='center')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[✓] Feature importance plot saved to {output_path}")
        
        plt.close()


# ============================================================================
# Proof Generator & Logical Chain Visualization
# ============================================================================

class ProofVisualizer:
    """Format and visualize logical reasoning chains from Module 7."""
    
    @staticmethod
    def format_proof_chain(diagnosis: str, patient_data: dict,
                          verification_results: dict) -> str:
        """
        Format verification results as human-readable logical proof.
        
        Args:
            diagnosis: Predicted diagnosis
            patient_data: Patient information dict
            verification_results: Results from symbolic verifier
        
        Returns: Formatted proof string
        """
        proof = f"""
╔═══════════════════════════════════════════════════════════════════╗
║              SYMBOLIC VERIFICATION PROOF                         ║
╚═══════════════════════════════════════════════════════════════════╝

DIAGNOSIS: {diagnosis}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATIENT CONTEXT:
  • Age: {patient_data.get('age', 'Unknown')}
  • Sex: {patient_data.get('sex', 'Unknown')}  
  • Comorbidities: {', '.join(patient_data.get('comorbidities', ['None']))}
  • Symptoms: {', '.join(patient_data.get('symptoms', ['Not reported']))}

VERIFICATION CHECKS:
"""
        
        for check, result in verification_results.items():
            if check in ['fully_consistent', 'confidence_score']:
                continue
            
            status_icon = "✓" if result == "PASS" else ("⚠" if result == "WARN" else "✗")
            proof += f"  {status_icon} {check.replace('_', ' ').title()}: {result}\n"
        
        # Overall assessment
        if verification_results.get('fully_consistent'):
            proof += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            proof += "CONCLUSION: ✓ All checks PASSED - Diagnosis is CONSISTENT\n"
        else:
            proof += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            proof += "CONCLUSION: ⚠ Some checks FAILED - Review diagnosis carefully\n"
        
        confidence = verification_results.get('confidence_score', 0)
        proof += f"CONFIDENCE: {confidence:.1%}\n"
        proof += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        return proof
    
    @staticmethod
    def create_proof_flowchart(diagnosis: str, predictions: Dict[str, float],
                               output_path: str = None) -> None:
        """
        Create decision flowchart visualization.
        
        Args:
            diagnosis: Top diagnosis
            predictions: Dict of diagnosis probabilities
            output_path: Path to save plot
        """
        # Sort predictions
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Top diagnosis box
        top_dx, top_prob = sorted_preds[0]
        ax.text(0.5, 0.9, f"DIAGNOSIS: {top_dx}", 
               ha='center', va='center', fontsize=16, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', linewidth=2))
        
        # Probability ranking
        ax.text(0.5, 0.75, "DIFFERENTIAL DIAGNOSIS (Ranked by Probability):", 
               ha='center', va='center', fontsize=12, fontweight='bold')
        
        for i, (dx, prob) in enumerate(sorted_preds):
            y_pos = 0.70 - (i * 0.12)
            
            # Bar
            ax.barh(y_pos, prob, height=0.08, left=0.1, color=plt.cm.RdYlGn(prob))
            
            # Label
            ax.text(0.08, y_pos, f"{i+1}.", va='center', ha='right', fontsize=10, fontweight='bold')
            ax.text(0.12, y_pos, f"{dx}", va='center', ha='left', fontsize=10)
            ax.text(0.62, y_pos, f"{prob:.1%}", va='center', ha='left', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"[✓] Proof flowchart saved to {output_path}")
        
        plt.close()
    
    @staticmethod
    def export_proof_json(diagnosis: str, patient_data: dict,
                         verification_results: dict,
                         predictions: Dict[str, float],
                         output_path: str = None) -> dict:
        """
        Export complete proof as JSON for downstream processing.
        
        Args:
            diagnosis: Primary diagnosis
            patient_data: Patient information
            verification_results: Verification check results
            predictions: All disease probabilities
            output_path: Path to save JSON
        
        Returns: JSON-serializable proof dictionary
        """
        proof_data = {
            "timestamp": str(datetime.now()),
            "diagnosis": {
                "primary": diagnosis,
                "confidence": verification_results.get('confidence_score', 0),
                "consistent": verification_results.get('fully_consistent', False)
            },
            "patient": patient_data,
            "verification": {k: v for k, v in verification_results.items() 
                           if k not in ['confidence_score', 'fully_consistent']},
            "predictions": {k: float(v) for k, v in predictions.items()},
            "reasoning": ProofVisualizer.format_proof_chain(diagnosis, patient_data, verification_results)
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(proof_data, f, indent=2)
            print(f"[✓] Proof JSON saved to {output_path}")
        
        return proof_data


# ============================================================================
# Medical Report Generation
# ============================================================================

class MedicalReportGenerator:
    """Generate clinical reports from predictions."""
    
    @staticmethod
    def generate_report(patient_id: str, diagnosis: str,
                       confidence: float, predictions: Dict[str, float],
                       patient_data: dict = None) -> str:
        """
        Generate formatted clinical report.
        
        Args:
            patient_id: Patient identifier
            diagnosis: Primary diagnosis
            confidence: Confidence score (0-1)
            predictions: All disease probabilities
            patient_data: Patient demographic/clinical data
        
        Returns: Formatted report string
        """
        report = f"""
╔═══════════════════════════════════════════════════════════════════╗
║                   MEDICAL REPORT                                 ║
║              MedSymbol AI Analysis System v0.1                   ║
╚═══════════════════════════════════════════════════════════════════╝

REPORT GENERATION DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
PATIENT ID: {patient_id}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PRIMARY DIAGNOSIS:
  • Finding: {diagnosis}
  • Confidence: {confidence:.1%}
  • Reliability: {'✓ High' if confidence > 0.80 else ('⚠ Moderate' if confidence > 0.60 else '✗ Low')}

DIFFERENTIAL DIAGNOSES (Top 5):
"""
        
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (dx, prob) in enumerate(sorted_preds, 1):
            report += f"  {i}. {dx:.<30} {prob:.1%}\n"
        
        if patient_data:
            report += "\nPATIENT CONTEXT:\n"
            report += f"  • Age: {patient_data.get('age', 'Unknown')}\n"
            report += f"  • Sex: {patient_data.get('sex', 'Unknown')}\n"
            report += f"  • Comorbidities: {', '.join(patient_data.get('comorbidities', ['None']))}\n"
        
        report += "\n" + "━"*67 + "\n"
        report += "DISCLAIMER: This report is generated by an AI system and should be\n"
        report += "reviewed by qualified medical professionals before clinical use.\n"
        report += "━"*67 + "\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    print("[*] Interpretability module loaded")
    print(f"[*] SHAP available: {SHAPExplainer.has_shap()}")
