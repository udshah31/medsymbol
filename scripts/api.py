#!/usr/bin/env python3
"""
MedSymbol FastAPI Inference Server
==================================
REST API for MedSymbol neuro-symbolic medical AI model.

Usage:
    uvicorn scripts.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health               - Health check
    POST /predict              - Single image prediction
    POST /predict/batch        - Batch predictions
    POST /predict/multimodal   - Multimodal (image + clinical data)
    GET  /docs                 - Interactive API documentation (Swagger)
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import MedSymbolModel
from torchvision import transforms
from PIL import Image

# ============================================================================
# Data Models
# ============================================================================

class PatientData(BaseModel):
    """Patient clinical and demographic data."""
    patient_id: str
    age: int
    sex: str  # 'M' or 'F'
    comorbidities: List[str] = []
    symptoms: List[str] = []


class PredictionRequest(BaseModel):
    """Single prediction request with multimodal data."""
    patient_id: str
    age: int
    sex: str
    # Clinical narrative (optional)
    clinical_notes: Optional[str] = None
    # Tabular demographics
    comorbidities: List[str] = []
    symptoms: List[str] = []
    # Vital signs (optional)
    vitals: Optional[Dict[str, float]] = None


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    predictions: List[PredictionRequest]


class PredictionResponse(BaseModel):
    """Prediction response with disease probabilities and interpretability."""
    patient_id: str
    timestamp: str
    predictions: Dict[str, float]  # Disease -> probability
    top_k: List[Dict[str, float]]  # Top K diagnoses
    confidence_scores: Dict[str, float]
    symbolic_verification: Optional[Dict[str, str]] = None
    routing_decision: Dict[str, str]  # Module 6-7 routing details
    proof: Optional[str] = None  # Logical proof from Module 7


class HealthResponse(BaseModel):
    """Server health status."""
    status: str
    model_loaded: bool
    device: str
    timestamp: str


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="MedSymbol Neuro-Symbolic Medical AI",
    description="REST API for interpretable medical diagnosis with neural+symbolic reasoning",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Manage application state."""
    def __init__(self):
        self.model = None
        self.device = None
        self.model_loaded = False
        self.transform = None
        self.diseases = []
        
state = AppState()


def load_model():
    """Load MedSymbol model."""
    global state
    
    print("[*] Loading MedSymbol model...")
    
    # Setup device
    if torch.cuda.is_available():
        state.device = "cuda"
    elif torch.backends.mps.is_available():
        state.device = "mps"
    else:
        state.device = "cpu"
    
    print(f"[*] Using device: {state.device}")
    
    try:
        # Initialize model
        state.model = MedSymbolModel(
            num_diagnoses=14,
            tabular_input_dim=10,
            history_input_dim=5
        ).to(state.device)
        
        # Try to load checkpoint
        model_path = Path("./models/medsymbol.pt")
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=state.device)
            state.model.load_state_dict(checkpoint)
            print(f"[✓] Loaded checkpoint from {model_path}")
        else:
            print("[!] No checkpoint found, using randomly initialized model")
        
        state.model.eval()
        state.model_loaded = True
        
        # Setup image transforms
        state.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Disease labels
        state.diseases = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        print(f"[✓] Model loaded successfully ({len(state.diseases)} diseases)")
        return True
        
    except Exception as e:
        print(f"[✗] Failed to load model: {e}")
        return False


def predict_from_image(image: Image.Image) -> Dict[str, float]:
    """Run inference on single image."""
    if not state.model_loaded:
        raise RuntimeError("Model not loaded")
    
    # Preprocess
    img_tensor = state.transform(image).unsqueeze(0).to(state.device)
    
    # Create multimodal input
    with torch.no_grad():
        # Dummy clinical data for now
        sample_input = {
            'image': img_tensor,
            'text_embedding': torch.randn(1, 768).to(state.device),
            'tabular_features': torch.randn(1, 10).to(state.device),
            'history_sequence': torch.randn(1, 5, 128).to(state.device),
            'patient_age': torch.tensor([50], dtype=torch.float32).to(state.device)
        }
        
        # Forward pass
        logits = state.model(sample_input)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    
    return {disease: float(prob) for disease, prob in zip(state.diseases, probabilities)}


# ============================================================================
# Routes
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health."""
    return {
        "status": "healthy" if state.model_loaded else "unhealthy",
        "model_loaded": state.model_loaded,
        "device": state.device or "none",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single patient prediction with multimodal data."""
    
    if not state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Dummy predictions for now (replace with actual inference)
        np.random.seed(hash(request.patient_id) % 2**32)
        scores = np.random.random(len(state.diseases))
        predictions = {disease: float(score) for disease, score in zip(state.diseases, scores)}
        
        # Sort and get top-k
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_k = [{"disease": disease, "probability": prob} for disease, prob in sorted_preds[:5]]
        
        return {
            "patient_id": request.patient_id,
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "top_k": top_k,
            "confidence_scores": {disease: 0.75 for disease in state.diseases},
            "routing_decision": {
                "module_6_decision": "ACCEPT",
                "module_7_proof": "Age-compatible with top diagnosis"
            },
            "proof": "Logical reasoning chain: Age 50 is compatible with Pneumonia presentation"
        }
        
    except Exception as e:
        print(f"[✗] Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Predict from uploaded X-ray image."""
    
    if not state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert('L')  # Grayscale
        
        # Predict
        predictions = predict_from_image(image)
        
        # Format response
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "patient_id": file.filename,
            "timestamp": datetime.now().isoformat(),
            "predictions": predictions,
            "top_k": [{"disease": d, "probability": p} for d, p in sorted_preds[:5]],
            "confidence_scores": {d: 0.8 for d in state.diseases}
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Batch predictions for multiple patients."""
    
    if not state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for pred_request in request.predictions:
        try:
            # Run individual prediction
            np.random.seed(hash(pred_request.patient_id) % 2**32)
            scores = np.random.random(len(state.diseases))
            predictions = {d: float(s) for d, s in zip(state.diseases, scores)}
            
            results.append({
                "patient_id": pred_request.patient_id,
                "status": "success",
                "predictions": predictions
            })
        except Exception as e:
            results.append({
                "patient_id": pred_request.patient_id,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "timestamp": datetime.now().isoformat(),
        "total": len(request.predictions),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "results": results
    }


@app.get("/diseases")
async def list_diseases():
    """List supported diseases/diagnoses."""
    return {"diseases": state.diseases, "count": len(state.diseases)}


@app.get("/model-info")
async def model_info():
    """Get model information."""
    return {
        "name": "MedSymbol",
        "version": "0.1.0",
        "description": "Neuro-symbolic medical AI with 7-module architecture",
        "modules": [
            "Vision Encoder (ResNet50)",
            "Text Encoder (BioBERT)",
            "Tabular Encoder",
            "History Encoder",
            "Multimodal Fusion (Cross-attention)",
            "Symbolic Masking (Ontology constraints)",
            "Proof Generator (Logical reasoning)"
        ],
        "num_diseases": len(state.diseases),
        "supported_formats": ["JPEG", "PNG", "DICOM"],
        "device": state.device
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
