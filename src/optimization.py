"""
Inference Optimization & Performance Tuning
===========================================

Model quantization, pruning, batching, and optimization.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import time


class ModelOptimizer:
    """Optimize models for deployment."""
    
    @staticmethod
    def quantize_model(model: nn.Module, bits: int = 8) -> nn.Module:
        """
        Quantize model weights to INT8 or INT4.
        
        ~4x memory reduction, minimal accuracy loss.
        """
        return torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={torch.nn.Linear},
            dtype=torch.qint8
        )
    
    @staticmethod
    def prune_weights(model: nn.Module, sparsity: float = 0.3) -> nn.Module:
        """
        Magnitude-based weight pruning.
        Removes sparsity% of smallest weights.
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.utils.prune.l1_unstructured(
                    module, 'weight', amount=sparsity
                )
        
        return model
    
    @staticmethod
    def distill_model(teacher: nn.Module, student: nn.Module,
                     train_data: List[torch.Tensor],
                     temperature: float = 4.0,
                     epochs: int = 10) -> nn.Module:
        """
        Knowledge distillation: train smaller student model to mimic teacher.
        ~50% smaller with comparable accuracy.
        """
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        mse_loss = nn.MSELoss()
        
        teacher.eval()
        student.train()
        
        for epoch in range(epochs):
            for batch in train_data:
                with torch.no_grad():
                    teacher_logits = teacher(batch)
                
                student_logits = student(batch)
                
                # Distillation loss + task loss
                kl = kl_loss(
                    torch.log_softmax(student_logits / temperature, dim=1),
                    torch.softmax(teacher_logits / temperature, dim=1)
                )
                
                loss = kl * (temperature ** 2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return student
    
    @staticmethod
    def convert_to_onnx(model: nn.Module, dummy_input: torch.Tensor,
                       output_path: str) -> str:
        """Convert model to ONNX format for cross-platform deployment."""
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            opset_version=11,
            dynamic_axes={'input': {0: 'batch_size'}}
        )
        return output_path


class BatchProcessor:
    """Efficient batch processing and inference."""
    
    def __init__(self, model: nn.Module, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        self.inference_times = []
    
    def process_batches(self, data_loader) -> Tuple[torch.Tensor, List[float]]:
        """Process all data in batches and collect timing."""
        all_outputs = []
        timings = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                start = time.time()
                output = self.model(batch)
                elapsed = time.time() - start
                
                all_outputs.append(output)
                timings.append(elapsed)
        
        return torch.cat(all_outputs, dim=0), timings
    
    def get_throughput(self, timings: List[float]) -> Dict[str, float]:
        """Calculate throughput metrics."""
        total_time = sum(timings)
        avg_time = total_time / len(timings)
        
        return {
            'samples_per_second': self.batch_size / avg_time if avg_time > 0 else 0,
            'avg_batch_time_ms': avg_time * 1000,
            'total_time_s': total_time,
            'min_time_ms': min(timings) * 1000,
            'max_time_ms': max(timings) * 1000,
        }


class LatencyOptimizer:
    """Target specific latency requirements."""
    
    @staticmethod
    def optimize_for_latency(model: nn.Module, target_latency_ms: float) -> Dict[str, any]:
        """Find best optimization strategy to meet latency target."""
        results = {}
        
        # Test different optimization levels
        strategies = [
            ("fp32_baseline", lambda m: m),
            ("fp16_mixed", lambda m: m.half()),
            ("int8_quantized", ModelOptimizer.quantize_model),
            ("pruned_30%", lambda m: ModelOptimizer.prune_weights(m, 0.3)),
            ("pruned_50%", lambda m: ModelOptimizer.prune_weights(m, 0.5)),
        ]
        
        for name, optimization_fn in strategies:
            try:
                optimized = optimization_fn(model)
                # Benchmark
                dummy_input = torch.randn(1, 512)
                start = time.time()
                for _ in range(100):
                    _ = optimized(dummy_input)
                latency = (time.time() - start) / 100 * 1000  # ms
                
                results[name] = {
                    'latency_ms': latency,
                    'meets_target': latency <= target_latency_ms,
                }
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results


if __name__ == "__main__":
    # Example: Optimize model for deployment
    model = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 14)
    )
    
    # Quantize
    quantized = ModelOptimizer.quantize_model(model)
    print(f"Quantized model: {quantized}")
    
    # Test latency
    optimizer = LatencyOptimizer()
    results = optimizer.optimize_for_latency(model, target_latency_ms=50)
    print(f"\nLatency optimization results:")
    for strategy, metrics in results.items():
        if 'latency_ms' in metrics:
            print(f"  {strategy}: {metrics['latency_ms']:.2f}ms")
