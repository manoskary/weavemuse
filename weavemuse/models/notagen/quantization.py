"""
NotaGen Quantization Utilities
=============================
Tools for quantizing the NotaGen model to reduce size and inference time.
"""

import torch
import torch.quantization as quantization
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class NotaGenQuantizer:
    """Quantization utilities for NotaGen model."""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize quantizer.
        
        Args:
            model: NotaGen model instance
            device: Device to run quantization on
        """
        self.model = model
        self.device = device
        self.original_size = None
        self.quantized_size = None
    
    def get_model_size(self, model_path=None):
        """Get model size in MB."""
        if model_path and os.path.exists(model_path):
            return os.path.getsize(model_path) / (1024 * 1024)
        else:
            # Calculate from parameters
            param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
            return (param_size + buffer_size) / (1024 * 1024)
    
    def dynamic_quantization(self, qconfig_spec=None):
        """
        Apply dynamic quantization (weights only).
        Best for models where inference time is dominated by loading weights.
        
        Args:
            qconfig_spec: Quantization configuration
            
        Returns:
            Quantized model
        """
        logger.info("Applying dynamic quantization to NotaGen...")
        
        # Default quantization config for linear layers
        if qconfig_spec is None:
            qconfig_spec = {
                torch.nn.Linear: torch.quantization.default_dynamic_qconfig
            }
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.model.cpu(),  # Move to CPU for quantization
            qconfig_spec,
            dtype=torch.qint8
        )
        
        self.original_size = self.get_model_size()
        logger.info(f"Original model size: {self.original_size:.2f} MB")
        
        # Estimate quantized size (approximately 4x smaller for int8)
        self.quantized_size = self.original_size / 4
        logger.info(f"Estimated quantized size: {self.quantized_size:.2f} MB")
        
        return quantized_model
    
    def static_quantization(self, calibration_data=None):
        """
        Apply static quantization (requires calibration data).
        Better accuracy but requires representative data.
        
        Args:
            calibration_data: Data for calibration
            
        Returns:
            Quantized model
        """
        logger.info("Applying static quantization to NotaGen...")
        
        # Prepare model for static quantization
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Fuse modules if possible (improves performance)
        try:
            fused_model = torch.quantization.fuse_modules(self.model, [])
        except:
            logger.warning("Module fusion not possible, proceeding without fusion")
            fused_model = self.model
        
        # Prepare for quantization
        prepared_model = torch.quantization.prepare(fused_model)
        
        # Calibration (if data provided)
        if calibration_data is not None:
            logger.info("Running calibration...")
            prepared_model.eval()
            with torch.no_grad():
                for data in calibration_data:
                    prepared_model(data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        return quantized_model
    
    def bitsandbytes_quantization(self):
        """
        Apply 8-bit quantization using bitsandbytes.
        Good for CUDA inference with minimal accuracy loss.
        
        Returns:
            Configuration for model loading
        """
        try:
            from transformers import BitsAndBytesConfig
            
            logger.info("Creating 8-bit quantization config for NotaGen...")
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_threshold=6.0
            )
            
            return quantization_config
            
        except ImportError:
            logger.error("bitsandbytes not available for 8-bit quantization")
            return None
    
    def save_quantized_model(self, quantized_model, save_path):
        """Save quantized model to disk."""
        logger.info(f"Saving quantized model to {save_path}")
        
        # Create directory if it doesn't exist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the quantized model
        torch.save(quantized_model.state_dict(), save_path)
        
        # Check saved file size
        saved_size = os.path.getsize(save_path) / (1024 * 1024)
        logger.info(f"Saved quantized model size: {saved_size:.2f} MB")
        
        return saved_size
    
    def benchmark_model(self, model, test_input, num_runs=10):
        """
        Benchmark model inference time.
        
        Args:
            model: Model to benchmark
            test_input: Test input data
            num_runs: Number of benchmark runs
            
        Returns:
            Average inference time in seconds
        """
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(test_input)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(test_input)
        
        avg_time = (time.time() - start_time) / num_runs
        logger.info(f"Average inference time: {avg_time:.4f} seconds")
        
        return avg_time


def quantize_notagen_model(model_path, output_path, method='dynamic'):
    """
    Quantize NotaGen model.
    
    Args:
        model_path: Path to original model weights
        output_path: Path to save quantized model
        method: Quantization method ('dynamic', 'static', '8bit')
        
    Returns:
        Path to quantized model
    """
    from ..inference import model  # Import the loaded model
    
    quantizer = NotaGenQuantizer(model)
    
    if method == 'dynamic':
        quantized_model = quantizer.dynamic_quantization()
    elif method == 'static':
        # For static quantization, we'd need calibration data
        logger.warning("Static quantization requires calibration data")
        return None
    elif method == '8bit':
        # For 8-bit, return config instead of model
        config = quantizer.bitsandbytes_quantization()
        return config
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    # Save quantized model
    saved_size = quantizer.save_quantized_model(quantized_model, output_path)
    
    return output_path, saved_size


if __name__ == "__main__":
    # Example usage
    original_weights = "weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth"
    quantized_weights = "weights_notagenx_quantized_int8.pth"
    
    if os.path.exists(original_weights):
        result = quantize_notagen_model(original_weights, quantized_weights, method='dynamic')
        if result:
            print(f"Quantized model saved to: {result[0]}")
            print(f"Size reduction: {result[1]:.2f} MB")
    else:
        print("Original model weights not found")
