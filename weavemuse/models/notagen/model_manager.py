"""
NotaGen Model Management - Handles different model variants and automatic selection.
"""
import os
import torch
import logging
from typing import Tuple, Optional, Dict, Any
from huggingface_hub import hf_hub_download, HfApi
from pathlib import Path

logger = logging.getLogger(__name__)

class NotaGenModelManager:
    """
    Manages NotaGen model variants (quantized vs unquantized) and automatic selection.
    """
    
    # Model configurations
    MODELS = {
        "original": {
            "hf_repo": "emmanouil-karystinaios/NotaGenX",
            "filename": "weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth",
            "device": "auto",  # Will use GPU if available
            "quantized": False,
            "memory_req_gb": 4.0,
            "description": "Original full-precision NotaGen model"
        },
        "quantized_int8": {
            "hf_repo": "emmanouil-karystinaios/NotaGenX",
            "filename": "weights_notagenx_quantized_int8.pth",
            "device": "cpu",  # Quantized models typically run on CPU
            "quantized": True,
            "memory_req_gb": 1.5,
            "description": "INT8 quantized NotaGen model for resource-constrained systems"
        },
        "quantized_fp16": {
            "hf_repo": "emmanouil-karystinaios/NotaGenX", 
            "filename": "weights_notagenx_quantized_fp16.pth",
            "device": "auto",
            "quantized": True,
            "memory_req_gb": 2.0,
            "description": "FP16 quantized NotaGen model - good balance of speed and quality"
        }
    }
    
    def __init__(self, cache_dir: str = ".cache", force_model: Optional[str] = None):
        """
        Initialize the model manager.
        
        Args:
            cache_dir: Directory to cache downloaded models
            force_model: Force use of specific model variant ("original", "quantized_int8", "quantized_fp16")
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.force_model = force_model
        
    def get_system_info(self) -> Dict[str, float]:
        """Get system resource information."""
        try:
            import psutil
            system_ram = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            system_ram = 8.0  # Default assumption
            
        gpu_ram = 0.0
        if torch.cuda.is_available():
            try:
                gpu_ram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                pass
                
        return {
            "system_ram_gb": system_ram,
            "gpu_ram_gb": gpu_ram,
            "cuda_available": torch.cuda.is_available()
        }
    
    def select_best_model(self) -> str:
        """
        Automatically select the best model variant based on system resources.
        
        Returns:
            str: Model variant name
        """
        # If user forced a specific model, use it
        if self.force_model:
            if self.force_model in self.MODELS:
                logger.info(f"Using forced model: {self.force_model}")
                return self.force_model
            else:
                logger.warning(f"Forced model '{self.force_model}' not found, auto-selecting...")
        
        # Check environment variable
        env_model = os.environ.get('NOTAGEN_MODEL_VARIANT')
        if env_model and env_model in self.MODELS:
            logger.info(f"Using model from environment: {env_model}")
            return env_model
            
        # Auto-select based on system resources
        system_info = self.get_system_info()
        logger.info(f"System info: {system_info}")
        
        # Decision logic:
        # 1. If no CUDA or low GPU memory, use INT8 quantized
        # 2. If medium memory, use FP16 quantized  
        # 3. If high memory, use original
        
        if not system_info["cuda_available"] or system_info["gpu_ram_gb"] < 4.0:
            model_choice = "quantized_int8"
            reason = "Limited GPU resources"
        elif system_info["system_ram_gb"] < 8.0 or system_info["gpu_ram_gb"] < 8.0:
            model_choice = "quantized_fp16"
            reason = "Medium system resources"
        else:
            model_choice = "original"
            reason = "Sufficient resources for full model"
            
        logger.info(f"Selected model '{model_choice}' - {reason}")
        return model_choice
    
    def download_model(self, model_variant: str) -> str:
        """
        Download a specific model variant.
        
        Args:
            model_variant: Model variant to download
            
        Returns:
            str: Path to downloaded model file
        """
        if model_variant not in self.MODELS:
            raise ValueError(f"Unknown model variant: {model_variant}")
            
        config = self.MODELS[model_variant]
        local_path = self.cache_dir / config["filename"]
        
        # Check if already downloaded
        if local_path.exists():
            logger.info(f"Model already cached: {local_path}")
            return str(local_path)
            
        # Download from Hugging Face Hub
        try:
            logger.info(f"Downloading {model_variant} from {config['hf_repo']}")
            downloaded_path = hf_hub_download(
                repo_id=config["hf_repo"],
                filename=config["filename"],
                cache_dir=str(self.cache_dir),
                force_download=False
            )
            
            # Create symlink for easier access
            if not local_path.exists():
                local_path.symlink_to(downloaded_path)
                
            logger.info(f"✅ Model downloaded: {local_path}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    def get_model_config(self, model_variant: str) -> Dict[str, Any]:
        """Get configuration for a specific model variant."""
        if model_variant not in self.MODELS:
            raise ValueError(f"Unknown model variant: {model_variant}")
        return self.MODELS[model_variant].copy()
    
    def prepare_model(self, model_variant: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare a model for use (download if needed, return config).
        
        Args:
            model_variant: Specific variant to use, or None for auto-selection
            
        Returns:
            Tuple of (model_path, model_config)
        """
        if model_variant is None:
            model_variant = self.select_best_model()
            
        model_path = self.download_model(model_variant)
        model_config = self.get_model_config(model_variant)
        
        return model_path, model_config

# Global instance for easy access
model_manager = NotaGenModelManager()

def get_optimal_notagen_model() -> Tuple[str, Dict[str, Any]]:
    """
    Get the optimal NotaGen model for the current system.
    
    Returns:
        Tuple of (model_path, model_config)
    """
    return model_manager.prepare_model()
