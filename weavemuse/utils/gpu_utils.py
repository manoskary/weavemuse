"""
GPU and VRAM detection utilities for WeaveMuse.

This module provides functions to detect GPU capabilities and automatically select
appropriate model configurations based on available hardware resources.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class VRAMTier(Enum):
    """VRAM tier categories for model selection."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class GPUInfo:
    """Information about GPU and system capabilities."""
    has_cuda: bool
    gpu_count: int
    gpu_name: str
    total_vram_gb: float
    free_vram_gb: float
    system_ram_gb: float
    vram_tier: VRAMTier
    recommended_model_id: str
    use_quantization: bool
    device_map: str

def get_system_memory_gb() -> float:
    """Get system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        logger.warning("psutil not available, assuming 8GB system RAM")
        return 8.0

def get_gpu_memory_info() -> Tuple[float, float]:
    """
    Get GPU memory information in GB.
    
    Returns:
        Tuple[float, float]: (total_vram_gb, free_vram_gb)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    try:
        # Get GPU 0 properties
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / (1024**3)
        
        # Get current memory usage
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        cached = torch.cuda.memory_reserved(0) / (1024**3)
        free_memory = total_memory - max(allocated, cached)
        
        return total_memory, free_memory
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return 0.0, 0.0

def get_gpu_name() -> str:
    """Get the name of the primary GPU."""
    if not torch.cuda.is_available():
        return "No GPU"
    
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "Unknown GPU"

def categorize_vram_tier(total_vram_gb: float, free_vram_gb: float) -> VRAMTier:
    """
    Categorize VRAM into tiers based on total and available VRAM.
    
    Args:
        total_vram_gb: Total VRAM in GB
        free_vram_gb: Free VRAM in GB
    
    Returns:
        VRAMTier: The appropriate tier
    """
    # Use the more conservative of total or free VRAM for categorization
    effective_vram = min(total_vram_gb, free_vram_gb + 2.0)  # Add 2GB buffer for free VRAM
    
    if effective_vram >= 45.0:  # High-end GPUs (RTX 4090, A100, etc.)
        return VRAMTier.HIGH
    elif effective_vram >= 20.0:  # Mid-range GPUs (RTX 4070, RTX 3080, etc.)
        return VRAMTier.MEDIUM
    else:  # Low-end GPUs or limited VRAM
        return VRAMTier.LOW

def get_model_for_tier(tier: VRAMTier) -> str:
    """
    Get the recommended model ID for a given VRAM tier.
    
    Args:
        tier: The VRAM tier
    
    Returns:
        str: Model ID
    """
    model_mapping = {
        VRAMTier.LOW: "Qwen/Qwen2.5-Coder-7B-Instruct",
        VRAMTier.MEDIUM: "Qwen/Qwen2.5-Coder-14B-Instruct", 
        VRAMTier.HIGH: "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    }
    return model_mapping[tier]

def should_use_quantization(tier: VRAMTier, has_cuda: bool) -> bool:
    """
    Determine if quantization should be used based on tier and CUDA availability.
    
    Args:
        tier: The VRAM tier
        has_cuda: Whether CUDA is available
    
    Returns:
        bool: Whether to use quantization
    """
    if not has_cuda:
        return False  # No quantization for CPU
    
    # Use quantization for low and medium tiers to maximize efficiency
    return tier in [VRAMTier.LOW, VRAMTier.MEDIUM, VRAMTier.HIGH]

def get_device_map(has_cuda: bool, force_cpu: bool = False) -> str:
    """
    Get the appropriate device map configuration.
    
    Args:
        has_cuda: Whether CUDA is available
        force_cpu: Whether to force CPU mode
    
    Returns:
        str: Device map configuration
    """
    if force_cpu or not has_cuda:
        return "cpu"
    return "auto"

def test_gpu_allocation(test_size_mb: int = 100) -> bool:
    """
    Test if GPU allocation works by trying to allocate a small tensor.
    
    Args:
        test_size_mb: Size of test tensor in MB
    
    Returns:
        bool: Whether GPU allocation works
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Calculate tensor size for test_size_mb
        elements = (test_size_mb * 1024 * 1024) // 4  # 4 bytes per float32
        test_tensor = torch.zeros(elements, device='cuda', dtype=torch.float32)
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except (torch.OutOfMemoryError, RuntimeError) as e:
        logger.warning(f"GPU allocation test failed: {e}")
        torch.cuda.empty_cache()
        return False

def detect_gpu_capabilities(force_cpu: bool = False) -> GPUInfo:
    """
    Detect GPU capabilities and recommend model configuration.
    
    Args:
        force_cpu: Whether to force CPU mode regardless of GPU availability
    
    Returns:
        GPUInfo: Comprehensive GPU and system information
    """
    logger.info("🔍 Detecting GPU capabilities...")
    
    # System memory
    system_ram_gb = get_system_memory_gb()
    
    # Check if we should force CPU mode
    if force_cpu:
        logger.info("🔒 CPU mode forced by user")
        has_cuda = False
        gpu_count = 0
        gpu_name = "Forced CPU"
        total_vram_gb = 0.0
        free_vram_gb = 0.0
        vram_tier = VRAMTier.LOW
    else:
        # GPU detection
        has_cuda = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if has_cuda else 0
        gpu_name = get_gpu_name()
        
        if has_cuda:
            # Test actual GPU allocation
            if not test_gpu_allocation():
                logger.warning("⚠️  GPU allocation test failed, falling back to CPU")
                has_cuda = False
                gpu_count = 0
                gpu_name = "GPU allocation failed"
                total_vram_gb = 0.0
                free_vram_gb = 0.0
                vram_tier = VRAMTier.LOW
            else:
                total_vram_gb, free_vram_gb = get_gpu_memory_info()
                vram_tier = categorize_vram_tier(total_vram_gb, free_vram_gb)
        else:
            total_vram_gb = 0.0
            free_vram_gb = 0.0
            vram_tier = VRAMTier.LOW
    
    # Model selection based on tier
    recommended_model_id = get_model_for_tier(vram_tier)
    use_quantization = should_use_quantization(vram_tier, has_cuda)
    device_map = get_device_map(has_cuda, force_cpu)
    
    gpu_info = GPUInfo(
        has_cuda=has_cuda,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        total_vram_gb=total_vram_gb,
        free_vram_gb=free_vram_gb,
        system_ram_gb=system_ram_gb,
        vram_tier=vram_tier,
        recommended_model_id=recommended_model_id,
        use_quantization=use_quantization,
        device_map=device_map
    )
    
    # Log the detection results
    logger.info(f"🖥️  System RAM: {system_ram_gb:.1f}GB")
    if has_cuda:
        logger.info(f"🎮 GPU: {gpu_name}")
        logger.info(f"💾 VRAM: {total_vram_gb:.1f}GB total, {free_vram_gb:.1f}GB free")
        logger.info(f"📊 VRAM Tier: {vram_tier.value.upper()}")
        logger.info(f"🤖 Recommended Model: {recommended_model_id}")
        logger.info(f"⚡ Quantization: {'Enabled' if use_quantization else 'Disabled'}")
    else:
        logger.info("🔄 Running in CPU mode")
        logger.info(f"🤖 CPU Model: {recommended_model_id}")
    
    return gpu_info

def get_quantization_config(gpu_info: GPUInfo) -> Optional[Any]:
    """
    Get the appropriate quantization configuration based on GPU info.
    
    Args:
        gpu_info: GPU information from detect_gpu_capabilities
    
    Returns:
        Optional quantization config for transformers
    """
    if not gpu_info.use_quantization or not gpu_info.has_cuda:
        return None
    
    try:
        from transformers import BitsAndBytesConfig
        
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("✅ 4-bit quantization configuration created")
        return config
    except ImportError:
        logger.warning("⚠️  BitsAndBytesConfig not available, quantization disabled")
        return None

def print_gpu_summary(gpu_info: GPUInfo) -> None:
    """
    Print a formatted summary of GPU capabilities.
    
    Args:
        gpu_info: GPU information to summarize
    """
    print("\n" + "="*60)
    print("🎵 WeaveMuse GPU Detection Summary")
    print("="*60)
    print(f"🖥️  System RAM: {gpu_info.system_ram_gb:.1f}GB")
    
    if gpu_info.has_cuda:
        print(f"🎮 GPU: {gpu_info.gpu_name}")
        print(f"💾 VRAM: {gpu_info.total_vram_gb:.1f}GB total, {gpu_info.free_vram_gb:.1f}GB free")
        print(f"📊 VRAM Tier: {gpu_info.vram_tier.value.upper()}")
        print(f"⚡ Quantization: {'✅ Enabled' if gpu_info.use_quantization else '❌ Disabled'}")
    else:
        print("🔄 Mode: CPU Only")
    
    print(f"🤖 Selected Model: {gpu_info.recommended_model_id}")
    print(f"🔧 Device Map: {gpu_info.device_map}")
    print("="*60 + "\n")

# Environment variable for forcing CPU mode
def check_force_cpu_mode() -> bool:
    """Check if CPU mode is forced via environment variable."""
    return os.environ.get("WEAVEMUSE_FORCE_CPU", "").lower() in ["1", "true", "yes"]

if __name__ == "__main__":
    # Test the GPU detection when run directly
    force_cpu = check_force_cpu_mode()
    gpu_info = detect_gpu_capabilities(force_cpu=force_cpu)
    print_gpu_summary(gpu_info)
