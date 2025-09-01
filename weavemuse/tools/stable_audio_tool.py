"""
Stable Audio Tool - High-quality audio generation using Stable Audio models.
"""

import logging
import tempfile
import os
from typing import Optional, Dict, Any, Union
from pathlib import Path
from smolagents.tools import Tool  # type: ignore
import torch
import torchaudio


try:    
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    STABLE_AUDIO_AVAILABLE = True
    print("✅ Stable Audio dependencies loaded successfully!")
except ImportError as e:
    STABLE_AUDIO_AVAILABLE = False
    import_error = str(e)
    print(f"❌ Stable Audio dependencies failed: {e}")    
    get_pretrained_model = None
    generate_diffusion_cond = None

from .base_tools import ManagedDiffusersTool

logger = logging.getLogger(__name__)


class StableAudioTool(ManagedDiffusersTool):
    """
    Tool for high-quality audio generation using Stable Audio.
    
    This tool can:
    - Generate high-quality audio from text descriptions
    - Create music in various styles and genres
    - Produce sound effects and ambient audio
    - Generate audio of specific lengths
    
    Now with lazy loading and VRAM management!
    """
    
    # Class attributes required by smolagents
    name = "stable_audio"
    description = (
        "High-quality audio generation tool using Stable Audio models. "
        "Creates music, sound effects, and ambient audio from text descriptions "
        "with precise control over duration and style."
    )
    inputs = {
        "prompt": {
            "type": "string",
            "description": "Text description of the audio to generate"
        },
        "duration": {
            "type": "string",
            "description": "Duration in seconds (default: 30, max: 47)",
            "nullable": True
        },
        "steps": {
            "type": "string",
            "description": "Number of diffusion steps (default: 100, recommended: 50-200)",
            "nullable": True
        },
        "cfg_scale": {
            "type": "string",
            "description": "Classifier-free guidance scale (default: 7.0, range: 1.0-15.0)",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(
        self,
        device: str = "auto",
        model_id: str = "stabilityai/stable-audio-open-1.0",
        **kwargs
    ):
        """
        Initialize Stable Audio tool with lazy loading.
        
        Args:
            device: Device to run on ("auto", "cuda", "cpu")
            model_id: Stable Audio model ID
            **kwargs: Additional arguments
        """
        if not STABLE_AUDIO_AVAILABLE:
            raise ImportError(f"Stable Audio dependencies not available: {import_error}")
        
        # Stable Audio Open 1.0 is ~1.2B params, estimate VRAM usage
        estimated_vram = 5000.0  # Conservative estimate for diffusion model
        
        super().__init__(
            model_id=model_id,
            device=device,
            estimated_vram_mb=estimated_vram,
            torch_dtype="float16" if device == "cuda" else "float32",
            priority=3,  # Medium priority tool
            **kwargs
        )
        
        logger.info(f"Stable Audio tool initialized (lazy loading enabled)")
    
    def _load_model(self) -> Dict[str, Any]:
        """
        Load the Stable Audio model and conditioning.
        
        Returns:
            Dictionary containing model, sample_rate, and sample_size
        """
        if not STABLE_AUDIO_AVAILABLE:
            raise ImportError(f"Stable Audio dependencies not available: {import_error}")
            
        logger.info(f"Loading Stable Audio model: {self.model_id}")
        
        # Load model
        if get_pretrained_model is None:
            raise ImportError("get_pretrained_model not available")
            
        model, model_config = get_pretrained_model(self.model_id)
        
        # Move to device
        model = model.to(self.device)
        
        # Extract sample rate and size from config
        sample_rate = model_config.get("sample_rate", 44100)
        sample_size = model_config.get("sample_size", sample_rate * 47)  # Max 47 seconds
        
        logger.info(f"Stable Audio model loaded successfully on {self.device}")
        logger.info(f"Sample rate: {sample_rate} Hz, Max duration: {sample_size / sample_rate:.1f}s")
        
        return {
            "model": model,
            "model_config": model_config,
            "sample_rate": sample_rate,
            "sample_size": sample_size
        }
    
    def _call_model(
        self,
        model: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Generate audio using Stable Audio model.
        
        Args:
            model: Dictionary containing loaded components
            **kwargs: Arguments passed from forward() including prompt, duration, steps, cfg_scale
            
        Returns:
            Path to generated audio file
        """
        try:
            # Extract components
            audio_model = model["model"]
            model_config = model["model_config"]
            sample_rate = model["sample_rate"]
            sample_size = model["sample_size"]
            
            # Extract parameters from kwargs
            prompt = kwargs.get("prompt", "")
            duration = float(kwargs.get("duration", "30"))
            steps = int(kwargs.get("steps", "100"))
            cfg_scale = float(kwargs.get("cfg_scale", "7.0"))
            
            # Validate parameters
            max_duration = sample_size / sample_rate
            duration = min(duration, max_duration)
            steps = max(10, min(steps, 1000))
            cfg_scale = max(1.0, min(cfg_scale, 15.0))
            
            logger.info(f"Generating audio: '{prompt}' ({duration}s, {steps} steps, CFG: {cfg_scale})")
            
            # Setup conditioning
            conditioning = {
                "prompt": [prompt],
                "seconds_start": [0],
                "seconds_total": [duration]
            }
            
            # Generate audio
            if generate_diffusion_cond is None:
                raise ImportError("generate_diffusion_cond not available")
                
            # Generate in evaluation mode
            audio_model.eval()
            if torch is None:
                raise ImportError("torch not available")
            with torch.no_grad():
                output = generate_diffusion_cond(
                    model=audio_model,
                    steps=steps,
                    cfg_scale=int(cfg_scale),  # Convert to int
                    conditioning=conditioning,
                    sample_size=int(duration * sample_rate),
                    sample_rate=sample_rate,
                    device=self.device
                )
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
            
            if torchaudio is None:
                raise ImportError("torchaudio not available")
                
            # Ensure output is in correct format
            if output.dim() == 1:
                output = output.unsqueeze(0)  # Add channel dimension if mono
            elif output.dim() == 3:
                output = output.squeeze(0)  # Remove batch dimension if present
                
            # Normalize and save
            output = output.clamp(-1, 1)  # Ensure values are in valid range
            torchaudio.save(temp_path, output.cpu(), sample_rate)
            
            logger.info(f"Audio generated successfully: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return f"Error: {str(e)}"
    
    def _unload_model(self, model: Dict[str, Any]) -> None:
        """
        Unload the Stable Audio model components.
        
        Args:
            model: Dictionary containing model components
        """
        try:
            # Clean up model
            if "model" in model:
                audio_model = model["model"]
                del model["model"]
                del audio_model
            
            # Clean up config
            if "model_config" in model:
                del model["model_config"]
            
            # Force cleanup
            import gc
            gc.collect()
            
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        except Exception as e:
            logger.warning(f"Error during Stable Audio cleanup: {e}")

    def forward(self, prompt, cfg_scale=5, steps=50, duration=30):
        """
        Forward method to generate audio.
        
        Args:
            cfg_scale: Classifier-free guidance scale
            steps: Number of diffusion steps
            prompt: Text description of the audio
            duration: Duration in seconds
            
        Returns:
            Path to generated audio file
        """
        kwargs = {
            "cfg_scale": cfg_scale,
            "steps": steps,
            "prompt": prompt,
            "duration": duration
        }
        return super().forward(**kwargs)        