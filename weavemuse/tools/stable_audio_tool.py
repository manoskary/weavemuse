"""
Stable Audio Tool - High-quality audio generation using Stable Audio models.
"""

import logging
import tempfile
import os
import time
from typing import Optional, Dict, Any, Union
from pathlib import Path
from smolagents.tools import Tool  # type: ignore
import torch
import soundfile as sf
import torchaudio


try:    
    from diffusers.pipelines.stable_audio.pipeline_stable_audio import StableAudioPipeline
    STABLE_AUDIO_AVAILABLE = True
    print("✅ Stable Audio dependencies loaded successfully!")
except ImportError as e:
    STABLE_AUDIO_AVAILABLE = False
    import_error = str(e)
    print(f"❌ Stable Audio dependencies failed: {e}")    
    StableAudioPipeline = None

from .base_tools import ManagedDiffusersTool

logger = logging.getLogger(__name__)


class RemoteStableAudioTool(Tool):
    """
    Tool for high-quality audio generation using Stable Audio via diffusers.
    
    This tool can:
    - Generate high-quality audio from text descriptions
    - Create music in various styles and genres
    - Produce sound effects and ambient audio
    - Generate audio of specific lengths (up to 47 seconds)

    Uses the StableAudioPipeline Space for efficient inference without GPU dependencies.
    """
    # Class attributes required by smolagents
    name = "stable_audio"
    description = (
        "High-quality audio generation tool using Stable Audio models. "
        "Creates music, sound effects, and ambient audio from text descriptions "
        "with precise control over duration and style."
        "Returns the path to the generated audio file."
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
            "description": "Number of inference steps (default: 50, recommended: 20-50)",
            "nullable": True
        },
        "cfg_scale": {
            "type": "string",
            "description": "Guidance scale (default: 7.0, range: 1.0-15.0)",
            "nullable": True
        }
    }
    output_type = "string"
    def __init__(
        self,
        output_dir: str = "/tmp/stable_audio",
        space_id: str = "manoskary/stable-audio-open-1.0-music",
        **kwargs
    ):
        self.output_dir = output_dir
        self.space_id = space_id
        super().__init__(**kwargs)


    def forward(self, prompt, duration=None, steps=None, cfg_scale=None) -> str:
        """
        Generate audio using Stable Audio pipeline.
        
        Args:
            model: Dictionary containing loaded components
            **kwargs: Arguments passed from forward() including prompt, duration, steps, cfg_scale
            
        Returns:
            Path to generated audio file
        """
        try:
            from gradio_client import Client

            # Extract parameters from kwargs            
            duration = 30 if duration is None else float(duration)
            steps = 100 if steps is None else int(steps)
            cfg_scale = 7.0 if cfg_scale is None else float(cfg_scale)

            client = Client(
                self.space_id,
                hf_token=os.getenv("HF_TOKEN")
            )
            tmp_audio_path = client.predict(
                prompt=prompt,
                seconds_total=duration,
                steps=steps,
                cfg_scale=cfg_scale,
                api_name="/generate"
            )

            # Save audio to output_dir
            os.makedirs(self.output_dir, exist_ok=True)
            # Give the audio a timestamped filename            
            timestamp = int(time.time())
            output_filename = f"stable_audio_{timestamp}.wav"
            temp_path = os.path.join(self.output_dir, output_filename)            
            print(f"Audio saved: {temp_path}")

            # result is a path, move it to output_dir
            os.rename(tmp_audio_path, temp_path)

            logger.info(f"Audio generated successfully: {temp_path}")
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return f"Error: {str(e)}"


class StableAudioTool(ManagedDiffusersTool):
    """
    Tool for high-quality audio generation using Stable Audio via diffusers.
    
    This tool can:
    - Generate high-quality audio from text descriptions
    - Create music in various styles and genres
    - Produce sound effects and ambient audio
    - Generate audio of specific lengths (up to 47 seconds)
    
    Uses the diffusers StableAudioPipeline for efficient inference with VRAM management.
    """
    
    # Class attributes required by smolagents
    name = "stable_audio"
    description = (
        "High-quality audio generation tool using Stable Audio models. "
        "Creates music, sound effects, and ambient audio from text descriptions "
        "with precise control over duration and style."
        "Returns the path to the generated audio file."
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
            "description": "Number of inference steps (default: 50, recommended: 20-50)",
            "nullable": True
        },
        "cfg_scale": {
            "type": "string",
            "description": "Guidance scale (default: 7.0, range: 1.0-15.0)",
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
        self.output_dir = Path(kwargs.get("output_dir", "/tmp/stable_audio"))
        logger.info(f"Stable Audio tool initialized (lazy loading enabled)")
    
    def _load_model(self) -> Dict[str, Any]:
        """
        Load the Stable Audio model using diffusers pipeline.
        
        Returns:
            Dictionary containing pipeline and config
        """
        if not STABLE_AUDIO_AVAILABLE:
            raise ImportError(f"Stable Audio dependencies not available: {import_error}")
            
        logger.info(f"Loading Stable Audio model: {self.model_id}")
        
        # Load pipeline
        if StableAudioPipeline is None:
            raise ImportError("StableAudioPipeline not available")
        
        # Determine torch_dtype    
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        pipeline = StableAudioPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch_dtype
        )
        
        # Move to device
        pipeline = pipeline.to(self.device)
        
        # Get sampling rate from VAE
        sample_rate = pipeline.vae.sampling_rate
        
        logger.info(f"Stable Audio pipeline loaded successfully on {self.device}")
        logger.info(f"Sample rate: {sample_rate} Hz")
        
        return {
            "pipeline": pipeline,
            "sample_rate": sample_rate
        }
    
    def _call_model(
        self,
        model: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Generate audio using Stable Audio pipeline.
        
        Args:
            model: Dictionary containing loaded components
            **kwargs: Arguments passed from forward() including prompt, duration, steps, cfg_scale
            
        Returns:
            Path to generated audio file
        """
        try:
            # Extract components
            pipeline = model["pipeline"]
            sample_rate = model["sample_rate"]
            
            # Extract parameters from kwargs
            prompt = kwargs.get("prompt", "")
            duration = float(kwargs.get("duration", "30"))
            steps = int(kwargs.get("steps", "100"))
            cfg_scale = float(kwargs.get("cfg_scale", "7.0"))
            
            # Validate parameters
            max_duration = 47.0  # Stable Audio Open max duration
            duration = min(duration, max_duration)
            steps = max(10, min(steps, 50))
            cfg_scale = max(1.0, min(cfg_scale, 15.0))
            negative_prompt = "Low quality."
            
            logger.info(f"Generating audio: '{prompt}' ({duration}s, {steps} steps, CFG: {cfg_scale})")
            
            # Set up generator for reproducible results
            generator = torch.Generator(self.device).manual_seed(42)
            
            # Generate audio using the diffusers pipeline
            with torch.no_grad():
                print(f"Debug: About to call pipeline with:")
                print(f"  prompt: {prompt}")
                print(f"  num_inference_steps: {steps}")
                print(f"  guidance_scale: {cfg_scale}")
                print(f"  audio_end_in_s: {duration}")
                print(f"  device: {self.device}")
                
                try:
                    # Use the diffusers pipeline
                    result = pipeline(
                        prompt=prompt,
                        num_inference_steps=steps,
                        guidance_scale=cfg_scale,                                
                        audio_end_in_s=duration,
                        negative_prompt=negative_prompt,
                        num_waveforms_per_prompt=1,
                        generator=generator,
                    )
                    
                    # Extract audio from result
                    audio = result.audios[0]  # Get first (and only) waveform
                    
                    print(f"Debug: pipeline returned audio with shape: {audio.shape}")
                        
                except Exception as e:
                    print(f"Debug: Exception in pipeline: {e}")
                    print(f"Debug: Exception type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    raise
            
            # Save audio to output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Give the audio a timestamped filename            
            timestamp = int(time.time())
            output_filename = f"stable_audio_{timestamp}.wav"
            temp_path = self.output_dir / output_filename
                
            # Convert to numpy format for soundfile
            output = audio.T.float().cpu().numpy()
            
            # Save using soundfile
            sf.write(str(temp_path), output, sample_rate)
            
            logger.info(f"Audio generated successfully: {temp_path}")
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return f"Error: {str(e)}"
    
    def _unload_model(self, model: Dict[str, Any]) -> None:
        """
        Unload the Stable Audio pipeline components.
        
        Args:
            model: Dictionary containing model components
        """
        try:
            # Clean up pipeline
            if "pipeline" in model:
                pipeline = model["pipeline"]
                del model["pipeline"]
                del pipeline
            
            # Force cleanup
            import gc
            gc.collect()
            
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        except Exception as e:
            logger.warning(f"Error during Stable Audio cleanup: {e}")

    def forward(self, prompt, duration=None, steps=None, cfg_scale=None):
        """
        Forward method to generate audio.
        
        Args:
            prompt: Text description of the audio
            duration: Duration in seconds (default: 30, max: 47)
            steps: Number of diffusion steps (default: 100, recommended: 50-200)
            cfg_scale: Classifier-free guidance scale (default: 7.0, range: 1.0-15.0)
            
        Returns:
            Path to generated audio file
        """
        kwargs = {
            "prompt": prompt,
            "duration": duration or "30",
            "steps": steps or "100", 
            "cfg_scale": cfg_scale or "7.0"
        }
        return super().forward(**kwargs)        