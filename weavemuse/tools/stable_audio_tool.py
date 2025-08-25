"""
Stable Audio Tool - High-quality audio generation using Stable Audio Open.
"""

import logging
import tempfile
import os
from typing import Optional, Dict, Any, Union
from pathlib import Path
from smolagents.tools import Tool

try:
    import torch
    import torchaudio
    import soundfile as sf
    from diffusers import StableAudioPipeline
    import numpy as np
    STABLE_AUDIO_AVAILABLE = True
    print("✅ Stable Audio dependencies loaded successfully!")
except ImportError as e:
    STABLE_AUDIO_AVAILABLE = False
    import_error = str(e)
    print(f"❌ Stable Audio dependencies failed: {e}")
    torch = None
    torchaudio = None
    sf = None
    StableAudioPipeline = None


logger = logging.getLogger(__name__)


class StableAudioTool(Tool):
    """
    Tool for high-quality audio generation using Stable Audio Open.
    
    This tool can:
    - Generate audio from text descriptions
    - Create music in various styles and genres
    - Generate sound effects and ambient sounds
    - Control audio duration and quality
    """
    
    # Class attributes required by smolagents
    name = "stable_audio"
    description = (
        "Generates high-quality audio and music from text descriptions. "
        "Can create various musical styles, sound effects, and ambient sounds. "
        "Supports duration control and produces 44.1kHz stereo audio."
    )
    inputs = {
        "prompt": {
            "type": "string", 
            "description": "Text description of the audio to generate"
        },
        "duration": {
            "type": "string", 
            "description": "Duration in seconds (default: 10, max: 47)",
            "nullable": True
        },
        "negative_prompt": {
            "type": "string",
            "description": "What to avoid in generation (optional)",
            "nullable": True
        },
        "steps": {
            "type": "string",
            "description": "Number of inference steps (default: 20, max: 50)",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, device: str = "auto", model_id: str = "stabilityai/stable-audio-open-1.0"):
        self.device = self._get_device(device)
        self.model_id = model_id
        self.model = None
        self.model_config = None
        self.pipeline = None
        
        super().__init__()
        
        self._initialize_model()
    
    def _get_device(self, device: str) -> str:
        """Get the appropriate device for the model."""
        if device == "auto":
            if torch and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _initialize_model(self):
        """Initialize the Stable Audio model."""
        try:
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Stable Audio model: {self.model_id}")
            
            # Use diffusers pipeline (stable-audio-tools not available)
            try:
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
                self.pipeline = StableAudioPipeline.from_pretrained(
                    self.model_id, 
                    torch_dtype=torch_dtype
                )
                self.pipeline = self.pipeline.to(self.device)
                logger.info("Stable Audio model loaded with diffusers")
                return
            except Exception as e:
                logger.warning(f"Could not load with diffusers: {e}")
            
            logger.error("Failed to load Stable Audio model with any method")
            
        except Exception as e:
            logger.error(f"Failed to initialize Stable Audio model: {e}")
            self.model = None
            self.pipeline = None
    
    def forward(
        self, 
        prompt: str,
        duration: str = "30",
        negative_prompt: Optional[str] = None,
        steps: str = "20"
    ) -> str:
        """
        Generate audio from text description.
        
        Args:
            prompt: Text description of audio to generate
            duration: Duration in seconds
            negative_prompt: What to avoid in generation
            steps: Number of inference steps
            
        Returns:
            Path to generated audio file
        """
        try:
            # Parse parameters
            duration_sec = float(duration)
            num_steps = int(steps)
            num_steps = min(max(num_steps, 5), 50)  # Clamp steps to [5, 50]

            # Clamp duration to model limits
            duration_sec = min(max(duration_sec, 1), 47)
            
            # Generate audio
            if self.model is not None and self.model_config is not None:
                audio_path = self._generate_with_tools(
                    prompt, duration_sec, negative_prompt, num_steps
                )
            elif self.pipeline is not None:
                audio_path = self._generate_with_pipeline(
                    prompt, duration_sec, negative_prompt, num_steps
                )
            else:
                return "Stable Audio model is not available. Please check the installation."
            
            logger.info(f"Generated audio for prompt: {prompt[:50]}...")
            return f"Generated audio saved to: {audio_path}"
            
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return f"Error generating audio: {str(e)}"
    
    def _generate_with_tools(
        self, 
        prompt: str, 
        duration: float,
        negative_prompt: Optional[str],
        steps: int
    ) -> str:
        """Generate audio using stable-audio-tools (disabled due to missing dependencies)."""
        return "stable-audio-tools method is not available. Using diffusers pipeline instead."
    
    def _generate_with_pipeline(
        self, 
        prompt: str, 
        duration: float,
        negative_prompt: Optional[str],
        steps: int
    ) -> str:
        """Generate audio using diffusers pipeline."""
        
        # Set up generation parameters
        generator = torch.Generator(self.device).manual_seed(42)
        
        # Generate audio
        audio = self.pipeline(
            prompt,
            negative_prompt=negative_prompt or "Low quality, distorted",
            num_inference_steps=steps,
            audio_end_in_s=duration,
            num_waveforms_per_prompt=1,
            generator=generator,
        ).audios
        
        # Process audio
        output = audio[0].T.float().cpu().numpy()
        
        # Save to file
        output_path = self._save_audio(output, self.pipeline.vae.sampling_rate)
        return output_path
    
    def _save_audio(self, audio_data, sample_rate: int) -> str:
        """Save audio data to file."""
        # Create temporary file
        temp_dir = Path(tempfile.gettempdir()) / "music_agent_audio"
        temp_dir.mkdir(exist_ok=True)
        
        output_path = temp_dir / f"generated_audio_{hash(str(audio_data.tobytes()))}.wav"
        
        # Save audio file
        sf.write(str(output_path), audio_data, sample_rate)
        
        return str(output_path)
    
    def generate_music(
        self, 
        description: str, 
        genre: Optional[str] = None,
        bpm: Optional[str] = None,
        duration: float = 10
    ) -> str:
        """Generate music with specific parameters."""
        
        prompt_parts = [description]
        
        if bpm:
            prompt_parts.append(f"{bpm} BPM")
        
        if genre:
            prompt_parts.append(genre)
        
        prompt = " ".join(prompt_parts)
        return self.forward(prompt, duration=str(duration))
    
    def generate_sound_effect(self, description: str, duration: float = 5) -> str:
        """Generate sound effects."""
        prompt = f"Sound effect: {description}"
        return self.forward(prompt, duration=str(duration))
    
    def generate_ambient(self, description: str, duration: float = 30) -> str:
        """Generate ambient sounds."""
        prompt = f"Ambient sound: {description}"
        return self.forward(prompt, duration=str(duration))
