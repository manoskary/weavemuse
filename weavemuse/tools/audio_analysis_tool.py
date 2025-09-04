"""
Audio Analysis Tool - Powered by Qwen2-Audio-7B for comprehensive audio understanding.
"""

import logging
import tempfile
import os
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen

from smolagents.tools import Tool
from .base_tools import ManagedTransformersTool

# Global imports that will be loaded lazily
AutoProcessor = None
Qwen2AudioForConditionalGeneration = None
librosa = None
np = None
torch = None

logger = logging.getLogger(__name__)


class AudioAnalysisTool(ManagedTransformersTool):
    """
    Tool for comprehensive audio analysis and understanding using Qwen2-Audio-7B.
    
    This tool can:
    - Generate natural language descriptions of audio content
    - Analyze musical characteristics and features
    - Describe audio events and sounds
    - Provide detailed audio captions and analysis
    
    Now with lazy loading and VRAM management!
    """
    # Class attributes required by smolagents
    name = "audio_analysis"
    description = (
        "Analyzes audio files using Qwen2-Audio-7B to generate "
        "natural language descriptions and detailed analysis of audio content, "
        "musical features, and sound characteristics."
    )
    inputs = {
        "audio_file": {
            "type": "string", 
            "description": "Path to the audio file to analyze"
        },
        "analysis_type": {
            "type": "string", 
            "description": "Type of analysis: 'caption', 'musical', 'detailed', 'tempo', 'harmony'",
            "required": False,
            "nullable": True
        },
        "custom_prompt": {
            "type": "string",
            "description": "Custom prompt for specific analysis (optional)",
            "required": False,
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(
        self, 
        device: str = "auto", 
        model_id: str = "Qwen/Qwen2-Audio-7B", 
        **kwargs
    ):
        """
        Initialize AudioAnalysis tool with lazy loading.
        
        Args:
            device: Device to run on ("auto", "cuda", "cpu")
            model_id: Qwen2-Audio model ID
            **kwargs: Additional arguments
        """
        
        # Qwen2-Audio is a larger model, estimate VRAM usage
        estimated_vram = 8000.0  # ~8GB for 7B model
        
        super().__init__(
            model_id=model_id,
            device=device,
            estimated_vram_mb=estimated_vram,
            torch_dtype="float16" if device == "cuda" else "float32",
            priority=3,  # Medium priority for audio analysis
            **kwargs
        )
        
        logger.info(f"AudioAnalysis tool initialized (lazy loading enabled)")
    
    def _load_model(self) -> Dict[str, Any]:
        """
        Load the Qwen2-Audio model.
        
        Returns:
            Dictionary containing model, processor and other components
        """
        
        try:
            import torch
            import librosa
            import numpy as np
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
            QWEN_AVAILABLE = True
            print("✅ Qwen2-Audio dependencies loaded successfully!")
        except ImportError as e:
            QWEN_AVAILABLE = False
            import_error = str(e)
            print(f"❌ Qwen2-Audio dependencies failed: {e}")            
            raise ImportError(f"Qwen2-Audio dependencies not available: {import_error}")            
            
        logger.info(f"Loading Qwen2-Audio model: {self.model_id}")
        
        # Load processor first
        processor = AutoProcessor.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )
        
        # Load model with appropriate settings
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_id, 
            trust_remote_code=True,                
            device_map=self.device if self.device != "auto" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        if self.device == "cuda":
            model = model.to("cuda")
        
        model_dict = {
            "model": model,
            "processor": processor,
            "librosa": librosa,
            "np": np,
            "torch": torch
        }
        
        logger.info(f"Qwen2-Audio model loaded successfully on {self.device}")
        
        return model_dict
    
    def forward(
        self,
        audio_file: str,
        analysis_type: str = "caption",
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Analyze audio file using Qwen2-Audio model.
        
        Args:
            audio_file: Path to the audio file to analyze
            analysis_type: Type of analysis ('caption', 'musical', 'detailed', 'tempo', 'harmony')
            custom_prompt: Custom prompt for specific analysis (optional)
            
        Returns:
            Analysis result as string
        """
        # Convert explicit parameters to kwargs and call parent's forward method
        kwargs = {
            "audio_file": audio_file,
            "analysis_type": analysis_type,
            "custom_prompt": custom_prompt
        }
        
        # Call the parent's forward method which handles the VRAM management
        return super().forward(**kwargs)
    
    def _load_audio_with_librosa(self, audio_source: str, processor, librosa):
        """Load audio from file path or URL."""
        try:
            if audio_source.startswith(('http://', 'https://')):
                # Load from URL
                audio_data = urlopen(audio_source).read()
                audio, sr = librosa.load(
                    BytesIO(audio_data), 
                    sr=processor.feature_extractor.sampling_rate
                )
            else:
                # Load from file
                audio, sr = librosa.load(
                    audio_source, 
                    sr=processor.feature_extractor.sampling_rate
                )
            
            return audio
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return None
    
    def _generate_response_with_model(self, audio_data, prompt: str, model, processor, torch) -> str:
        """Generate response using Qwen2-Audio model."""
        try:
            # Prepare inputs
            inputs = processor(text=prompt, audios=audio_data, return_tensors="pt")
            
            # Move inputs to device if using CUDA
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_length=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Decode response
            generated_ids = generated_ids[:, inputs['input_ids'].size(1):]
            response = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def _call_model(
        self,
        model: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Analyze audio using Qwen2-Audio model.
        
        Args:
            model: Dictionary containing loaded components
            **kwargs: Arguments passed from forward() including audio_file, analysis_type, custom_prompt
            
        Returns:
            Analysis result as string
        """
        try:
            # Extract components
            model_components = model["model"]
            processor = model["processor"]
            librosa = model["librosa"]
            np = model["np"]
            torch = model["torch"]
            
            # Extract parameters from kwargs
            audio_file = kwargs.get("audio_file")
            analysis_type = kwargs.get("analysis_type", "caption")
            custom_prompt = kwargs.get("custom_prompt")
            
            if not audio_file:
                return "Error: No audio file provided"
            
            if not os.path.exists(audio_file) and not audio_file.startswith(('http://', 'https://')):
                return f"Audio file not found: {audio_file}"
            
            # Load audio
            audio_data = self._load_audio_with_librosa(audio_file, processor, librosa)
            if audio_data is None:
                # Fallback to basic analysis if audio loading fails
                return self._fallback_analysis(audio_file, analysis_type, librosa, np)
            
            # Generate analysis based on type
            if custom_prompt:
                prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>{custom_prompt}"
            else:
                prompt = self._get_analysis_prompt(analysis_type)
            
            # Process audio and generate response
            response = self._generate_response_with_model(audio_data, prompt, model_components, processor, torch)
            
            # Add file information
            duration = len(audio_data) / processor.feature_extractor.sampling_rate
            file_info = f"\n\n📁 File: {os.path.basename(audio_file)} ({duration:.2f}s)"
            
            return f"{response}{file_info}"
            
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            # Try fallback analysis on error
            try:
                return self._fallback_analysis(
                    kwargs.get("audio_file", "unknown"),
                    kwargs.get("analysis_type", "caption"),
                    model.get("librosa"),
                    model.get("np")
                )
            except Exception as fallback_error:
                logger.error(f"Fallback analysis also failed: {fallback_error}")
                return f"Error analyzing audio: {str(e)}"
    
    def _get_analysis_prompt(self, analysis_type: str) -> str:
        """Get prompt based on analysis type."""
        prompts = {
            "caption": "<|audio_bos|><|AUDIO|><|audio_eos|>Generate a detailed caption describing this audio in English:",
            "musical": "<|audio_bos|><|AUDIO|><|audio_eos|>Analyze the musical characteristics of this audio including genre, tempo, instruments, and mood:",
            "detailed": "<|audio_bos|><|AUDIO|><|audio_eos|>Provide a comprehensive analysis of this audio including all musical and acoustic characteristics:",
            "tempo": "<|audio_bos|><|AUDIO|><|audio_eos|>Analyze the tempo and rhythmic characteristics of this music:",
            "harmony": "<|audio_bos|><|AUDIO|><|audio_eos|>Describe the harmonic content, key, and chord progressions in this music:",
            "instruments": "<|audio_bos|><|AUDIO|><|audio_eos|>Identify and describe the instruments and sounds present in this audio:",
            "mood": "<|audio_bos|><|AUDIO|><|audio_eos|>Describe the mood, emotion, and atmosphere of this audio:",
            "structure": "<|audio_bos|><|AUDIO|><|audio_eos|>Analyze the structural elements and form of this musical piece:"
        }
        
        return prompts.get(analysis_type, prompts["caption"])
    
    def _fallback_analysis(self, audio_file: str, analysis_type: str, librosa_mod, np_mod) -> str:
        """Fallback analysis when Qwen2-Audio is not available."""
        try:
            if not audio_file.startswith(('http://', 'https://')):
                if not os.path.exists(audio_file):
                    return f"Audio file not found: {audio_file}"
            
            # Try to use librosa for basic analysis
            if librosa_mod is not None:
                try:
                    y, sr = librosa_mod.load(audio_file, sr=None)
                    duration = len(y) / sr
                    
                    # Basic tempo detection
                    tempo, _ = librosa_mod.beat.beat_track(y=y, sr=sr)
                    
                    # Basic spectral analysis
                    spectral_centroid = np_mod.mean(librosa_mod.feature.spectral_centroid(y=y, sr=sr))
                    
                    return f"""🎵 Basic Audio Analysis (Fallback Mode)
                    
📁 File: {os.path.basename(audio_file)}
⏱️ Duration: {duration:.2f} seconds
🎼 Estimated Tempo: {tempo:.1f} BPM
🔊 Spectral Centroid: {spectral_centroid:.1f} Hz

Note: Qwen2-Audio model not available. Using basic librosa analysis.
For detailed AI-powered analysis, please ensure Qwen2-Audio dependencies are installed."""
                    
                except Exception as e:
                    logger.error(f"Fallback analysis error: {e}")
            
            return f"""Audio Analysis Tool
            
📁 File: {os.path.basename(audio_file) if not audio_file.startswith(('http://', 'https://')) else 'URL Audio'}
⚠️ Analysis Type: {analysis_type}

Note: Qwen2-Audio model and librosa dependencies not available. 
Please install required dependencies for full audio analysis capabilities."""
            
        except Exception as e:
            return f"Error in fallback analysis: {str(e)}"
    
    # Convenience methods for specific analysis types
    async def analyze_tempo(self, audio_file: str) -> str:
        """Analyze tempo and rhythm specifically."""
        return self.forward(audio_file=audio_file, analysis_type="tempo")
    
    async def analyze_harmony(self, audio_file: str) -> str:
        """Analyze harmony and musical key specifically."""
        return self.forward(audio_file=audio_file, analysis_type="harmony")
    
    async def analyze_instruments(self, audio_file: str) -> str:
        """Identify instruments in the audio."""
        return self.forward(audio_file=audio_file, analysis_type="instruments")
    
    async def analyze_mood(self, audio_file: str) -> str:
        """Analyze mood and emotion in the audio."""
        return self.forward(audio_file=audio_file, analysis_type="mood")
    
    async def analyze_structure(self, audio_file: str) -> str:
        """Analyze musical structure and form."""
        return self.forward(audio_file=audio_file, analysis_type="structure")
    
    async def generate_caption(self, audio_file: str) -> str:
        """Generate a natural language caption for the audio."""
        return self.forward(audio_file=audio_file, analysis_type="caption")
    
    async def custom_analysis(self, audio_file: str, prompt: str) -> str:
        """Perform custom analysis with a specific prompt."""
        return self.forward(audio_file=audio_file, custom_prompt=prompt)
