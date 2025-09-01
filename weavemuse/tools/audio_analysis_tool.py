"""
Audio Analysis Tool - Powered by Qwen2-Audio-7B-GGUF for comprehensive audio understanding.
"""

import logging
import tempfile
import os
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen

try:
    from smolagents.tools import Tool
except ImportError:
    # Fallback for development
    class Tool:
        def __init__(self, name: str, description: str, inputs: dict, output_type: str):
            self.name = name
            self.description = description
            self.inputs = inputs
            self.output_type = output_type

try:
    import librosa
    import numpy as np
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    QWEN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Qwen2-Audio dependencies not available: {e}")
    QWEN_AVAILABLE = False

try:
    # Fallback dependencies for basic analysis
    import librosa
    import numpy as np
except ImportError as e:
    logging.warning(f"Basic audio dependencies not available: {e}")


logger = logging.getLogger(__name__)


class AudioAnalysisTool(Tool):
    """
    Tool for comprehensive audio analysis and understanding using Qwen2-Audio-7B-GGUF.
    
    This tool can:
    - Generate natural language descriptions of audio content
    - Analyze musical characteristics and features
    - Describe audio events and sounds
    - Provide detailed audio captions and analysis
    """
    # Class attributes required by smolagents
    name="audio_analysis"
    description=(
        "Analyzes audio files using Qwen2-Audio-7B-GGUF to generate "
        "natural language descriptions and detailed analysis of audio content, "
        "musical features, and sound characteristics."
    )
    inputs={
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
    output_type="string"

    def __init__(self, device: str = "auto", model: str = "Qwen/Qwen2-Audio-7B", **kwargs):        

        # Set the description attribute directly for smolagents compatibility
        self.description = (
            "Analyzes audio files using Qwen2-Audio-7B to generate "
            "natural language descriptions and detailed analysis of audio content, "
            "musical features, and sound characteristics."
        )
        
        self.device = device
        self.model_id = model
        self.processor = None
        self.model_loaded = False 
        self.quantization_config = kwargs.get("quantization_config", None) 

        super().__init__(
            name=self.name,
            description=self.description,
            inputs=self.inputs,
            output_type=self.output_type
        )


        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Qwen2-Audio model."""
        if not QWEN_AVAILABLE:
            logger.warning("Qwen2-Audio dependencies not available. Tool will use fallback analysis.")
            return
            
        try:                            
            
            logger.info("Initializing Qwen2-Audio-7B model...")
            
            # Load model and processor
            if self.model_id.startswith("Qwen/"):
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    self.model_id, 
                    trust_remote_code=True,                
                    device_map=self.device,
                    quantization_config=self.quantization_config
                )

            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            
            if self.device.startswith("cuda"):
                self.model = self.model.to(self.device)

            self.model_loaded = True
            logger.info("Qwen2-Audio model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qwen2-Audio model: {e}")
            self.model_loaded = False
    
    def forward(
        self, 
        audio_file: str,
        analysis_type: str = "caption",
        custom_prompt: Optional[str] = None
    ) -> str:
        """
        Analyze an audio file using Qwen2-Audio and return comprehensive analysis.
        
        Args:
            audio_file: Path to audio file or URL
            analysis_type: Type of analysis to perform
            custom_prompt: Custom prompt for specific analysis
            
        Returns:
            Analysis results as natural language description
        """
        try:
            if not os.path.exists(audio_file) and not audio_file.startswith(('http://', 'https://')):
                return f"Audio file not found: {audio_file}"
            
            # If model is not loaded, use fallback analysis
            if not self.model_loaded:
                return self._fallback_analysis(audio_file, analysis_type)
            
            # Load audio
            audio_data = self._load_audio(audio_file)
            if audio_data is None:
                return "Error: Could not load audio file"
            
            # Generate analysis based on type
            if custom_prompt:
                prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>{custom_prompt}"
            else:
                prompt = self._get_analysis_prompt(analysis_type)
            
            # Process audio and generate response
            response = self._generate_response(audio_data, prompt)
            
            # Add file information
            duration = len(audio_data) / self.processor.feature_extractor.sampling_rate
            file_info = f"\n\n📁 File: {os.path.basename(audio_file)} ({duration:.2f}s)"
            
            return f"{response}{file_info}"
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return f"Error analyzing audio: {str(e)}"
    
    def _load_audio(self, audio_source: str):
        """Load audio from file path or URL."""
        try:
            if audio_source.startswith(('http://', 'https://')):
                # Load from URL
                audio_data = urlopen(audio_source).read()
                audio, sr = librosa.load(
                    BytesIO(audio_data), 
                    sr=self.processor.feature_extractor.sampling_rate
                )
            else:
                # Load from file
                audio, sr = librosa.load(
                    audio_source, 
                    sr=self.processor.feature_extractor.sampling_rate
                )
            
            return audio
            
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return None
    
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
    
    def _generate_response(self, audio_data, prompt: str) -> str:
        """Generate response using Qwen2-Audio model."""
        try:
            # Prepare inputs
            inputs = self.processor(text=prompt, audios=audio_data, return_tensors="pt")
            
            # Move inputs to device if using CUDA
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_length=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
            # Decode response
            generated_ids = generated_ids[:, inputs['input_ids'].size(1):]
            response = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def _fallback_analysis(self, audio_file: str, analysis_type: str) -> str:
        """Fallback analysis when Qwen2-Audio is not available."""
        try:
            if not audio_file.startswith(('http://', 'https://')):
                if not os.path.exists(audio_file):
                    return f"Audio file not found: {audio_file}"
            
            # Try to use librosa for basic analysis
            if 'librosa' in globals():
                try:
                    y, sr = librosa.load(audio_file, sr=None)
                    duration = len(y) / sr
                    
                    # Basic tempo detection
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    
                    # Basic spectral analysis
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                    
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
    def analyze_tempo(self, audio_file: str) -> str:
        """Analyze tempo and rhythm specifically."""
        return self.forward(audio_file, analysis_type="tempo")
    
    def analyze_harmony(self, audio_file: str) -> str:
        """Analyze harmony and musical key specifically."""
        return self.forward(audio_file, analysis_type="harmony")
    
    def analyze_instruments(self, audio_file: str) -> str:
        """Identify instruments in the audio."""
        return self.forward(audio_file, analysis_type="instruments")
    
    def analyze_mood(self, audio_file: str) -> str:
        """Analyze mood and emotion in the audio."""
        return self.forward(audio_file, analysis_type="mood")
    
    def analyze_structure(self, audio_file: str) -> str:
        """Analyze musical structure and form."""
        return self.forward(audio_file, analysis_type="structure")
    
    def generate_caption(self, audio_file: str) -> str:
        """Generate a natural language caption for the audio."""
        return self.forward(audio_file, analysis_type="caption")
    
    def custom_analysis(self, audio_file: str, prompt: str) -> str:
        """Perform custom analysis with a specific prompt."""
        return self.forward(audio_file, custom_prompt=prompt)
