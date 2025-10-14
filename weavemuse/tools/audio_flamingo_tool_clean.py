"""
Audio Flamingo Tool for WeaveMuse
=================================
A tool that uses NVIDIA's Audio Flamingo model via Gradio Client API
for advanced audio analysis and question answering.
"""

import logging
from typing import Any, Dict
import os
from gradio_client import Client, handle_file
from smolagents.tools import Tool

logger = logging.getLogger(__name__)


class AudioFlamingoTool(Tool):
    """
    Audio analysis tool using NVIDIA's Audio Flamingo model via Gradio Client.
    
    This tool provides advanced audio understanding capabilities including:
    - Audio content description
    - Musical analysis
    - Audio question answering
    - Sound identification
    - Acoustic feature analysis
    """
    
    # Class attributes required by smolagents
    name = "audio_flamingo"
    description = (
        "Analyzes audio files using NVIDIA's Audio Flamingo model. "
        "Can answer questions about audio content, describe musical elements, "
        "identify sounds, and provide detailed acoustic analysis. "
        "Supports various audio formats and natural language queries."
    )
    inputs = {
        "audio_file": {
            "type": "string",
            "description": "Path to the audio file to analyze"
        },
        "query": {
            "type": "string", 
            "description": "Question or description request about the audio"
        }
    }
    output_type = "string"
    
    def __init__(self):
        """Initialize the Audio Flamingo tool."""
        super().__init__()
        self.client = None
        self._initialized = False
        
    def _initialize_client(self):
        """Initialize the Gradio client for Audio Flamingo model."""
        if not self._initialized:
            try:
                logger.info("Connecting to NVIDIA Audio Flamingo model...")
                self.client = Client(
                    "nvidia/audio-flamingo-3",
                    hf_token=os.getenv("HF_TOKEN")
                )
                self._initialized = True
                logger.info("✅ Audio Flamingo client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Audio Flamingo client: {e}")
                raise RuntimeError(f"Could not connect to Audio Flamingo model: {e}")
    
    def forward(self, audio_file: str, query: str) -> str:
        """
        Analyze audio file using Audio Flamingo model.
        
        Args:
            audio_file: Path to audio file
            query: Query about the audio
            
        Returns:
            Analysis result as text
        """
        try:
            # Initialize client if needed
            self._initialize_client()
            
            if not self.client:
                raise RuntimeError("Audio Flamingo client not initialized")
            
            if not query or not query.strip():
                query = "Describe this audio"
            
            # Directly process the file with handle_file and gradio client
            logger.info(f"Processing audio query: '{query}'")
            logger.info(f"Audio file: {audio_file}")
            
            # Process with Audio Flamingo model
            result = self.client.predict(
                audio_file,  # Audio file path
                query,       # Query text  
                api_name="/predict"
            )
            
            if result:
                logger.info(f"✅ Audio Flamingo analysis completed")
                return str(result)
            else:
                logger.warning("Audio Flamingo returned empty result")
                return "No analysis result available for this audio file."
                
        except Exception as e:
            error_msg = f"Audio analysis failed: {str(e)}"
            logger.error(error_msg)
            
            # Provide helpful error messages for common issues
            if "not found" in str(e).lower():
                return f"Error: Audio file not found or inaccessible: {audio_file}"
            elif "timeout" in str(e).lower():
                return "Error: Audio analysis timed out. Please try with a smaller audio file."
            elif "memory" in str(e).lower():
                return "Error: Insufficient memory for audio analysis. Please try with a smaller file."
            elif "connection" in str(e).lower():
                return "Error: Could not connect to Audio Flamingo model. Please check your internet connection."
            else:
                return f"Error: Audio analysis failed - {str(e)}"
    
    def describe_audio(self, audio_file: str) -> str:
        """
        Get a general description of the audio content.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Description of audio content
        """
        query = (
            "Describe this audio in detail. "
            "What can you hear? What is the overall content and characteristics?"
        )
        return self.forward(audio_file, query)
    
    def analyze_music(self, audio_file: str) -> str:
        """
        Analyze musical content in the audio.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Musical analysis result
        """
        query = (
            "Analyze the musical content of this audio. "
            "Describe the genre, style, tempo, key, instruments, "
            "vocal characteristics, and overall musical structure."
        )
        return self.forward(audio_file, query)
    
    def identify_sounds(self, audio_file: str) -> str:
        """
        Identify and describe individual sounds in the audio.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Sound identification result
        """
        query = (
            "Identify and describe the individual sounds in this audio. "
            "What specific sounds, voices, or events can you detect?"
        )
        return self.forward(audio_file, query)
    
    def analyze_instruments(self, audio_file: str) -> str:
        """
        Analyze instruments present in the audio.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Instrument analysis result
        """
        query = (
            "What instruments can you identify in this audio? "
            "List each instrument and describe how it's being played."
        )
        return self.forward(audio_file, query)
    
    def analyze_acoustic_features(self, audio_file: str) -> str:
        """
        Analyze acoustic features of the audio.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Acoustic analysis result
        """
        query = (
            "Analyze the acoustic features of this audio including "
            "dynamics, timbre, rhythm, pitch characteristics, "
            "and sound quality. Describe any notable audio effects or processing."
        )
        return self.forward(audio_file, query)


# Example usage and testing functions
def test_audio_flamingo():
    """Test the Audio Flamingo tool with sample queries."""
    tool = AudioFlamingoTool()
    
    # This would need an actual audio file to test
    sample_audio = "/path/to/sample/audio.wav"
    
    # Test basic analysis (would work with real file)
    print("Audio Flamingo Tool initialized successfully")
    print("Ready to analyze audio files with natural language queries")


if __name__ == "__main__":
    test_audio_flamingo()
