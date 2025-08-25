"""
ChatMusician Tool - Advanced music understanding and generation using ChatMusician model.
"""

import logging
import tempfile
import os
from typing import Optional, Dict, Any, Union
from pathlib import Path

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
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import accelerate
    CHAT_MUSICIAN_AVAILABLE = True
    print("✅ ChatMusician dependencies loaded successfully!")
except ImportError as e:
    CHAT_MUSICIAN_AVAILABLE = False
    import_error = str(e)
    print(f"❌ ChatMusician dependencies failed: {e}")
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None

logger = logging.getLogger(__name__)


class ChatMusicianTool(Tool):
    """
    Tool for advanced music understanding and generation using ChatMusician.
    
    This tool can:
    - Analyze musical compositions and styles
    - Generate symbolic music notation
    - Answer complex music theory questions
    - Provide musical analysis and insights
    - Create music in ABC notation format
    """
    
    # Class attributes required by smolagents
    name = "chat_musician"
    description = (
        "Advanced music understanding and symbolic music generation tool. "
        "Analyzes musical structures, provides composition advice, generates ABC notation, "
        "and helps with music theory questions using the ChatMusician model."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Musical question, composition request, or music analysis task"
        },
        "max_tokens": {
            "type": "string", 
            "description": "Maximum tokens to generate (default: 512)",
            "nullable": True
        },
        "temperature": {
            "type": "string",
            "description": "Generation temperature 0.0-1.0 (default: 0.7)",
            "nullable": True
        }
    }
    output_type = "string"
    
    def __init__(self, device: str = "auto", model_id: str = "m-a-p/ChatMusician"):
        self.device = self._get_device(device)
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)  # Add logger before super().__init__
        
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
        """Initialize the ChatMusician model."""
        try:
            self.logger.info(f"Loading ChatMusician model: {self.model_id}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            if self.device != "cuda":
                self.model = self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device != "cuda" else None
            )
            
            self.logger.info(f"ChatMusician model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChatMusician model: {e}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def forward(
        self, 
        query: str,
        max_tokens: str = "512",
        temperature: str = "0.7"
    ) -> str:
        """
        Process a music-related query with ChatMusician.
        
        Args:
            query: Music-related question or request
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            
        Returns:
            Response from ChatMusician model
        """
        if not CHAT_MUSICIAN_AVAILABLE or self.pipeline is None:
            return "ChatMusician model is not available. Please check the installation."
        
        try:
            # Convert string parameters to appropriate types
            max_tokens_int = int(max_tokens) if max_tokens else 512
            temperature_float = float(temperature) if temperature else 0.7
            
            # Format the prompt for music understanding
            prompt = f"Music question: {query}"
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=max_tokens_int,
                temperature=temperature_float,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = response[0]["generated_text"]
            
            # Remove the original prompt from response
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            self.logger.info(f"Generated response for query: {query[:50]}...")
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Error processing ChatMusician query: {e}")
            return f"Error processing query: {str(e)}"
    
    def analyze_music(self, music_content: str) -> str:
        """Analyze musical content."""
        return self.forward(
            f"Please analyze this musical content: {music_content}",
            mode="analysis"
        )
    
    def generate_music(self, description: str, style: Optional[str] = None) -> str:
        """Generate music based on description."""
        prompt = description
        if style:
            prompt = f"{description} in {style} style"
        
        return self.forward(prompt, mode="generation", format="abc")
    
    def answer_theory_question(self, question: str) -> str:
        """Answer music theory questions."""
        return self.forward(question, mode="question")
    
    def compose_melody(self, key: str, tempo: str, style: str) -> str:
        """Compose a melody with specific parameters."""
        prompt = f"Compose a melody in {key} key, {tempo} tempo, {style} style"
        return self.forward(prompt, mode="generation", format="abc")
    
    def harmonize_melody(self, melody: str) -> str:
        """Provide harmonization for a given melody."""
        prompt = f"Please harmonize this melody: {melody}"
        return self.forward(prompt, mode="generation", format="abc")
    
    def explain_composition(self, composition: str) -> str:
        """Explain a musical composition."""
        prompt = f"Please explain this musical composition: {composition}"
        return self.forward(prompt, mode="analysis")
