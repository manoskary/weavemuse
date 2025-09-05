"""
ChatMusician Tool - Advanced music understanding and generation using ChatMusician model.
"""

import logging
import tempfile
import os
from typing import Optional, Dict, Any, Union
from pathlib import Path
from string import Template

from smolagents.tools import Tool
from .base_tools import ManagedTransformersTool

# Global imports that will be loaded lazily
AutoTokenizer = None
AutoModelForCausalLM = None
GenerationConfig = None
torch = None

logger = logging.getLogger(__name__)


class ChatMusicianTool(ManagedTransformersTool):
    """
    Tool for advanced music understanding and generation using ChatMusician.
    
    This tool can:
    - Analyze musical compositions and styles
    - Generate symbolic music notation
    - Answer complex music theory questions
    - Provide musical analysis and insights
    - Create music in ABC notation format
    
    Now with lazy loading and VRAM management!
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
    
    def __init__(
        self, 
        device: str = "auto", 
        model_id: str = "m-a-p/ChatMusician",
        load_in_4bit: bool = True,  # Enable 4-bit quantization by default
        **kwargs
    ):
        """
        Initialize ChatMusician tool with lazy loading.
        
        Args:
            device: Device to run on ("auto", "cuda", "cpu")
            model_id: HuggingFace model ID
            load_in_4bit: Use 4-bit quantization to save VRAM
            **kwargs: Additional arguments
        """
        
        # ChatMusician is a large model, estimate VRAM usage
        # With 4-bit quantization, should be around 3-4GB
        estimated_vram = 4000.0 if load_in_4bit else 8000.0
        
        super().__init__(
            model_id=model_id,
            device=device,
            estimated_vram_mb=estimated_vram,
            torch_dtype="float16" if device == "cuda" else "float32",
            priority=2,  # High priority for chat-based music generation
            **kwargs
        )
        
        self.load_in_4bit = load_in_4bit
        self.prompt_template = Template("Human: ${inst} </s> Assistant: ")
        
        logger.info(f"ChatMusician tool initialized (lazy loading enabled)")
    
    def _load_model(self) -> Dict[str, Any]:
        """
        Load the ChatMusician model.
        
        Returns:
            Dictionary containing model, tokenizer and other components
        """
        global AutoTokenizer, AutoModelForCausalLM, GenerationConfig, torch
        
        try:
            # Lazy import of required modules
            import torch as torch_mod
            from transformers import AutoTokenizer as AT, AutoModelForCausalLM as AM, GenerationConfig as GC
            
            torch = torch_mod
            AutoTokenizer = AT
            AutoModelForCausalLM = AM
            GenerationConfig = GC
            
        except ImportError as e:
            logger.error(f"Failed to import ChatMusician dependencies: {e}")
            raise ImportError(f"ChatMusician dependencies not available: {e}")
        
        logger.info(f"Loading ChatMusician model: {self.model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )
        
        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "resume_download": True
        }
        
        if self.device == "cuda":
            model_kwargs["device_map"] = "cuda"
        
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
                logger.info("Using 4-bit quantization")
            except ImportError:
                logger.warning("BitsAndBytesConfig not available, loading without quantization")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        ).eval()
        
        # Set up generation config
        generation_config = GenerationConfig(
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.1,
            min_new_tokens=10,
            max_new_tokens=1536
        )
        
        model_dict = {
            "model": model,
            "tokenizer": tokenizer,
            "generation_config": generation_config,
            "torch": torch
        }
        
        logger.info(f"ChatMusician model loaded successfully on {self.device}")
        
        return model_dict
    
    def forward(
        self,
        query: str,
        max_tokens: str = "512",
        temperature: str = "0.7"
    ) -> str:
        """
        Generate a response using ChatMusician model.
        
        Args:
            query: Musical question, composition request, or music analysis task
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Generation temperature 0.0-1.0 (default: 0.7)
            
        Returns:
            Generated response
        """
        # Convert explicit parameters to kwargs and call parent's forward method
        kwargs = {
            "query": query,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        # Call the parent's forward method which handles the VRAM management
        return super().forward(**kwargs)
    
    def _call_model(
        self,
        model: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Generate response using ChatMusician model.
        
        Args:
            model: Dictionary containing loaded components
            **kwargs: Arguments passed from forward() including query, max_tokens, temperature
            
        Returns:
            Generated response as string
        """
        try:
            # Extract components
            chat_model = model["model"]
            tokenizer = model["tokenizer"]
            generation_config = model["generation_config"]
            torch = model["torch"]
            
            # Extract parameters from kwargs
            query = kwargs.get("query", "")
            max_tokens = int(kwargs.get("max_tokens", "512"))
            temperature = float(kwargs.get("temperature", "0.7"))
            
            if not query:
                return "Error: No query provided"
            
            # Prepare prompt using the template
            prompt = self.prompt_template.safe_substitute({"inst": query})
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            
            # Move inputs to device if using CUDA
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Update generation config with provided parameters
            gen_config = generation_config
            gen_config.max_new_tokens = min(max_tokens, 1536)  # Cap at model's max
            gen_config.temperature = temperature
            
            # Generate response
            with torch.no_grad():
                response = chat_model.generate(
                    input_ids=inputs["input_ids"].to(chat_model.device),
                    attention_mask=inputs['attention_mask'].to(chat_model.device),
                    eos_token_id=tokenizer.eos_token_id,
                    generation_config=gen_config,
                )
            
            # Decode response (skip the input tokens)
            response = tokenizer.decode(
                response[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in ChatMusician generation: {e}")
            # Try fallback response
            try:
                return self._fallback_response(kwargs.get("query", ""), str(e))
            except Exception as fallback_error:
                logger.error(f"Fallback response also failed: {fallback_error}")
                return f"Error generating music response: {str(e)}"
    
    def _fallback_response(self, query: str, error: str) -> str:
        """Generate a fallback response when model fails."""
        return f"""🎵 ChatMusician Tool
        
📝 Query: {query}
⚠️ Model Response: Model temporarily unavailable

Note: ChatMusician model dependencies not fully available. 
For full music generation capabilities, please ensure all required dependencies are installed.

Error: {error}"""
    
    # Convenience methods for specific music tasks
    async def compose_music(self, style: str, instruments: str = "piano") -> str:
        """Compose music in a specific style."""
        query = f"Compose a musical piece in {style} style for {instruments}. Please provide the composition in ABC notation format."
        return self.forward(query=query, max_tokens="1024", temperature="0.8")
    
    async def analyze_chord_progression(self, chords: str) -> str:
        """Analyze a chord progression."""
        query = f"Analyze this chord progression and explain its harmonic structure: {chords}"
        return self.forward(query=query, max_tokens="512", temperature="0.3")
    
    async def generate_from_chords(self, chords: str) -> str:
        """Generate a musical piece from chord progression."""
        query = f"Develop a musical piece using the given chord progression: {chords}. Please provide in ABC notation format."
        return self.forward(query=query, max_tokens="1024", temperature="0.7")
    
    async def music_theory_question(self, question: str) -> str:
        """Answer music theory questions."""
        query = f"Music theory question: {question}"
        return self.forward(query=query, max_tokens="512", temperature="0.3")
    
    async def arrange_melody(self, melody: str, style: str = "classical") -> str:
        """Arrange a melody in a specific style."""
        query = f"Arrange this melody in {style} style: {melody}. Provide the arrangement in ABC notation."
        return self.forward(query=query, max_tokens="1024", temperature="0.7")


# Export the class for imports
__all__ = ['ChatMusicianTool']
