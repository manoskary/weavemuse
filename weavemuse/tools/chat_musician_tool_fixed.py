"""
ChatMusician Tool - Advanced music understanding and generation using ChatMusician model.
"""

print("🔥 STARTING CHAT_MUSICIAN_TOOL.PY MODULE EXECUTION")

import logging
import tempfile
import os
from typing import Optional, Dict, Any, Union
from pathlib import Path

from smolagents.tools import Tool  # type: ignore

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

from .base_tools import ManagedTransformersTool

logger = logging.getLogger(__name__)

print("📍 About to define ChatMusicianTool class...")


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
        if not CHAT_MUSICIAN_AVAILABLE:
            raise ImportError(f"ChatMusician dependencies not available: {import_error}")
        
        # ChatMusician is ~14B params, estimate VRAM usage
        estimated_vram = 4000.0 if load_in_4bit else 16000.0
        
        super().__init__(
            model_id=model_id,
            device=device,
            estimated_vram_mb=estimated_vram,
            torch_dtype="float16" if device == "cuda" else "float32",
            load_in_4bit=load_in_4bit,
            priority=2,  # High priority tool
            **kwargs
        )
        
        logger.info(f"ChatMusician tool initialized (lazy loading enabled)")
    
    def forward(self, **kwargs) -> str:
        """
        Generate a response using ChatMusician model.
        
        Args:
            query: Musical question, composition request, or music analysis task
            max_tokens: Maximum tokens to generate (default: 512)
            temperature: Generation temperature 0.0-1.0 (default: 0.7)
            
        Returns:
            Generated response
        """
        # Extract parameters from kwargs
        query = kwargs.get('query', '')
        max_tokens = kwargs.get('max_tokens', '512')
        temperature = kwargs.get('temperature', '0.7')
        
        # Simple synchronous implementation for now
        # This will be called by the smolagents framework
        if not CHAT_MUSICIAN_AVAILABLE:
            return "❌ ChatMusician not available - missing dependencies"
        
        try:
            # For now, return a simple response
            # In a full implementation, this would load the model and generate
            return f"ChatMusician response to: {query} (max_tokens: {max_tokens}, temperature: {temperature})"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _load_model(self) -> Dict[str, Any]:
        """
        Load the ChatMusician model and tokenizer.
        
        Returns:
            Dictionary containing model, tokenizer, and pipeline
        """
        if not CHAT_MUSICIAN_AVAILABLE:
            raise ImportError(f"ChatMusician dependencies not available: {import_error}")
            
        logger.info(f"Loading ChatMusician model: {self.model_id}")
        
        # Load tokenizer
        if AutoTokenizer is None:
            raise ImportError("AutoTokenizer not available")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )
        
        # Load model with optimization
        model_kwargs = self._get_model_kwargs()
        
        if AutoModelForCausalLM is None:
            raise ImportError("AutoModelForCausalLM not available")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if self.device != "cuda" or "device_map" not in model_kwargs:
            model = model.to(self.device)  # type: ignore
        
        # Create pipeline
        if pipeline is None:
            raise ImportError("pipeline not available")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=self.device if self.device != "cuda" else None
        )
        
        logger.info(f"ChatMusician model loaded successfully on {self.device}")
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "pipeline": pipe
        }
    
    def _call_model(
        self, 
        model: Dict[str, Any], 
        **kwargs
    ) -> str:
        """
        Call the ChatMusician model.
        
        Args:
            model: Dictionary containing loaded components
            **kwargs: Arguments passed from forward() including query, max_tokens, temperature
            
        Returns:
            Generated response
        """
        try:
            # Extract components
            pipe = model["pipeline"]
            
            # Extract parameters from kwargs
            query = kwargs.get("query", "")
            max_tokens = kwargs.get("max_tokens", "512")
            temperature = kwargs.get("temperature", "0.7")
            
            # Convert string parameters
            max_tokens_int = int(max_tokens) if max_tokens else 512
            temperature_float = float(temperature) if temperature else 0.7
            
            # Prepare prompt for ChatMusician
            if "User:" not in query and "Assistant:" not in query:
                formatted_query = f"User: {query}\nAssistant:"
            else:
                formatted_query = query
            
            # Generate response
            result = pipe(
                formatted_query,
                max_new_tokens=max_tokens_int,
                temperature=temperature_float,
                do_sample=temperature_float > 0,
                pad_token_id=pipe.tokenizer.eos_token_id,
                return_full_text=False,
                clean_up_tokenization_spaces=True
            )
            
            # Extract generated text
            if result and len(result) > 0:
                generated_text = result[0].get("generated_text", "").strip()
                
                # Clean up the response
                if generated_text.startswith("Assistant:"):
                    generated_text = generated_text[10:].strip()
                
                return generated_text
            else:
                return "No response generated."
                
        except Exception as e:
            logger.error(f"Error in ChatMusician generation: {e}")
            return f"Error generating response: {e}"
    
    def _unload_model(self, model: Dict[str, Any]) -> None:
        """
        Unload the ChatMusician model to free VRAM.
        
        Args:
            model: Dictionary containing model components to unload
        """
        try:
            logger.info("Unloading ChatMusician model")
            
            # Clean up model
            if "model" in model:
                model_obj = model["model"]
                del model["model"]
                del model_obj
            
            # Clean up tokenizer
            if "tokenizer" in model:
                del model["tokenizer"]
            
            # Force cleanup
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        except Exception as e:
            logger.warning(f"Error during ChatMusician cleanup: {e}")


print("✅ ChatMusicianTool class defined successfully!")
print(f"✅ Class: {ChatMusicianTool}")
print(f"✅ Base classes: {ChatMusicianTool.__bases__}")

# Export the class for imports
__all__ = ['ChatMusicianTool']
