"""
Memory-efficient tool utilities for the Music Agent Framework.
"""

import gc
import torch
from smolagents import Tool, ChatMessage, MessageRole
from .stable_audio_tool import StableAudioTool
from .audio_analysis_tool import AudioAnalysisTool
from .notagen_tool import NotaGenTool
from ..agents.models import TransformersModel


class GPUOnDemandTool(Tool):
    """Base class for tools that allocate GPU memory only when needed."""
    
    def __init__(self, tool_class, *args, **kwargs):
        super().__init__()
        self.tool_class = tool_class
        self.tool_args = args
        self.tool_kwargs = kwargs
        self.tool_instance = None
        self._last_used = None
        
    def _ensure_tool_loaded(self):
        """Load the tool to GPU if not already loaded."""
        if self.tool_instance is None:
            print(f"Loading {self.tool_class.__name__} to GPU...")
            # Force GPU allocation for the tool
            if 'device' in self.tool_kwargs:
                self.tool_kwargs['device'] = "cuda:0"
            self.tool_instance = self.tool_class(*self.tool_args, **self.tool_kwargs)
            
    def _cleanup_tool(self):
        """Clean up GPU memory after tool usage."""
        if self.tool_instance is not None:
            del self.tool_instance
            self.tool_instance = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            print(f"Cleaned up {self.tool_class.__name__} from GPU")
            
    def forward(self, *args, **kwargs):
        try:
            self._ensure_tool_loaded()
            result = self.tool_instance.forward(*args, **kwargs)
            return result
        finally:
            # Optional: cleanup immediately after use to free VRAM
            # Comment out the next line if you want to keep tools loaded for multiple uses
            self._cleanup_tool()
            
    def __getattr__(self, name):
        """Delegate attribute access to the underlying tool."""
        if self.tool_instance is None:
            # Return tool class attributes for inspection
            return getattr(self.tool_class, name, None)
        return getattr(self.tool_instance, name)


class MemoryEfficientStableAudioTool(GPUOnDemandTool):
    """Memory-efficient wrapper for StableAudioTool."""
    name = "stable_audio"
    description = "Generates high-quality audio and music from text descriptions using GPU on-demand"
    inputs = {
        "prompt": {"type": "string", "description": "Text description of the audio to generate"},
        "duration": {"type": "string", "description": "Duration in seconds (default: 10, max: 47)", "nullable": True},
        "negative_prompt": {"type": "string", "description": "What to avoid in generation (optional)", "nullable": True},
        "steps": {"type": "string", "description": "Number of inference steps (default: 20, max: 50)", "nullable": True}
    }
    output_type = "string"
    
    def __init__(self, device="cuda:0"):
        super().__init__(StableAudioTool, device=device)


class MemoryEfficientAudioAnalysisTool(GPUOnDemandTool):
    """Memory-efficient wrapper for AudioAnalysisTool."""
    name = "audio_analysis"
    description = "Analyzes audio files using AI models loaded on-demand to GPU"
    inputs = {
        "audio_file": {"type": "string", "description": "Path to the audio file to analyze"},
        "analysis_type": {"type": "string", "description": "Type of analysis to perform", "nullable": True}
    }
    output_type = "string"
    
    def __init__(self, device="cuda:0", model="mradermacher/Qwen2-Audio-7B-i1-GGUF"):
        super().__init__(AudioAnalysisTool, device=device, model=model)


class MemoryEfficientNotaGenTool(GPUOnDemandTool):
    """Memory-efficient wrapper for NotaGenTool."""
    name = "notagen"
    description = "Generates symbolic music in ABC notation with full conversion capabilities using GPU on-demand"
    inputs = {
        "period": {"type": "string", "description": "Musical period (e.g., Baroque, Classical, Romantic)"},
        "composer": {"type": "string", "description": "Composer style to emulate (e.g., Bach, Mozart, Chopin)"},
        "instrumentation": {"type": "string", "description": "Instruments to use (e.g., Piano, Violin, Orchestra)"}
    }
    output_type = "string"
    
    def __init__(self, device="cuda:0", output_dir=None):
        super().__init__(NotaGenTool, device=device, output_dir=output_dir)


class ConversationalTool(Tool):
    """Tool for general conversation that prompts users to ask about music-related topics."""
    name = "conversational_tool"
    description = "Engages in general conversation and prompts the user to ask about music-related topics when queries are not music-related."
    inputs = {
        "query": {
            "type": "string", 
            "description": "The user's non-music related query"
        }
    }
    output_type = "string"
    
    def __init__(self):
        super().__init__()
        conv_model_id = "Qwen/Qwen1.5-1.8B-Chat"
        self.conv_model = TransformersModel(
            model_id=conv_model_id, trust_remote_code=True, device_map="auto")
    
    def forward(self, query: str) -> str:        
        prompt = f"The user asked: '{query}'. This seems to be a general question not related to music. Please respond politely and encourage them to ask about music-related topics like audio generation, music analysis, or music creation."
        try:
            # Try using the correct message format for generate
            messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
            response = self.conv_model.generate(messages, max_new_tokens=150)
            if isinstance(response, str):
                return response
            else:
                return str(response)
        except Exception as e:
            # Fallback to a simple response
            return f"Hello! I see you asked about '{query}'. I'm a music assistant focused on helping with music-related tasks like audio generation, music analysis, and music creation. How can I help you with music today?"
