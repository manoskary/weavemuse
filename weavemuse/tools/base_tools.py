"""
Base classes for VRAM-managed tools that integrate with smolagents.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from smolagents.tools import Tool

from .memory_manager import ManagedTool, vram_manager, unload_transformers_model


logger = logging.getLogger(__name__)


class LazyLoadableTool(Tool, ABC):
    """
    Base class for tools that support lazy loading and VRAM management.
    
    This class extends smolagents.Tool with lazy loading capabilities,
    automatic model management, and resource tracking.
    """
    
    def __init__(
        self,
        device: str = "auto",
        estimated_vram_mb: float = 1000.0,
        priority: int = 1,
        **kwargs
    ):
        """
        Initialize the lazy loadable tool.
        
        Args:
            device: Device to load models on ("auto", "cuda", "cpu")
            estimated_vram_mb: Estimated VRAM usage in MB
            priority: Tool priority (higher = kept longer in cache)
            **kwargs: Additional arguments passed to Tool
        """
        super().__init__()
        
        self.device = self._resolve_device(device)
        self.estimated_vram_mb = estimated_vram_mb
        self.priority = priority
        self._managed_tool: Optional[ManagedTool] = None
        
        # Register with VRAM manager after initialization
        self._register_with_manager()
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _register_with_manager(self) -> None:
        """Register this tool with the VRAM manager."""
        self._managed_tool = ManagedTool(
            name=self.name,
            loader=self._load_model,
            caller=self._call_model,
            unloader=self._unload_model,
            estimted_vram_mb=self.estimated_vram_mb,
            priority=self.priority
        )
        vram_manager.register_tool(self._managed_tool)
        logger.info(f"Registered {self.name} with VRAM manager")
    
    @abstractmethod
    def _load_model(self) -> Any:
        """
        Load the model synchronously.
        
        This method should load and return the model/pipeline.
        Called in a thread pool by the VRAM manager.
        
        Returns:
            The loaded model object
        """
        pass
    
    @abstractmethod
    def _call_model(self, model: Any, **kwargs) -> Any:
        """
        Call the model synchronously.
        
        Args:
            model: The loaded model object
            **kwargs: Arguments passed from forward()
            
        Returns:
            The model output
        """
        pass
    
    def _unload_model(self, model: Any) -> None:
        """
        Unload the model synchronously.
        
        Default implementation works for most transformers models.
        Override for custom cleanup logic.
        
        Args:
            model: The model to unload
        """
        unload_transformers_model(model)
    
    async def _async_forward(self, **kwargs) -> Any:
        """
        Asynchronous version of forward that uses the VRAM manager.
        
        Args:
            **kwargs: Arguments to pass to the model
            
        Returns:
            Model output
        """
        async with vram_manager.acquire_tool(self.name) as tool:
            return await tool(**kwargs)
    
    def forward(self, **kwargs) -> Any:
        """
        Synchronous forward method required by smolagents.
        
        This method bridges between smolagents' synchronous interface
        and our asynchronous VRAM management system.
        
        Args:
            **kwargs: Arguments to pass to the model
            
        Returns:
            Model output
        """
        # Run the async method in the current event loop or create one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, but smolagents expects sync
                # We need to run in a new thread to avoid blocking
                import concurrent.futures
                import threading
                
                result = None
                exception = None
                
                def run_async():
                    nonlocal result, exception
                    new_loop = None
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        result = new_loop.run_until_complete(self._async_forward(**kwargs))
                    except Exception as e:
                        exception = e
                    finally:
                        if new_loop:
                            new_loop.close()
                
                thread = threading.Thread(target=run_async)
                thread.start()
                thread.join()
                
                if exception:
                    raise exception
                return result
            else:
                # No running loop, safe to use run_until_complete
                return loop.run_until_complete(self._async_forward(**kwargs))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self._async_forward(**kwargs))
    
    async def unload(self) -> None:
        """Manually unload this tool."""
        if self._managed_tool:
            await self._managed_tool.unload()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this tool."""
        if self._managed_tool:
            return {
                "load_count": self._managed_tool.metrics.load_count,
                "call_count": self._managed_tool.metrics.call_count,
                "avg_load_time": self._managed_tool.metrics.avg_load_time,
                "avg_inference_time": self._managed_tool.metrics.avg_inference_time,
                "success_rate": self._managed_tool.metrics.success_rate,
                "error_count": self._managed_tool.metrics.error_count,
                "peak_vram_mb": self._managed_tool.metrics.peak_vram_mb,
                "is_loaded": self._managed_tool.model is not None
            }
        return {}
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return bool(self._managed_tool and self._managed_tool.model is not None)


class ManagedTransformersTool(LazyLoadableTool):
    """
    Base class for tools using Hugging Face transformers models.
    
    Provides common patterns for loading transformers models with
    proper device handling and memory optimization.
    """
    
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        estimated_vram_mb: float = 2000.0,
        torch_dtype: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        **kwargs
    ):
        """
        Initialize managed transformers tool.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to load on
            estimated_vram_mb: Estimated VRAM usage
            torch_dtype: Torch data type ("float16", "bfloat16", etc.)
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
            **kwargs: Additional arguments
        """
        self.model_id = model_id
        self.torch_dtype = self._resolve_torch_dtype(torch_dtype)
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        
        # Adjust VRAM estimate based on quantization
        if load_in_4bit:
            estimated_vram_mb *= 0.25
        elif load_in_8bit:
            estimated_vram_mb *= 0.5
        
        super().__init__(
            device=device,
            estimated_vram_mb=estimated_vram_mb,
            **kwargs
        )
    
    def _resolve_torch_dtype(self, torch_dtype: Optional[str]):
        """Resolve torch dtype string to actual dtype."""
        if not torch_dtype:
            return None
        
        try:
            import torch
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
                "int8": torch.int8,
            }
            return dtype_map.get(torch_dtype, torch.float16)
        except ImportError:
            return None
    
    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for model loading."""
        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if self.torch_dtype:
            kwargs["torch_dtype"] = self.torch_dtype
        
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
        
        # Add quantization config if requested
        if self.load_in_4bit or self.load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=self.load_in_4bit,
                    load_in_8bit=self.load_in_8bit,
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4" if self.load_in_4bit else "fp4",
                )
            except ImportError:
                logger.warning("BitsAndBytesConfig not available, skipping quantization")
        
        return kwargs


class ManagedDiffusersTool(LazyLoadableTool):
    """
    Base class for tools using Diffusers pipelines.
    
    Provides common patterns for loading diffusers pipelines with
    proper memory management and optimization.
    """
    
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        estimated_vram_mb: float = 3000.0,
        torch_dtype: Optional[str] = "float16",
        enable_memory_efficient_attention: bool = True,
        enable_vae_slicing: bool = True,
        **kwargs
    ):
        """
        Initialize managed diffusers tool.
        
        Args:
            model_id: HuggingFace model identifier
            device: Device to load on
            estimated_vram_mb: Estimated VRAM usage
            torch_dtype: Torch data type
            enable_memory_efficient_attention: Enable memory efficient attention
            enable_vae_slicing: Enable VAE slicing for memory efficiency
            **kwargs: Additional arguments
        """
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        self.enable_vae_slicing = enable_vae_slicing
        
        super().__init__(
            device=device,
            estimated_vram_mb=estimated_vram_mb,
            **kwargs
        )
    
    def _optimize_pipeline(self, pipeline) -> None:
        """Apply memory optimizations to the pipeline."""
        try:
            if self.enable_memory_efficient_attention and hasattr(pipeline, 'enable_attention_slicing'):
                pipeline.enable_attention_slicing(1)
            
            if self.enable_vae_slicing and hasattr(pipeline, 'enable_vae_slicing'):
                pipeline.enable_vae_slicing()
            
            # Enable model CPU offload if available
            if hasattr(pipeline, 'enable_model_cpu_offload') and self.device == "cuda":
                pipeline.enable_model_cpu_offload()
                
        except Exception as e:
            logger.warning(f"Could not apply all optimizations: {e}")
    
    def _unload_model(self, model: Any) -> None:
        """Custom unloader for diffusers pipelines."""
        from .memory_manager import unload_diffusers_pipeline
        unload_diffusers_pipeline(model)
