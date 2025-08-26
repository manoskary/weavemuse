"""
VRAM Memory Manager for WeaveMuse Tools

This module provides sophisticated memory management for ML models to prevent OOM errors
and optimize GPU utilization through lazy loading, LRU caching, and automatic unloading.
"""

import asyncio
import torch
import time
import logging
import psutil
try:
    import GPUtil
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    # Mock GPUtil for when it's not available
    class GPUtil:
        @staticmethod
        def getGPUs():
            return []
from collections import OrderedDict
from typing import Dict, Any, Optional, Callable, Union, List
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
import json


logger = logging.getLogger(__name__)


@dataclass
class ToolMetrics:
    """Metrics for tracking tool performance and resource usage."""
    name: str
    load_count: int = 0
    call_count: int = 0
    total_load_time: float = 0.0
    total_inference_time: float = 0.0
    last_load_time: float = 0.0
    last_inference_time: float = 0.0
    peak_vram_mb: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    last_used: float = field(default_factory=time.time)
    
    @property
    def avg_load_time(self) -> float:
        return self.total_load_time / max(1, self.load_count)
    
    @property
    def avg_inference_time(self) -> float:
        return self.total_inference_time / max(1, self.call_count)


class ManagedTool:
    """
    A wrapper for ML tools that provides lazy loading, automatic unloading,
    and resource tracking capabilities.
    """
    
    def __init__(
        self,
        name: str,
        loader: Callable[[], Any],
        caller: Callable[..., Any],
        unloader: Callable[[Any], None],
        estimted_vram_mb: float = 1000.0,
        priority: int = 1  # Higher priority = kept longer in cache
    ):
        self.name = name
        self.loader = loader
        self.caller = caller
        self.unloader = unloader
        self.estimated_vram_mb = estimted_vram_mb
        self.priority = priority
        
        self.model = None
        self.is_loading = False
        self.metrics = ToolMetrics(name=name)
        
        # Threading primitives
        self._load_lock = asyncio.Lock()
    
    async def ensure_loaded(self) -> Any:
        """Ensure the model is loaded, loading it if necessary."""
        if self.model is not None:
            self.metrics.last_used = time.time()
            return self.model
        
        async with self._load_lock:
            # Double-check pattern
            if self.model is not None:
                self.metrics.last_used = time.time()
                return self.model
            
            if self.is_loading:
                # Wait for another thread to finish loading
                while self.is_loading:
                    await asyncio.sleep(0.1)
                return self.model
            
            # Load the model
            self.is_loading = True
            try:
                start_time = time.time()
                logger.info(f"Loading model for tool: {self.name}")
                
                # Get VRAM before loading
                vram_before = self._get_gpu_memory_mb()
                
                # Load model in thread pool to avoid blocking
                self.model = await asyncio.to_thread(self.loader)
                
                # Track metrics
                load_time = time.time() - start_time
                vram_after = self._get_gpu_memory_mb()
                actual_vram = vram_after - vram_before
                
                self.metrics.load_count += 1
                self.metrics.total_load_time += load_time
                self.metrics.last_load_time = load_time
                self.metrics.peak_vram_mb = max(self.metrics.peak_vram_mb, actual_vram)
                self.metrics.last_used = time.time()
                
                logger.info(
                    f"Tool {self.name} loaded in {load_time:.2f}s, "
                    f"using ~{actual_vram:.1f}MB VRAM"
                )
                
                return self.model
                
            except Exception as e:
                self.metrics.error_count += 1
                logger.error(f"Failed to load tool {self.name}: {e}")
                raise
            finally:
                self.is_loading = False
    
    async def __call__(self, **kwargs) -> Any:
        """Execute the tool with the given arguments."""
        await self.ensure_loaded()
        
        start_time = time.time()
        try:
            # Execute in thread pool
            result = await asyncio.to_thread(self.caller, self.model, **kwargs)  # type: ignore
            
            # Track success metrics
            inference_time = time.time() - start_time
            self.metrics.call_count += 1
            self.metrics.total_inference_time += inference_time
            self.metrics.last_inference_time = inference_time
            self.metrics.last_used = time.time()
            
            # Update success rate (exponential moving average)
            self.metrics.success_rate = 0.95 * self.metrics.success_rate + 0.05 * 1.0
            
            return result
            
        except Exception as e:
            # Track error metrics
            self.metrics.error_count += 1
            self.metrics.success_rate = 0.95 * self.metrics.success_rate + 0.05 * 0.0
            logger.error(f"Error in tool {self.name}: {e}")
            raise
    
    async def unload(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            logger.info(f"Unloading tool: {self.name}")
            try:
                await asyncio.to_thread(self.unloader, self.model)
            except Exception as e:
                logger.warning(f"Error during unload of {self.name}: {e}")
            finally:
                self.model = None
                # Force garbage collection
                torch.cuda.empty_cache()
    
    def _get_gpu_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except:
            pass
        return 0.0


class VRAMManager:
    """
    Sophisticated VRAM manager that handles tool loading/unloading with LRU caching,
    resource monitoring, and intelligent scheduling.
    """
    
    def __init__(
        self,
        max_loaded_tools: int = 2,
        max_vram_mb: Optional[float] = None,
        metrics_file: Optional[str] = None
    ):
        self.max_loaded_tools = max_loaded_tools
        self.max_vram_mb = max_vram_mb or self._get_max_vram_mb()
        self.metrics_file = Path(metrics_file) if metrics_file else None
        
        self.tools: OrderedDict[str, ManagedTool] = OrderedDict()
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(max_loaded_tools)
        
        # Routing table for intelligent tool selection
        self.routing_table: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            f"VRAMManager initialized: max_tools={max_loaded_tools}, "
            f"max_vram={self.max_vram_mb:.0f}MB"
        )
    
    def register_tool(self, tool: ManagedTool) -> None:
        """Register a tool with the manager."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    @asynccontextmanager
    async def acquire_tool(self, name: str):
        """Context manager to acquire and use a tool safely."""
        async with self.semaphore:  # Limit concurrent tools
            async with self.lock:
                tool = await self._ensure_capacity_for_tool(name)
            
            try:
                yield tool
            finally:
                # Update LRU order
                async with self.lock:
                    if name in self.tools:
                        # Move to end (most recently used)
                        self.tools.move_to_end(name)
    
    async def _ensure_capacity_for_tool(self, name: str) -> ManagedTool:
        """Ensure there's capacity to load the requested tool."""
        if name not in self.tools:
            raise ValueError(f"Tool {name} not registered")
        
        tool = self.tools[name]
        
        # If already loaded, just update LRU order
        if tool.model is not None:
            self.tools.move_to_end(name)
            return tool
        
        # Check if we need to evict tools
        loaded_tools = [t for t in self.tools.values() if t.model is not None]
        
        # Evict by count
        if len(loaded_tools) >= self.max_loaded_tools:
            await self._evict_lru_tool(exclude=name)
        
        # Evict by VRAM if needed
        current_vram = self._get_current_vram_mb()
        if current_vram + tool.estimated_vram_mb > self.max_vram_mb:
            await self._evict_by_vram(required_mb=tool.estimated_vram_mb, exclude=name)
        
        return tool
    
    async def _evict_lru_tool(self, exclude: Optional[str] = None) -> None:
        """Evict the least recently used tool."""
        candidates = [
            t for t in self.tools.values() 
            if t.model is not None and t.name != exclude
        ]
        
        if not candidates:
            return
        
        # Sort by priority (lower first) then by last used time
        victim = min(candidates, key=lambda t: (t.priority, t.metrics.last_used))
        logger.info(f"Evicting LRU tool: {victim.name}")
        await victim.unload()
    
    async def _evict_by_vram(self, required_mb: float, exclude: Optional[str] = None) -> None:
        """Evict tools until we have enough VRAM."""
        candidates = [
            t for t in self.tools.values() 
            if t.model is not None and t.name != exclude
        ]
        
        # Sort by priority (lower first) then by last used time
        candidates.sort(key=lambda t: (t.priority, t.metrics.last_used))
        
        current_vram = self._get_current_vram_mb()
        target_vram = current_vram + required_mb
        
        for tool in candidates:
            if current_vram + required_mb <= self.max_vram_mb:
                break
            
            logger.info(f"Evicting tool {tool.name} to free VRAM")
            await tool.unload()
            current_vram = self._get_current_vram_mb()
    
    def _get_current_vram_mb(self) -> float:
        """Get current VRAM usage in MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except:
            pass
        return 0.0
    
    def _get_max_vram_mb(self) -> float:
        """Get maximum available VRAM in MB."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                # Use 80% of available VRAM to leave headroom
                return gpus[0].memoryTotal * 0.8
        except:
            pass
        
        # Fallback: assume 8GB if we can't detect
        return 8192.0
    
    async def unload_all(self) -> None:
        """Unload all tools."""
        async with self.lock:
            for tool in self.tools.values():
                await tool.unload()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics for all tools."""
        summary = {
            "manager": {
                "max_loaded_tools": self.max_loaded_tools,
                "max_vram_mb": self.max_vram_mb,
                "current_vram_mb": self._get_current_vram_mb(),
                "loaded_tools": [
                    name for name, tool in self.tools.items() 
                    if tool.model is not None
                ]
            },
            "tools": {}
        }
        
        for name, tool in self.tools.items():
            metrics = tool.metrics
            summary["tools"][name] = {
                "load_count": metrics.load_count,
                "call_count": metrics.call_count,
                "avg_load_time": metrics.avg_load_time,
                "avg_inference_time": metrics.avg_inference_time,
                "success_rate": metrics.success_rate,
                "error_count": metrics.error_count,
                "peak_vram_mb": metrics.peak_vram_mb,
                "is_loaded": tool.model is not None
            }
        
        return summary
    
    async def save_metrics(self) -> None:
        """Save metrics to file if configured."""
        if self.metrics_file:
            metrics = self.get_metrics_summary()
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)


# Utility functions for model unloading
def unload_transformers_model(model) -> None:
    """Safely unload a transformers model."""
    try:
        # Get device before deletion
        device = next(model.parameters()).device
        logger.debug(f"Unloading model from device: {device}")
    except (StopIteration, AttributeError):
        pass
    
    # Delete model
    del model
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def unload_diffusers_pipeline(pipeline) -> None:
    """Safely unload a diffusers pipeline."""
    try:
        # Move components to CPU if needed
        if hasattr(pipeline, 'to'):
            pipeline.to('cpu')
        
        # Clear individual components
        for component_name in ['unet', 'vae', 'text_encoder', 'scheduler']:
            if hasattr(pipeline, component_name):
                component = getattr(pipeline, component_name)
                if component is not None:
                    del component
                    setattr(pipeline, component_name, None)
    
    except Exception as e:
        logger.warning(f"Error during pipeline component cleanup: {e}")
    
    # Delete pipeline
    del pipeline
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# Global instance for the application
vram_manager = VRAMManager(
    max_loaded_tools=2,
    metrics_file="~/.weavemuse/metrics.json"
)
