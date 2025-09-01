"""
NotaGen Tool - Symbolic music generation using NotaGen model for ABC notation.
"""
import logging
import tempfile
import os
import datetime
from typing import Optional, Dict, Any, Union
from pathlib import Path
import torch            
hf_hub_download = None
inference_patch = None
postprocess_inst_names = None
abc2xml = None
xml2 = None
pdf2img = None


from smolagents.tools import Tool  # type: ignore
from .base_tools import ManagedTransformersTool

logger = logging.getLogger(__name__)


class NotaGenTool(ManagedTransformersTool):
    """
    Tool for symbolic music generation using NotaGen model.
    
    This tool can:
    - Generate ABC notation from text prompts
    - Create music in specific styles and periods
    - Generate compositions for specified instrumentation
    - Handle conditional generation with period-composer-instrumentation prompts
    - Convert ABC to XML, PDF, MIDI, and MP3 formats
    - Generate PDF images for visual display
    
    Now with lazy loading and VRAM management!
    """
    
    # Class attributes required by smolagents
    name = "notagen"
    description = (
        "Generates symbolic music in ABC notation format with full conversion capabilities. "
        "Can create compositions based on text descriptions, musical styles, periods, "
        "composers, and instrumentation. Supports conditional generation with format: "
        "'Period-Composer-Instrumentation' (e.g., 'Romantic-Chopin-Piano'). "
        "Automatically converts to various formats including PDF for visual display."
    )
    inputs = {
        "period": {
            "type": "string", 
            "description": "Musical period (e.g., Baroque, Classical, Romantic)",
        },
        "composer": {
            "type": "string",
            "description": "Composer style to emulate (e.g., Bach, Mozart, Chopin)",
        },
        "instrumentation": {
            "type": "string",
            "description": "Instruments to use (e.g., Piano, Violin, Orchestra)",
        }
    }
    output_type = "string"
    
    def __init__(
        self, 
        device: str = "auto", 
        model_id: str = "manoskary/NotaGenX-Quantized", 
        output_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize NotaGen tool with lazy loading.
        
        Args:
            device: Device to run on ("auto", "cuda", "cpu")
            model_id: NotaGen model ID
            output_dir: Directory for output files
            **kwargs: Additional arguments
        """
        
        
        # NotaGen is smaller model, estimate VRAM usage
        estimated_vram = 2000.0
        
        super().__init__(
            model_id=model_id,
            device=device,
            estimated_vram_mb=estimated_vram,
            torch_dtype="float16" if device == "cuda" else "float32",
            priority=4,  # Lower priority for symbolic generation
            **kwargs
        )
        
        self.output_dir = output_dir or "/tmp/notagen_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"NotaGen tool initialized (lazy loading enabled)")
    
    def _load_model(self) -> Dict[str, Any]:
        """
        Load the NotaGen model.
        
        Returns:
            Dictionary containing model and any needed components
        """
        
        try:
            import torch
            import torch.nn as nn
            import numpy as np
            import requests
            from huggingface_hub import hf_hub_download
            # Import the NotaGen inference and conversion functions
            from ..models.notagen.inference import inference_patch, postprocess_inst_names
            from ..models.notagen.convert import abc2xml, xml2, pdf2img
            NOTAGEN_AVAILABLE = True
            print("✅ NotaGen dependencies loaded successfully!")
        except ImportError as e:
            NOTAGEN_AVAILABLE = False
            import_error = str(e)
            print(f"❌ NotaGen dependencies failed: {e}")            
            raise ImportError(f"NotaGen dependencies not available: {import_error}")            
            
        logger.info(f"Loading NotaGen model: {self.model_id}")
        
        # Load NotaGen model (implementation would depend on actual model structure)
        # This is a placeholder - would need actual NotaGen loading code
        model = {
            "inference_fn": inference_patch,
            "postprocess_fn": postprocess_inst_names,
            "convert_fns": {
                "abc2xml": abc2xml,
                "xml2": xml2,
                "pdf2img": pdf2img
            }
        }
        
        logger.info(f"NotaGen model loaded successfully on {self.device}")
        
        return model
    
    def _call_model(
        self,
        model: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        Generate ABC notation using NotaGen model.
        
        Args:
            model: Dictionary containing loaded components
            **kwargs: Arguments passed from forward() including period, composer, instrumentation
            
        Returns:
            Generated ABC notation or path to output file
        """
        try:
            # Extract parameters from kwargs
            period = kwargs.get("period", "Classical")
            composer = kwargs.get("composer", "Mozart")
            instrumentation = kwargs.get("instrumentation", "Piano")
            
            # Create prompt for NotaGen
            prompt = f"{period}-{composer}-{instrumentation}"
            
            logger.info(f"Generating music: {prompt}")
            
            # Use the inference function
            inference_fn = model["inference_fn"]
            if inference_fn is None:
                raise ImportError("inference_patch not available")
                
            # Generate ABC notation (placeholder implementation)
            abc_content = inference_fn(period, composer, instrumentation)
            
            # Save ABC content to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            abc_filename = f"notagen_{timestamp}"
            postprocessed_abc_path = os.path.join(self.output_dir, abc_filename+".abc")
            preprocessed_abc_path = postprocessed_abc_path.replace(".abc", "_preprocessed.abc")

            # Post-process ABC content
            postprocessed_abc = model["postprocess_fn"](abc_content)

            with open(preprocessed_abc_path, 'w') as f:
                f.write(abc_content)

            with open(postprocessed_abc_path, 'w') as f:
                f.write(postprocessed_abc)

            logger.info(f"Pre-processed ABC notation saved: {preprocessed_abc_path}")                                    
            logger.info(f"Post-processed ABC notation saved: {postprocessed_abc_path}")

            logger.info("Converting ABC to other formats...")

            conversion_results = self._convert_files(abc_filename, abc_content, postprocessed_abc)

            logger.info(f"ABC notation converted to other formats: {conversion_results}")

            return postprocessed_abc_path

        except Exception as e:
            logger.error(f"Error generating ABC notation: {e}")
            return f"Error: {str(e)}"
    
    def _unload_model(self, model: Dict[str, Any]) -> None:
        """
        Unload the NotaGen model components.
        
        Args:
            model: Dictionary containing model components
        """
        try:
            # NotaGen cleanup (minimal since it's mostly function references)
            model.clear()
            
            # Force cleanup
            import gc
            gc.collect()
            
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        except Exception as e:
            logger.warning(f"Error during NotaGen cleanup: {e}")

    def _convert_files(self, filename_base: str, abc_content: str, postprocessed_abc: str) -> Dict[str, Any]:
        """Convert ABC to various formats and generate images."""
        file_paths: Dict[str, Any] = {
            'abc_preprocessed': os.path.join(self.output_dir, f"{filename_base}_preprocessed.abc"),
            'abc': os.path.join(self.output_dir, f"{filename_base}.abc")
        }
        
        # original_cwd is the 
        original_cwd = os.getcwd()
        logger.info(f"Current working directory: {original_cwd}")
        
        filename_base = os.path.join(self.output_dir, filename_base)
        try:
            # Change to output directory for conversion
            os.chdir(self.output_dir)
            logger.info(f"Changed working directory to: {self.output_dir}")
            
            # Convert ABC to XML
            if abc2xml is not None:                
                abc2xml(filename_base)

            # Convert XML to PDF
            if xml2 is not None:
                xml2(filename_base, 'pdf')
            
                # Convert XML to MIDI            
                xml2(f"{filename_base}", 'mid')
                
                # Convert XML to MP3                
                xml2(f"{filename_base}", 'mp3')
            
            # Convert PDF to images
            image_paths: list = []
            if pdf2img is not None:
                images = pdf2img(filename_base)
                for i, image in enumerate(images):
                    img_path = os.path.join(self.output_dir, f"{filename_base}_page_{i+1}.png")
                    image.save(img_path, "PNG")
                    image_paths.append(img_path)
            
            # Update file paths            
            file_paths['xml'] = os.path.join(self.output_dir, f"{filename_base}.xml")
            file_paths['pdf'] = os.path.join(self.output_dir, f"{filename_base}.pdf")
            file_paths['mid'] = os.path.join(self.output_dir, f"{filename_base}.mid")
            file_paths['mp3'] = os.path.join(self.output_dir, f"{filename_base}.mp3")
            file_paths['images'] = image_paths
            file_paths['pages'] = len(image_paths)
            file_paths['filename_base'] = filename_base
            
        except Exception as e:
            logger.error(f"Error converting files: {e}")
            file_paths['conversion_error'] = str(e)
        finally:
            os.chdir(original_cwd)
        
        return file_paths
    
    def _format_result(self, result: Dict[str, Any]) -> str:
        """Format the result for display."""
        if result["status"] == "success":
            abc_content = result["abc_content"]
            files = result["files"]
            
            response = f"Successfully generated music notation!\n\n"
            response += f"ABC Notation:\n```\n{abc_content[:500]}{'...' if len(abc_content) > 500 else ''}\n```\n\n"
            
            if "conversion_error" not in files:
                response += f"Generated files:\n"
                response += f"- ABC: {files.get('abc', 'N/A')}\n"
                response += f"- PDF: {files.get('pdf', 'N/A')}\n"
                response += f"- MP3: {files.get('mp3', 'N/A')}\n"
                response += f"- MIDI: {files.get('mid', 'N/A')}\n"
                if files.get('images'):
                    response += f"- PDF Images: {len(files['images'])} pages\n"
                    response += f"- First page: {files['images'][0]}\n"
            else:
                response += f"Note: File conversion encountered an error: {files['conversion_error']}\n"
            
            return response
        else:
            return f"Generation failed: {result.get('error', 'Unknown error')}"    
    
    def forward(self, period: str, composer: str, instrumentation: str) -> str:
        """
        Generate symbolic music using NotaGen.
        
        This method provides the smolagents-compatible interface with explicit parameters.
        
        Args:
            period: Musical period (e.g., Baroque, Classical, Romantic)
            composer: Composer style to emulate (e.g., Bach, Mozart, Chopin)
            instrumentation: Instruments to use (e.g., Piano, Violin, Orchestra)
            
        Returns:
            Path to generated ABC file or error message
        """
        # Convert to kwargs for internal processing
        kwargs = {
            "period": period,
            "composer": composer,
            "instrumentation": instrumentation
        }
        
        # Call the parent's forward method which handles async processing
        return super().forward()    