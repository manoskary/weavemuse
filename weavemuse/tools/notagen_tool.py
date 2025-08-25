"""
NotaGen Tool - Symbolic music generation using NotaGen model for ABC notation.
"""
import logging
import tempfile
import os
import datetime
from typing import Optional, Dict, Any, Union
from pathlib import Path

from smolagents.tools import Tool

import torch
import torch.nn as nn
import numpy as np
import requests
from huggingface_hub import hf_hub_download
# Import the NotaGen inference and conversion functions
from ..models.notagen.inference import inference_patch, postprocess_inst_names
from ..models.notagen.convert import abc2xml, xml2, pdf2img
NOTAGEN_AVAILABLE = True



logger = logging.getLogger(__name__)


class NotaGenTool(Tool):
    """
    Tool for symbolic music generation using NotaGen model.
    
    This tool can:
    - Generate ABC notation from text prompts
    - Create music in specific styles and periods
    - Generate compositions for specified instrumentation
    - Handle conditional generation with period-composer-instrumentation prompts
    - Convert ABC to XML, PDF, MIDI, and MP3 formats
    - Generate PDF images for visual display
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
    
    def __init__(self, device: str = "auto", output_dir: Optional[str] = None):
        # Initialize the Tool with proper parameters
        super().__init__()
        
        self.device = device
        self.output_dir = output_dir or "/tmp/notagen_output"
        self.model_initialized = False
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not NOTAGEN_AVAILABLE:
            logger.warning("NotaGen dependencies not available. Tool will use fallback mode.")
        else:
            self.model_initialized = True
            logger.info("NotaGen model initialized successfully")
    
    def forward(
        self, 
        period: str,
        composer: str,
        instrumentation: str
    ):
        """
        Generate ABC notation and convert to multiple formats.
        
        Args:
            period: Musical period (e.g., Baroque, Classical, Romantic)
            composer: Composer to emulate (e.g., Bach, Mozart, Chopin)
            instrumentation: Instruments to use (e.g., Piano, Violin, Orchestra)
            
        Returns:
            Path to generated files and ABC notation
        """
        if not NOTAGEN_AVAILABLE:
            return self._generate_fallback_abc(period, composer, instrumentation)
        
        try:
            logger.info(f"Generating music: {period}-{composer}-{instrumentation}")
            
            # Generate ABC notation using NotaGen
            if inference_patch is not None:
                abc_content = inference_patch(period, composer, instrumentation)
            else:
                raise RuntimeError("inference_patch function not available")
            
            # Post-process instrument names
            if postprocess_inst_names is not None:
                postprocessed_abc = postprocess_inst_names(abc_content)
            else:
                postprocessed_abc = abc_content
            
            # Create unique filename based on timestamp and parameters
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_str = f"{period}_{composer}_{instrumentation}"
            filename_base = f"{timestamp}_{prompt_str}"
            
            # Save ABC files
            abc_path = os.path.join(self.output_dir, f"{filename_base}.abc")
            postprocessed_abc_path = os.path.join(self.output_dir, f"{filename_base}_postinst.abc")
            
            with open(abc_path, "w", encoding="utf-8") as f:
                f.write(abc_content)
            
            with open(postprocessed_abc_path, "w", encoding="utf-8") as f:
                f.write(postprocessed_abc)
            
            # Convert to various formats
            file_paths = self._convert_files(filename_base, abc_content, postprocessed_abc)
            
            # Return comprehensive result
            result = {
                "abc_content": abc_content,
                "postprocessed_abc": postprocessed_abc,
                "files": file_paths,
                "status": "success"
            }
            
            return self._format_result(result)
            
        except Exception as e:
            logger.error(f"Error generating music with NotaGen: {e}")
            return f"Error generating music: {str(e)}"
    
    def _convert_files(self, filename_base: str, abc_content: str, postprocessed_abc: str) -> Dict[str, Any]:
        """Convert ABC to various formats and generate images."""
        file_paths: Dict[str, Any] = {
            'abc_preprocessed': os.path.join(self.output_dir, f"{filename_base}_preprocessed.abc"),
            'abc': os.path.join(self.output_dir, f"{filename_base}.abc")
        }
        
        original_cwd = os.getcwd()
        filename_base = os.path.join(self.output_dir, filename_base)
        try:
            # Change to output directory for conversion
            os.chdir(self.output_dir)
            
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
    
    def _generate_fallback_abc(self, period: str, composer: str, instrumentation: str) -> str:
        """Generate ABC notation using templates (fallback method)."""
        logger.info(f"Using fallback ABC generation for: {period}-{composer}-{instrumentation}")
        
        # Generate based on style and composer
        if composer.lower() == "bach":
            abc_notation = self._generate_bach_style(instrumentation, 2)
        elif composer.lower() == "chopin":
            abc_notation = self._generate_chopin_style(instrumentation, 2)
        elif composer.lower() == "mozart":
            abc_notation = self._generate_mozart_style(instrumentation, 2)
        else:
            abc_notation = self._generate_generic_style(instrumentation, 2)
        
        return f"Generated ABC notation (fallback mode):\n```\n{abc_notation}\n```"
    
    def _generate_bach_style(self, instrumentation: str, length_mult: int) -> str:
        """Generate Bach-style ABC notation."""
        header = """X:1
T:Generated Piece in Bach Style
C:J.S. Bach (AI Generated)
M:4/4
L:1/8
K:C"""
        
        # Simple Bach-style melody pattern
        melody_patterns = [
            "CDEF GABC | d2c2 B2A2 | GFED C4 |",
            "G2AB c2d2 | e2dc B2AG | F2E2 D4 |",
            "cBcd efga | f2ed c2BA | G2F2 E4 |"
        ]
        
        melody = "\n".join(melody_patterns[:length_mult])
        return f"{header}\n{melody}"
    
    def _generate_chopin_style(self, instrumentation: str, length_mult: int) -> str:
        """Generate Chopin-style ABC notation."""
        header = """X:1
T:Generated Piece in Chopin Style
C:F. Chopin (AI Generated)
M:3/4
L:1/8
K:Am"""
        
        # Simple Chopin-style waltz pattern
        melody_patterns = [
            "A2 c2e2 | g2f2e2 | d2c2B2 | A4 z2 |",
            "e2 g2a2 | b2a2g2 | f2e2d2 | c4 z2 |",
            "c2 e2g2 | a2g2f2 | e2d2c2 | B4 z2 |"
        ]
        
        melody = "\n".join(melody_patterns[:length_mult])
        return f"{header}\n{melody}"
    
    def _generate_mozart_style(self, instrumentation: str, length_mult: int) -> str:
        """Generate Mozart-style ABC notation."""
        header = """X:1
T:Generated Piece in Mozart Style
C:W.A. Mozart (AI Generated)
M:4/4
L:1/8
K:G"""
        
        # Simple Mozart-style melody
        melody_patterns = [
            "G2AB c2d2 | e2dc B2AG | A2GF G4 |",
            "d2ef g2a2 | b2ag f2ed | c2BA G4 |",
            "B2cd e2f2 | g2fe d2cb | A2GF G4 |"
        ]
        
        melody = "\n".join(melody_patterns[:length_mult])
        return f"{header}\n{melody}"
    
    def _generate_generic_style(self, instrumentation: str, length_mult: int) -> str:
        """Generate generic classical-style ABC notation."""
        header = """X:1
T:Generated Classical Piece
C:AI Generated
M:4/4
L:1/8
K:C"""
        
        melody_patterns = [
            "CDEF GABc | d2c2 B2A2 | G2F2 E2D2 | C4 z4 |",
            "G2AB c2d2 | e2d2 c2B2 | A2G2 F2E2 | D4 z4 |",
            "c2de f2g2 | a2g2 f2e2 | d2c2 B2A2 | G4 z4 |"
        ]
        
        melody = "\n".join(melody_patterns[:length_mult])
        return f"{header}\n{melody}"
    
    def generate_from_description(self, description: str) -> str:
        """Generate music from a natural language description."""
        # Parse description to extract period, composer, instrumentation
        parts = description.split()
        period = "Classical"
        composer = "Bach"
        instrumentation = "Piano"
        
        # Simple keyword extraction
        if "romantic" in description.lower():
            period = "Romantic"
        elif "baroque" in description.lower():
            period = "Baroque"
        
        if "chopin" in description.lower():
            composer = "Chopin"
        elif "mozart" in description.lower():
            composer = "Mozart"
        
        if "violin" in description.lower():
            instrumentation = "Violin"
        elif "orchestra" in description.lower():
            instrumentation = "Orchestra"
        
        return self.forward(period, composer, instrumentation)
    
    def generate_in_style(self, style: str, composer: str, instrumentation: str) -> str:
        """Generate music in a specific style."""
        return self.forward(style, composer, instrumentation)
