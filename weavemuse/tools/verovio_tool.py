"""
Verovio Tool - Music notation visualization and rendering using Verovio.
"""

import logging
import tempfile
import os
import subprocess
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
    import requests
    from PIL import Image
    import base64
except ImportError as e:
    logging.warning(f"Some dependencies for VerovioTool are not available: {e}")


logger = logging.getLogger(__name__)


class VerovioTool(Tool):
    """
    Tool for music notation visualization using Verovio.
    
    This tool can:
    - Render ABC notation to SVG images
    - Convert ABC to other music formats
    - Create sheet music visualizations
    - Generate high-quality music notation images
    """
    
    def __init__(self):
        super().__init__(
            name="verovio",
            description=(
                "Renders music notation from ABC format into visual sheet music. "
                "Creates high-quality SVG and PNG images of musical scores. "
                "Can handle various musical elements including notes, rhythms, "
                "chords, and musical symbols."
            ),
            inputs={
                "abc_notation": {
                    "type": "string", 
                    "description": "ABC notation string to render"
                },
                "output_format": {
                    "type": "string", 
                    "description": "Output format: 'svg', 'png', 'pdf' (default: 'svg')",
                    "required": False
                },
                "page_width": {
                    "type": "string",
                    "description": "Page width in pixels (default: 1200)",
                    "required": False
                },
                "page_height": {
                    "type": "string",
                    "description": "Page height in pixels (default: 800)",
                    "required": False
                }
            },
            output_type="string"
        )
        
        self.verovio_available = self._check_verovio_availability()
        
    def _check_verovio_availability(self) -> bool:
        """Check if Verovio is available for rendering."""
        try:
            # Try to use verovio command line tool
            result = subprocess.run(
                ["verovio", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                logger.info("Verovio command line tool is available")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Try using online Verovio service (backup)
        try:
            response = requests.get("https://www.verovio.org/", timeout=5)
            if response.status_code == 200:
                logger.info("Verovio online service is available")
                return True
        except Exception:
            pass
        
        logger.warning("Verovio is not available - using fallback rendering")
        return False
    
    def forward(
        self, 
        abc_notation: str,
        output_format: str = "svg",
        page_width: str = "1200",
        page_height: str = "800"
    ) -> str:
        """
        Render ABC notation to visual music notation.
        
        Args:
            abc_notation: ABC notation string
            output_format: Output format (svg, png, pdf)
            page_width: Page width in pixels
            page_height: Page height in pixels
            
        Returns:
            Path to rendered image or error message
        """
        try:
            # Clean and validate ABC notation
            cleaned_abc = self._clean_abc_notation(abc_notation)
            
            if not self._validate_abc_notation(cleaned_abc):
                return "Invalid ABC notation provided"
            
            # Render the notation
            if self.verovio_available:
                output_path = self._render_with_verovio(
                    cleaned_abc, output_format, int(page_width), int(page_height)
                )
            else:
                output_path = self._render_fallback(
                    cleaned_abc, output_format, int(page_width), int(page_height)
                )
            
            if output_path:
                logger.info(f"Rendered ABC notation to: {output_path}")
                return f"Sheet music rendered to: {output_path}"
            else:
                return "Failed to render ABC notation"
            
        except Exception as e:
            logger.error(f"Error rendering ABC notation: {e}")
            return f"Error rendering notation: {str(e)}"
    
    def _clean_abc_notation(self, abc_notation: str) -> str:
        """Clean and format ABC notation."""
        lines = abc_notation.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        # Ensure we have at least basic ABC headers
        has_headers = any(line.startswith(('X:', 'T:', 'M:', 'L:', 'K:')) for line in cleaned_lines)
        
        if not has_headers:
            # Add minimal headers if missing
            header_lines = [
                "X:1",
                "T:Generated Music",
                "M:4/4",
                "L:1/8",
                "K:C"
            ]
            cleaned_lines = header_lines + cleaned_lines
        
        return '\n'.join(cleaned_lines)
    
    def _validate_abc_notation(self, abc_notation: str) -> bool:
        """Validate ABC notation format."""
        lines = abc_notation.split('\n')
        
        # Check for required headers
        has_x = any(line.startswith('X:') for line in lines)
        has_k = any(line.startswith('K:') for line in lines)
        
        if not (has_x and has_k):
            logger.warning("ABC notation missing required headers (X: and K:)")
            return False
        
        # Check for music content
        has_music = any(
            line and not line.startswith(('X:', 'T:', 'C:', 'M:', 'L:', 'K:', 'Q:', 'w:'))
            for line in lines
        )
        
        if not has_music:
            logger.warning("ABC notation appears to have no musical content")
            return False
        
        return True
    
    def _render_with_verovio(
        self, 
        abc_notation: str, 
        output_format: str,
        page_width: int,
        page_height: int
    ) -> Optional[str]:
        """Render using Verovio command line tool or online service."""
        
        try:
            # Try command line tool first
            return self._render_with_cli(abc_notation, output_format, page_width, page_height)
        except Exception as e:
            logger.warning(f"CLI rendering failed: {e}")
            
        try:
            # Fallback to online service
            return self._render_with_online_service(abc_notation, output_format, page_width, page_height)
        except Exception as e:
            logger.warning(f"Online rendering failed: {e}")
            
        return None
    
    def _render_with_cli(
        self, 
        abc_notation: str, 
        output_format: str,
        page_width: int,
        page_height: int
    ) -> str:
        """Render using Verovio command line interface."""
        
        # Create temporary files
        temp_dir = Path(tempfile.gettempdir()) / "music_agent_notation"
        temp_dir.mkdir(exist_ok=True)
        
        input_file = temp_dir / "input.abc"
        output_file = temp_dir / f"output.{output_format}"
        
        # Write ABC notation to file
        with open(input_file, 'w') as f:
            f.write(abc_notation)
        
        # Prepare verovio command
        cmd = [
            "verovio",
            str(input_file),
            "-o", str(output_file),
            "--page-width", str(page_width),
            "--page-height", str(page_height)
        ]
        
        if output_format == "svg":
            cmd.extend(["--svg-output"])
        elif output_format == "png":
            cmd.extend(["--png-output"])
        
        # Run verovio
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and output_file.exists():
            return str(output_file)
        else:
            raise Exception(f"Verovio CLI failed: {result.stderr}")
    
    def _render_with_online_service(
        self, 
        abc_notation: str, 
        output_format: str,
        page_width: int,
        page_height: int
    ) -> str:
        """Render using Verovio online service."""
        
        # This is a simplified example - you would need to implement
        # the actual Verovio online API integration
        logger.info("Using Verovio online service rendering")
        
        # For now, create a placeholder rendered file
        return self._create_placeholder_notation(abc_notation, output_format)
    
    def _render_fallback(
        self, 
        abc_notation: str, 
        output_format: str,
        page_width: int,
        page_height: int
    ) -> str:
        """Fallback rendering method when Verovio is not available."""
        
        logger.info("Using fallback notation rendering")
        return self._create_placeholder_notation(abc_notation, output_format)
    
    def _create_placeholder_notation(self, abc_notation: str, output_format: str) -> str:
        """Create a placeholder notation image when Verovio is not available."""
        
        try:
            # Create a simple text-based representation
            temp_dir = Path(tempfile.gettempdir()) / "music_agent_notation"
            temp_dir.mkdir(exist_ok=True)
            
            if output_format == "svg":
                output_file = temp_dir / "notation_placeholder.svg"
                self._create_svg_placeholder(abc_notation, output_file)
            else:
                output_file = temp_dir / "notation_placeholder.png" 
                self._create_png_placeholder(abc_notation, output_file)
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Failed to create placeholder: {e}")
            return None
    
    def _create_svg_placeholder(self, abc_notation: str, output_file: Path):
        """Create an SVG placeholder for the notation."""
        
        # Extract title and key information
        lines = abc_notation.split('\n')
        title = "Generated Music"
        key = "C"
        
        for line in lines:
            if line.startswith('T:'):
                title = line[2:].strip()
            elif line.startswith('K:'):
                key = line[2:].strip()
        
        # Create simple SVG
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
    <rect width="800" height="600" fill="white"/>
    <text x="400" y="50" text-anchor="middle" font-family="serif" font-size="24" font-weight="bold">{title}</text>
    <text x="50" y="100" font-family="serif" font-size="16">Key: {key}</text>
    
    <!-- Staff lines -->
    <g stroke="black" stroke-width="1">
        <line x1="50" y1="150" x2="750" y2="150"/>
        <line x1="50" y1="170" x2="750" y2="170"/>
        <line x1="50" y1="190" x2="750" y2="190"/>
        <line x1="50" y1="210" x2="750" y2="210"/>
        <line x1="50" y1="230" x2="750" y2="230"/>
    </g>
    
    <!-- Treble clef placeholder -->
    <text x="70" y="200" font-family="serif" font-size="40">𝄞</text>
    
    <!-- Note placeholder -->
    <circle cx="150" cy="190" r="8" fill="black"/>
    <circle cx="200" cy="170" r="8" fill="black"/>
    <circle cx="250" cy="150" r="8" fill="black"/>
    <circle cx="300" cy="170" r="8" fill="black"/>
    
    <text x="50" y="300" font-family="monospace" font-size="12" fill="gray">ABC Notation:</text>
    <text x="50" y="320" font-family="monospace" font-size="10" fill="gray">{abc_notation[:100]}...</text>
    
    <text x="400" y="550" text-anchor="middle" font-family="serif" font-size="12" fill="gray">
        Generated with Music Agent Framework (Verovio not available)
    </text>
</svg>'''
        
        with open(output_file, 'w') as f:
            f.write(svg_content)
    
    def _create_png_placeholder(self, abc_notation: str, output_file: Path):
        """Create a PNG placeholder for the notation."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Create image
            img = Image.new('RGB', (800, 600), 'white')
            draw = ImageDraw.Draw(img)
            
            # Try to use a font
            try:
                font_title = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", 24)
                font_text = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 16)
                font_mono = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 10)
            except:
                font_title = font_text = font_mono = ImageFont.load_default()
            
            # Extract title
            title = "Generated Music"
            for line in abc_notation.split('\n'):
                if line.startswith('T:'):
                    title = line[2:].strip()
                    break
            
            # Draw content
            draw.text((400, 30), title, fill='black', font=font_title, anchor="mt")
            
            # Draw staff lines
            for i in range(5):
                y = 150 + i * 20
                draw.line([(50, y), (750, y)], fill='black', width=1)
            
            # Draw placeholder notes
            note_positions = [(150, 190), (200, 170), (250, 150), (300, 170)]
            for x, y in note_positions:
                draw.ellipse([x-8, y-8, x+8, y+8], fill='black')
            
            # Draw ABC notation sample
            draw.text((50, 300), "ABC Notation:", fill='gray', font=font_text)
            abc_sample = abc_notation[:80] + "..." if len(abc_notation) > 80 else abc_notation
            draw.text((50, 320), abc_sample, fill='gray', font=font_mono)
            
            draw.text((400, 550), "Generated with Music Agent Framework", fill='gray', font=font_text, anchor="mt")
            
            # Save image
            img.save(output_file)
            
        except Exception as e:
            # Ultra-simple fallback
            logger.warning(f"Could not create PNG placeholder: {e}")
            with open(output_file, 'w') as f:
                f.write("Music notation placeholder - PNG generation failed")
    
    def render_abc_to_svg(self, abc_notation: str) -> str:
        """Render ABC notation to SVG format."""
        return self.forward(abc_notation, output_format="svg")
    
    def render_abc_to_png(self, abc_notation: str) -> str:
        """Render ABC notation to PNG format."""
        return self.forward(abc_notation, output_format="png")
    
    def create_sheet_music(self, abc_notation: str, title: Optional[str] = None) -> str:
        """Create formatted sheet music from ABC notation."""
        if title:
            # Add title to ABC notation if not present
            lines = abc_notation.split('\n')
            has_title = any(line.startswith('T:') for line in lines)
            if not has_title:
                # Insert title after X: header
                for i, line in enumerate(lines):
                    if line.startswith('X:'):
                        lines.insert(i + 1, f"T:{title}")
                        break
                abc_notation = '\n'.join(lines)
        
        return self.render_abc_to_svg(abc_notation)
