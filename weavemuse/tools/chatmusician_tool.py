"""
ChatMusician Tool - Music understanding and analysis using ChatMusician model.
"""

import logging
import tempfile
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
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    import torchaudio
    import re
    from string import Template
except ImportError as e:
    logging.warning(f"Some dependencies for ChatMusicianTool are not available: {e}")


logger = logging.getLogger(__name__)


class ChatMusicianTool(Tool):
    """
    Tool for music understanding and analysis using the ChatMusician model.
    
    This tool can:
    - Analyze music theory and harmony
    - Understand musical structures and forms
    - Provide composition guidance
    - Convert between different music representations
    - Generate music based on descriptions
    """
    
    def __init__(self, device: str = "auto", model_id: str = "m-a-p/ChatMusician"):
        super().__init__(
            name="chatmusician",
            description=(
                "Analyzes and understands music using natural language. "
                "Can analyze harmony, musical structure, provide composition guidance, "
                "and generate music based on text descriptions. Supports ABC notation "
                "and can work with chord progressions, melodies, and musical forms."
            ),
            inputs={
                "query": {
                    "type": "string", 
                    "description": "The music-related question or task"
                },
                "music_content": {
                    "type": "string", 
                    "description": "Optional ABC notation or music content to analyze",
                    "required": False
                },
                "context": {
                    "type": "string",
                    "description": "Optional additional context for the query",
                    "required": False
                }
            },
            output_type="string"
        )
        
        self.device = device
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ChatMusician model and tokenizer."""
        try:
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading ChatMusician model: {self.model_id}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            
            # Load model
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": self.device if self.device == "cuda" else None,
                "resume_download": True
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                **model_kwargs
            ).eval()
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set up generation configuration
            self.generation_config = GenerationConfig(
                temperature=0.2,
                top_k=40,
                top_p=0.9,
                do_sample=True,
                num_beams=1,
                repetition_penalty=1.1,
                min_new_tokens=10,
                max_new_tokens=1536
            )
            
            logger.info("ChatMusician model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatMusician model: {e}")
            self.model = None
            self.tokenizer = None
    
    def forward(
        self, 
        query: str, 
        music_content: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Process a music-related query using ChatMusician.
        
        Args:
            query: The music question or task
            music_content: Optional ABC notation or music content
            context: Optional additional context
            
        Returns:
            ChatMusician's response
        """
        if self.model is None or self.tokenizer is None:
            return "ChatMusician model is not available. Please check the installation."
        
        try:
            # Prepare the prompt
            prompt = self._prepare_prompt(query, music_content, context)
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                response = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    eos_token_id=self.tokenizer.eos_token_id,
                    generation_config=self.generation_config,
                )
            
            # Decode response
            response_text = self.tokenizer.decode(
                response[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            # Post-process response
            response_text = self._post_process_response(response_text)
            
            logger.info(f"ChatMusician processed query: {query[:50]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"Error processing ChatMusician query: {e}")
            return f"I encountered an error while processing your music query: {str(e)}"
    
    def _prepare_prompt(
        self, 
        query: str, 
        music_content: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """Prepare the prompt for ChatMusician."""
        
        prompt_template = Template("Human: ${inst} </s> Assistant: ")
        
        instruction_parts = []
        
        if context:
            instruction_parts.append(f"Context: {context}")
        
        if music_content:
            instruction_parts.append(f"Music content to analyze:\n{music_content}")
        
        instruction_parts.append(f"Query: {query}")
        
        if music_content and "abc" in music_content.lower():
            instruction_parts.append(
                "Please analyze the provided ABC notation and respond appropriately."
            )
        
        instruction = "\n\n".join(instruction_parts)
        return prompt_template.safe_substitute({"inst": instruction})
    
    def _post_process_response(self, response: str) -> str:
        """Post-process the model response."""
        # Remove extra whitespace and clean up formatting
        response = response.strip()
        
        # Extract ABC notation if present
        abc_pattern = r'(X:\d+\n(?:[^\n]*\n)+)'
        abc_matches = re.findall(abc_pattern, response + '\n')
        
        if abc_matches:
            # If ABC notation is generated, format it nicely
            for abc_match in abc_matches:
                formatted_abc = self._format_abc_notation(abc_match)
                response = response.replace(abc_match, formatted_abc)
        
        return response
    
    def _format_abc_notation(self, abc_notation: str) -> str:
        """Format ABC notation for better readability."""
        lines = abc_notation.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Add proper spacing for ABC headers
                if line.startswith(('X:', 'T:', 'M:', 'L:', 'K:', 'Q:')):
                    if formatted_lines and not formatted_lines[-1].startswith(('X:', 'T:', 'M:', 'L:', 'K:', 'Q:')):
                        formatted_lines.append('')
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def analyze_harmony(self, music_content: str) -> str:
        """Analyze harmony in the given music content."""
        query = "Analyze the harmony and chord progressions in this music."
        return self.forward(query, music_content=music_content)
    
    def analyze_structure(self, music_content: str) -> str:
        """Analyze musical structure and form."""
        query = "Analyze the musical structure, form, and overall organization of this piece."
        return self.forward(query, music_content=music_content)
    
    def generate_composition(
        self, 
        description: str, 
        style: Optional[str] = None,
        constraints: Optional[str] = None
    ) -> str:
        """Generate a musical composition based on description."""
        query_parts = [f"Compose music based on this description: {description}"]
        
        if style:
            query_parts.append(f"Style: {style}")
        if constraints:
            query_parts.append(f"Constraints: {constraints}")
        
        query = " | ".join(query_parts)
        return self.forward(query)
    
    def explain_theory(self, concept: str) -> str:
        """Explain a music theory concept."""
        query = f"Explain the music theory concept: {concept}"
        return self.forward(query)
