"""
Music Agent - A comprehensive agent for music understanding and generation.

This module provides a unified interface for various music-related tasks including
music understanding, symbolic music generation, audio analysis, and audio generation.
"""

import logging
from typing import List, Optional, Dict, Any
import warnings
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel
import os


# Import tools with fallbacks for missing dependencies
try:
    from ..tools import (
        ChatMusicianTool,
        NotaGenTool, 
        StableAudioTool,
        AudioAnalysisTool,
        MusicTheoryTool
    )
    HEAVY_TOOLS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some tools unavailable due to missing dependencies: {e}")
    # Import only lightweight tools
    try:
        from ..tools import MusicTheoryTool
    except ImportError:
        MusicTheoryTool = None
    ChatMusicianTool = None
    NotaGenTool = None
    StableAudioTool = None
    AudioAnalysisTool = None
    HEAVY_TOOLS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


class DummyLLMModel:
    def __init__(self):
        self.name = "DummyModel"

    def generate(self):
        return "Generated output from DummyModel"
    

class MusicAgent:
    """
    A comprehensive music agent that combines multiple AI models and tools
    for music understanding, generation, and analysis.
    
    Features:
    - Music understanding using natural language (ChatMusician)
    - Symbolic music generation in ABC notation (NotaGen)
    - Audio generation (Stable Audio)
    - Audio analysis and understanding
    - Music theory knowledge and assistance
    """
    
    def __init__(
        self,
        model_id: str = "microsoft/DialoGPT-medium",
        device: str = "auto",
        config: Optional[Dict] = None,
        tools: Optional[List] = None,
        enable_audio_tools: bool = True,
        enable_stable_audio: bool = True,
        enable_chat_musician: bool = True,
        **kwargs
    ):
        """
        Initialize the Music Agent.
        
        Args:
            model_id: The model to use for the agent
            device: Device to run models on ("cpu", "cuda", or "auto")
            config: Configuration dictionary
            tools: List of custom tools to use
            enable_audio_tools: Whether to enable audio processing tools
            enable_stable_audio: Whether to enable Stable Audio generation
            enable_chat_musician: Whether to enable ChatMusician
            **kwargs: Additional arguments
        """
        # Set up basic properties
        self.model_id = model_id
        self.device = device
        self.config = config or {}
        
        # Store capability flags
        self.enable_audio_tools = enable_audio_tools and HEAVY_TOOLS_AVAILABLE
        self.enable_stable_audio = enable_stable_audio and HEAVY_TOOLS_AVAILABLE
        self.enable_chat_musician = enable_chat_musician and HEAVY_TOOLS_AVAILABLE
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # For now, use a simple approach without smolagents for compatibility
        self.tools_registry = self._initialize_tools()
        
        # Initialize simple agent registry for testing
        self.agent_registry = {}
        self.manager_agent = None  # Initialize as None for now
        self.use_advanced_routing = False  # Disable advanced routing for now
        
        self.logger.info(f"Music Agent initialized with {len(self.tools_registry)} tools")
        
        # self.manager_agent = CodeAgent(
        #     model=InferenceClientModel("deepseek-ai/DeepSeek-R1", provider="together", max_tokens=8096),
        #     tools=self.tools_registry,
        #     managed_agents=self.agent_registry,
        #     additional_authorized_imports=[
        #         "geopandas",
        #         "plotly",
        #         "shapely",
        #         "json",
        #         "pandas",
        #         "numpy",
        #     ],
        #     planning_interval=5,
        #     verbosity_level=2,
        #     # final_answer_checks=[check_reasoning_and_plot],
        #     max_steps=15,
        # )
        
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools based on dependencies."""
        tools = {}
        
        # Always try to include music theory tool (lightweight)
        if MusicTheoryTool:
            tools['music_theory'] = MusicTheoryTool()
        
        if HEAVY_TOOLS_AVAILABLE:
            if self.enable_chat_musician and ChatMusicianTool:
                try:
                    tools['chat_musician'] = ChatMusicianTool(device=self.device)
                except Exception as e:
                    self.logger.warning(f"Failed to initialize ChatMusician: {e}")
            
            if self.enable_stable_audio and StableAudioTool:
                try:
                    tools['stable_audio'] = StableAudioTool(device=self.device)
                except Exception as e:
                    self.logger.warning(f"Failed to initialize StableAudio: {e}")
                    
            if NotaGenTool:
                try:
                    tools['nota_gen'] = NotaGenTool(device=self.device)
                except Exception as e:
                    self.logger.warning(f"Failed to initialize NotaGen: {e}")
                    
            if self.enable_audio_tools and AudioAnalysisTool:
                try:
                    tools['audio_analysis'] = AudioAnalysisTool(device=self.device)
                except Exception as e:
                    self.logger.warning(f"Failed to initialize AudioAnalysis: {e}")
        else:
            self.logger.info("Running in lightweight mode - only music theory tools available")
            
        return tools
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize specialized agents for different tasks."""
        agents = {}
        
        # Language detection and translation agent
        try:
            translation_agent = CodeAgent(
                model=InferenceClientModel("deepseek-ai/DeepSeek-R1", provider="together", max_tokens=4096),
                tools=[],
                additional_authorized_imports=["json"],
                name="TranslationAgent",
                description="Specialized agent for language detection and translation tasks"
            )
            agents['translator'] = translation_agent
        except Exception as e:
            self.logger.warning(f"Failed to initialize translation agent: {e}")
            
        # Music analysis and summarization agent
        try:
            analysis_agent = CodeAgent(
                model=InferenceClientModel("deepseek-ai/DeepSeek-R1", provider="together", max_tokens=4096),
                tools=[],
                additional_authorized_imports=["json", "numpy"],
                name="AnalysisAgent", 
                description="Specialized agent for music analysis, summarization and content combination"
            )
            agents['analyzer'] = analysis_agent
        except Exception as e:
            self.logger.warning(f"Failed to initialize analysis agent: {e}")
            
        return agents
    
    def run(self, query: str, **kwargs) -> str:
        """
        Process a user query using the multiagent system.
        
        This method implements a sophisticated workflow:
        1. Detect query language and translate to English if needed
        2. Route query to appropriate tools/agents via CodeAgent
        3. Combine and summarize results from multiple tools
        4. Translate final response back to original language
        
        Args:
            query: The user's query in any language
            **kwargs: Additional arguments
            
        Returns:
            The agent's response in the original query language
        """
        try:
            # Step 1: Detect language and prepare multilingual workflow
            original_language = self._detect_language(query)
            
            # Step 2: Translate query to English if needed
            english_query = self._translate_to_english(query, original_language)
            
            # Step 3: Process query through the main CodeAgent
            try:
                if self.manager_agent and self.use_advanced_routing:
                    agent_result = self.manager_agent.run(english_query, **kwargs)
                    agent_response = str(agent_result)
                else:
                    # Use simple tool routing (fallback mode)
                    agent_response = self._simple_tool_routing(english_query)
                    
                # Fallback if no response found
                if not agent_response.strip():
                    agent_response = self._simple_tool_routing(english_query)
                    
            except Exception as e:
                self.logger.warning(f"Error with CodeAgent: {e}")
                # Fallback to simple tool routing
                agent_response = self._simple_tool_routing(english_query)
            
            # Step 4: Post-process and enhance the response
            enhanced_response = self._enhance_response(agent_response, english_query)
            
            # Step 5: Translate back to original language if needed
            final_response = self._translate_from_english(enhanced_response, original_language)
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error in multiagent processing: {e}")
            return self._generate_error_response(str(e), query)
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of input text."""
        try:
            # Simple heuristic - if all ASCII, assume English
            if text.isascii():
                return "en"
            else:
                # For now, default to English. In production, use langdetect or similar
                return "en"
        except Exception:
            return "en"
    
    def _translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English if needed."""
        if source_lang == "en":
            return text
            
        # For now, return original text. In production, implement translation
        # using the translation agent or external API
        return text
    
    def _translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate response back to target language if needed."""
        if target_lang == "en":
            return text
            
        # For now, return original text. In production, implement translation
        return text
    
    def _enhance_response(self, response: str, query: str) -> str:
        """Enhance the response with additional context and summaries."""
        try:
            # Add metadata about generated content
            enhanced = response
            
            # Check if audio files were generated
            audio_files = self._find_generated_audio_files()
            if audio_files:
                enhanced += "\n\n🎵 **Generated Audio Files:**\n"
                for file_path in audio_files:
                    enhanced += f"- {file_path}\n"
                enhanced += "\nThese audio files have been created based on your request and are ready for playback."
            
            # Add helpful context based on query type
            if any(keyword in query.lower() for keyword in ['theory', 'explain', 'what is']):
                enhanced += "\n\n💡 **Additional Resources:** For more advanced music theory concepts, feel free to ask follow-up questions!"
            
            if any(keyword in query.lower() for keyword in ['generate', 'create', 'compose']):
                enhanced += "\n\n🎼 **Next Steps:** You can refine the generated content by providing more specific requirements or asking for variations."
                
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Error enhancing response: {e}")
            return response
    
    def _find_generated_audio_files(self) -> List[str]:
        """Find recently generated audio files."""
        try:
            audio_dir = "/tmp/music_agent_audio"
            if os.path.exists(audio_dir):
                files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
                return [os.path.join(audio_dir, f) for f in files]
            return []
        except Exception:
            return []
    
    def _generate_error_response(self, error: str, original_query: str) -> str:
        """Generate a helpful error response."""
        return f"""I encountered an issue while processing your request: "{original_query}"

Error: {error}

🔧 **Troubleshooting Tips:**
- Ensure all dependencies are installed
- Try rephrasing your query
- Check if the requested audio/music generation tools are available

I can still help with basic music theory questions. Would you like to try a different approach?"""
    
    def _simple_tool_routing(self, query: str) -> str:
        """Simple fallback tool routing when CodeAgent fails."""
        try:
            # Simple keyword-based routing (fallback behavior)
            query_lower = query.lower()
            
            if any(keyword in query_lower for keyword in ['scale', 'chord', 'theory', 'interval', 'key']):
                if 'music_theory' in self.tools_registry:
                    return self._execute_tool('music_theory', query)
                    
            elif any(keyword in query_lower for keyword in ['generate', 'create', 'compose']) and 'nota_gen' in self.tools_registry:
                return self._execute_tool('nota_gen', query)
                
            elif any(keyword in query_lower for keyword in ['audio', 'sound', 'wav', 'mp3']) and 'stable_audio' in self.tools_registry:
                return self._execute_tool('stable_audio', query)
                
            elif any(keyword in query_lower for keyword in ['analyze', 'understand', 'explain']) and 'chat_musician' in self.tools_registry:
                return self._execute_tool('chat_musician', query)
                
            # Fallback to music theory or basic response
            if 'music_theory' in self.tools_registry:
                return self._execute_tool('music_theory', query)
            else:
                return self._generate_basic_response(query)
                
        except Exception as e:
            self.logger.error(f"Error in simple routing: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def _execute_tool(self, tool_name: str, query: str) -> str:
        """Execute a specific tool with the given query."""
        tool = self.tools_registry.get(tool_name)
        if tool:
            # Try different execution methods in order of preference
            if hasattr(tool, 'execute'):
                return tool.execute(query)
            elif hasattr(tool, 'run'):
                return tool.run(query)
            elif hasattr(tool, 'forward'):
                # Handle models that use forward() method (like ChatMusician, StableAudio)
                return tool.forward(query)
            elif hasattr(tool, '__call__'):
                # Handle callable tools
                return tool(query)
            else:
                return f"Tool {tool_name} is available but doesn't have a standard interface."
        return f"Tool {tool_name} is not available."
    
    def _generate_basic_response(self, query: str) -> str:
        """Generate a basic response when no tools are available."""
        return f"""I understand you're asking about: "{query}"

I'm currently running in minimal mode with limited capabilities. Here's what I can help with:

🎵 **Music Theory**: I can explain scales, chords, intervals, key signatures, and music theory concepts.

🎼 **General Music Knowledge**: I can provide information about musical styles, composers, and basic composition techniques.

For advanced features like audio generation, music analysis, or symbolic composition, please ensure all dependencies are properly installed.

Would you like me to help with any music theory concepts or general music questions?"""
    
    def list_capabilities(self) -> List[str]:
        """List the available capabilities of this agent."""
        capabilities = []
        
        if 'music_theory' in self.tools_registry:
            capabilities.append("Music theory assistance")
            
        if 'chat_musician' in self.tools_registry:
            capabilities.append("Music understanding and analysis")
            
        if 'nota_gen' in self.tools_registry:
            capabilities.append("Symbolic music generation (ABC notation)")
            
        if 'stable_audio' in self.tools_registry:
            capabilities.append("Audio generation")
            
        if 'audio_analysis' in self.tools_registry:
            capabilities.append("Audio analysis and understanding")
            
        if not capabilities:
            capabilities.append("Basic music theory and general music knowledge")
            
        return capabilities
    
    def get_tool_info(self, tool_name: str) -> Optional[str]:
        """Get information about a specific tool."""
        if tool_name in self.tools_registry:
            tool = self.tools_registry[tool_name]
            return getattr(tool, 'description', f"Tool: {tool_name}")
        return None
    
    def chat(self, message: str, **kwargs) -> str:
        """Alias for run() method for chat-like interaction."""
        return self.run(message, **kwargs)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools_registry.keys())
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all available tools."""
        descriptions = {}
        for tool_name, tool in self.tools_registry.items():
            descriptions[tool_name] = getattr(tool, 'description', f'Tool: {tool_name}')
        return descriptions
    
    def create_gradio_interface(self):
        """Create a Gradio interface for long-term user interaction."""
        try:
            import gradio as gr
            
            def chat_interface(message, history):
                """Gradio chat interface function."""
                try:
                    response = self.run(message)
                    
                    # Check for generated audio files and include them
                    audio_files = self._find_generated_audio_files()
                    if audio_files:
                        # Return the latest audio file for playback
                        latest_audio = max(audio_files, key=os.path.getctime) if audio_files else None
                        return response, latest_audio
                    
                    return response, None
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    return error_msg, None
            
            # Create the interface
            with gr.Blocks(title="AI Music Agent") as interface:
                gr.Markdown("""
                # 🎵 AI Music Agent
                
                Your intelligent assistant for music theory, composition, and audio generation!
                
                **Capabilities:**
                - 🎼 Music theory explanations
                - 🤖 Advanced music understanding (ChatMusician)  
                - 🎵 AI audio generation (Stable Audio)
                - 🔍 Web search for music information
                - 🌍 Multilingual support
                """)
                
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            height=400,
                            label="Music Agent Chat",
                            show_copy_button=True
                        )
                        
                        msg = gr.Textbox(
                            label="Your Message",
                            placeholder="Ask me about music theory, request audio generation, or any music-related question...",
                            lines=2
                        )
                        
                        with gr.Row():
                            submit_btn = gr.Button("Send", variant="primary")
                            clear_btn = gr.Button("Clear", variant="secondary")
                    
                    with gr.Column(scale=1):
                        audio_output = gr.Audio(
                            label="Generated Audio",
                            type="filepath",
                            interactive=False
                        )
                        
                        gr.Markdown("""
                        ### Available Tools:
                        - Music Theory Tool
                        - ChatMusician (AI Music Understanding)
                        - Stable Audio (AI Audio Generation)
                        - Web Search
                        - Translation Support
                        """)
                
                # Handle message submission
                def submit_message(message, history):
                    if not message.strip():
                        return history, "", None
                        
                    # Add user message to history
                    history.append([message, ""])
                    
                    # Get agent response
                    response, audio = chat_interface(message, history)
                    
                    # Update history with response
                    history[-1][1] = response
                    
                    return history, "", audio
                
                submit_btn.click(
                    submit_message,
                    inputs=[msg, chatbot],
                    outputs=[chatbot, msg, audio_output]
                )
                
                msg.submit(
                    submit_message,
                    inputs=[msg, chatbot],
                    outputs=[chatbot, msg, audio_output]
                )
                
                clear_btn.click(
                    lambda: ([], None),
                    outputs=[chatbot, audio_output]
                )
                
                # Add examples
                gr.Examples(
                    examples=[
                        "What are the notes in a C major scale?",
                        "Explain the circle of fifths",
                        "Generate a happy piano melody",
                        "Create ambient soundscape audio",
                        "What is a diminished chord?",
                        "Compose a simple melody in MIDI"
                    ],
                    inputs=msg
                )
            
            return interface
            
        except ImportError:
            self.logger.warning("Gradio not available. Install with: pip install gradio")
            return None
        except Exception as e:
            self.logger.error(f"Error creating Gradio interface: {e}")
            return None
    
    def launch_gradio(self, **kwargs):
        """Launch the Gradio interface."""
        interface = self.create_gradio_interface()
        if interface:
            interface.launch(**kwargs)
        else:
            print("Gradio interface not available. Please install gradio: pip install gradio")

