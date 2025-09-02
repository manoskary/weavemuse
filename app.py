import os
import torch
import gradio as gr


if gr.NO_RELOAD:
    # Import GPU detection utilities
    from weavemuse.utils.gpu_utils import detect_gpu_capabilities, get_quantization_config, print_gpu_summary, check_force_cpu_mode
    
    # Detect GPU capabilities and get optimal model configuration
    force_cpu = check_force_cpu_mode()
    gpu_info = detect_gpu_capabilities(force_cpu=force_cpu)
    print_gpu_summary(gpu_info)
    
    # Extract configuration from GPU info
    device_mode = gpu_info.device_map

    from smolagents import (
        load_tool,
        CodeAgent,
        WebSearchTool,
        ChatMessage,
        MessageRole,    
    )
    from smolagents.tools import Tool    
    from weavemuse.interfaces.gradio_interface import WeaveMuseInterface
    from weavemuse.agents.models import TransformersModel
    from transformers import BitsAndBytesConfig


    


    # The cache path is USER_HOME/.cache/huggingface/hub/
    CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

    # Model configuration based on GPU detection
    gguf_file = None
    model_id = gpu_info.recommended_model_id  # Automatically selected based on VRAM tier
    
    # Get quantization configuration based on GPU capabilities
    quantization_config = get_quantization_config(gpu_info)
    
    if quantization_config:
        print("✅ Using 4-bit quantization for optimal VRAM usage")
    else:
        print("⚠️  Quantization disabled (CPU mode or not available)")

    # Create optimized main model with automatic configuration
    model = TransformersModel( 
        model_id=model_id, 
        trust_remote_code=True, 
        device_map=gpu_info.device_map,
        torch_dtype="auto",
        gguf_file=gguf_file if gguf_file else None,
        low_cpu_mem_usage=True,
        offload_buffers=True,  
        quantization_config=quantization_config,  # Automatically configured based on GPU tier
    )

    # Create web search agent
    web_agent = CodeAgent(
        tools=[WebSearchTool()],
        model=model,
        name="web_search_agent",
        description="Runs web searches for you. Give it your query as an argument.",
        additional_authorized_imports=["os", "pathlib", "tempfile"]
    )

    # Create a conversational tool when the users queries are not about music. The tool politely replies and prompts the user to ask about music-related topics.

    class ConversationalTool(Tool):
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
        
        def forward(self, query):        
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

    conversational_tool = ConversationalTool()


from weavemuse.tools.stable_audio_tool import StableAudioTool
from weavemuse.tools.audio_analysis_tool import AudioAnalysisTool
from weavemuse.tools.notagen_tool import NotaGenTool
from weavemuse.tools.chat_musician_tool import ChatMusicianTool
from weavemuse.tools.audio_flamingo_tool import AudioFlamingoTool


# Create ChatMusician tool for advanced music understanding
chat_musician_tool = ChatMusicianTool(device=gpu_info.device_map)

# Create NotaGen tool for symbolic music generation
notagen_tool = NotaGenTool(device=gpu_info.device_map, output_dir="/tmp/notagen_output")

# Create Audio Flamingo tool for advanced audio analysis via Gradio Client
audio_flamingo_tool = AudioFlamingoTool()

# Create Stable Audio tool for audio generation (temporarily disabled to focus on NotaGen and ChatMusician)
# stable_audio_tool = StableAudioTool(device="auto")

# Create Audio Analysis tool (commented out to save VRAM)  
# audio_analysis_tool = AudioAnalysisTool(device="auto")

# Create ChatMusician agent for advanced music understanding
chat_musician_agent = CodeAgent(
    tools=[chat_musician_tool],
    model=model,
    name="chat_musician_agent", 
    description="Advanced music understanding and analysis using AI. Can analyze musical structures, provide composition advice, and help with complex music theory questions.",
    additional_authorized_imports=["os", "pathlib", "tempfile"]
)

# Create symbolic music generation agent 
symbolic_music_agent = CodeAgent(
    tools=[notagen_tool],
    model=model,
    name="symbolic_music_agent",
    description="Generates/Composes symbolic music in ABC notation format with full conversion capabilities. Can create compositions based on musical periods, composers, and instrumentation. Returns a PDF, XML, MIDI, and MP3 of the score.",
    max_steps=1
)

# Create Audio Flamingo agent for advanced audio analysis
audio_flamingo_agent = CodeAgent(
    tools=[audio_flamingo_tool],
    model=model,
    name="audio_flamingo_agent",
    description=(
        "Analyzes audio files using NVIDIA's Audio Flamingo model. "
        "Can answer questions about audio content, describe musical elements, "
        "identify sounds, and provide detailed acoustic analysis. "
        "IMPORTANT: When users upload audio files, use the exact file path provided "
        "without checking if the file exists first. The tool will handle file validation internally."
    ),
    additional_authorized_imports=["gradio_client", "os", "pathlib", "tempfile", "shutil"]
)

# Audio generation and analysis agents (temporarily disabled to focus on NotaGen and ChatMusician)
# stable_audio_agent = CodeAgent(
#     tools=[stable_audio_tool],
#     model=model,
#     name="audio_generation_agent",
#     description="Generates audio from text descriptions. Use the 'prompt' argument to specify what you want to hear."
# )
# 
# audio_analysis_agent = CodeAgent(
#     tools=[audio_analysis_tool],
#     model=model,
#     name="audio_analysis_agent", 
#     description="Analyzes audio files using AI to provide detailed descriptions, musical analysis, and insights."
# )

# Main manager agent
manager_agent = CodeAgent(
    tools=[conversational_tool], 
    model=model, 
    managed_agents=[
        web_agent,
        chat_musician_agent, 
        symbolic_music_agent,
        audio_flamingo_agent,
    ],
    name="music_manager_agent",
    description=(
        "Manages music-related tasks, including web searches, advanced music analysis, "
        "symbolic music generation, and audio analysis using Audio Flamingo. "
        "When users upload audio files for analysis, forward the EXACT file path to the "
        "audio_flamingo_agent without checking if files exist first. "
        "If a task is not related to music, ask the user to provide a music-related query."
    ),
    add_base_tools=True,
    stream_outputs=True,
    max_steps=3,      
    additional_authorized_imports=[]
)

# Launch the WeaveMuse interface
demo = WeaveMuseInterface(
    agent=manager_agent,
    reset_agent_memory=False,
    file_upload_folder="/tmp/music_agent_uploads",
)

# Only launch if running directly, not when imported by gradio app.py
if __name__ == "__main__":
    demo.launch()