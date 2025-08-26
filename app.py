import os
import torch

# Check VRAM availability and force CPU mode if needed
def check_vram_and_set_device():
    """Check VRAM availability and set appropriate device."""
    try:
        if torch.cuda.is_available():
            # Try to allocate a small tensor to test VRAM
            test_tensor = torch.zeros(100, device='cuda')
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ CUDA available and working")
            return "auto"
        else:
            print("⚠️  CUDA not available, using CPU")
            return "cpu"
    except torch.OutOfMemoryError:
        print("❌ CUDA out of memory, forcing CPU mode")
        # Force CPU mode due to VRAM constraints
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["USE_CUDA"] = "false"
        return "cpu"
    except Exception as e:
        print(f"⚠️  CUDA error ({e}), using CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"

# Check and set device mode
device_mode = check_vram_and_set_device()
print(f"🖥️  Device mode: {device_mode}")

from smolagents import (
    load_tool,
    CodeAgent,
    WebSearchTool,
    ChatMessage,
    MessageRole,    
)
from smolagents.tools import Tool
from weavemuse.tools.stable_audio_tool import StableAudioTool
from weavemuse.tools.audio_analysis_tool import AudioAnalysisTool
from weavemuse.tools.notagen_tool import NotaGenTool
from weavemuse.tools.chat_musician_tool import ChatMusicianTool
from weavemuse.interfaces.gradio_interface import WeaveMuseInterface
from weavemuse.agents.models import TransformersModel
from transformers import BitsAndBytesConfig

# The cache path is USER_HOME/.cache/huggingface/hub/
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")

# Model configuration - using a smaller, more manageable model
gguf_file = None
model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"  # Good balance of capability and size

# Optimized quantization for VRAM management (only use when device mode is auto/cuda)
quantization_config = None
if device_mode == "auto" and torch.cuda.is_available() and torch.cuda.device_count() > 0:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    print("✅ Using 4-bit quantization for CUDA")
else:
    print("⚠️  Quantization disabled (CPU mode)")

# Determine device map based on device mode
device_map = "cpu" if device_mode == "cpu" else "auto"

# Create optimized main model (adaptive device mode)
model = TransformersModel( 
    model_id=model_id, 
    trust_remote_code=True, 
    device_map=device_map,
    torch_dtype="auto",
    gguf_file=gguf_file if gguf_file else None,
    low_cpu_mem_usage=True,
    offload_buffers=True,  
    quantization_config=quantization_config,  # None for CPU, quantized for CUDA      
)

# Create web search agent
web_agent = CodeAgent(
    tools=[WebSearchTool()],
    model=model,
    name="web_search_agent",
    description="Runs web searches for you. Give it your query as an argument."
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

# Create ChatMusician tool for advanced music understanding
chat_musician_tool = ChatMusicianTool(device=device_mode)

# Create NotaGen tool for symbolic music generation
notagen_tool = NotaGenTool(device=device_mode, output_dir="/tmp/notagen_output")

# Create Stable Audio tool for audio generation (temporarily disabled to focus on NotaGen and ChatMusician)
# stable_audio_tool = StableAudioTool(device="auto")

# Create Audio Analysis tool (commented out to save VRAM)  
# audio_analysis_tool = AudioAnalysisTool(device="auto")

# Create ChatMusician agent for advanced music understanding
chat_musician_agent = CodeAgent(
    tools=[chat_musician_tool],
    model=model,
    name="chat_musician_agent", 
    description="Advanced music understanding and analysis using AI. Can analyze musical structures, provide composition advice, and help with complex music theory questions."
)

# Create symbolic music generation agent 
symbolic_music_agent = CodeAgent(
    tools=[notagen_tool],
    model=model,
    name="symbolic_music_agent",
    description="Generates/Composes symbolic music in ABC notation format with full conversion capabilities. Can create compositions based on musical periods, composers, and instrumentation. Returns a PDF, XML, MIDI, and MP3 of the score."
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
        # stable_audio_agent,
        # audio_analysis_agent,     
    ],
    name="music_manager_agent",
    description="Manages music-related tasks, including web searches, advanced music analysis, and symbolic music generation. If a task is not related to music, it will ask the user to provide a music-related query.",
    add_base_tools=True,
    stream_outputs=True,    
)

# Launch the WeaveMuse interface
WeaveMuseInterface(
    agent=manager_agent,
    reset_agent_memory=False,
    file_upload_folder="/tmp/music_agent_uploads",
).launch()