import os
import torch
from smolagents import (
    load_tool,
    CodeAgent,
    # InferenceClientModel,    
    WebSearchTool,
    # LiteLLMModel,    
    ChatMessage,
    MessageRole,    
    # TransformersModel,        
)
from smolagents.tools import Tool
from .weavemuse.tools.stable_audio_tool import StableAudioTool
from .weavemuse.tools.audio_analysis_tool import AudioAnalysisTool
from .weavemuse.tools.notagen_tool import NotaGenTool
from .weavemuse.interfaces.gradio_interface import WeaveMuseInterface
from .weavemuse.agents.models import TransformersModel
from transformers import BitsAndBytesConfig

# The cache path is USER_HOME/.cache/huggingface/hub/
CACHE_PATH = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")




# model = InferenceClientModel()
# model = LiteLLMModel(
#     model_id="ollama_chat/qwen2.5-coder:14b-instruct-q4_K_M",
#     api_base="http://localhost:11434",
#     num_ctx=8192,   
# )
gguf_file = None
# model_id = "HuggingFaceTB/SmolLM3-3B"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
# model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
model_id = "Qwen/Qwen2.5-Coder-14B-Instruct"
# model_id = "openai/gpt-oss-20b"
# model_id = "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF"
# ggfu_file = "qwen2.5-coder-32b-instruct-q2_k.gguf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

if gguf_file is not None:
    # check if gguf_file exists exists in the cache otherwise download
    gguf_file_path = os.path.join(CACHE_PATH, gguf_file)
    if not os.path.exists(gguf_file_path):
        download_url = f"https://huggingface.co/{model_id}/resolve/main/{gguf_file}"
        print(f"Attempting to download GGUF file from {download_url} to {gguf_file_path}")
        import requests
        response = requests.get(download_url)
        if response.status_code == 200:
            with open(gguf_file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded GGUF file to {gguf_file_path}")
        else:
            print(f"Failed to download GGUF file from {download_url}. Status code: {response.status_code}")
            gguf_file_path = None
else:
    gguf_file_path = None


model = TransformersModel( 
    model_id=model_id, 
    trust_remote_code=True, 
    device_map="auto",  # Start with CPU to save VRAM
    torch_dtype="auto",
    gguf_file=gguf_file if gguf_file else None,
    low_cpu_mem_usage=True,
    offload_buffers=True,  
    quantization_config=quantization_config,      
)

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

# Create NotaGen tool for symbolic music generation
notagen_tool = NotaGenTool(device="auto", output_dir="/tmp/notagen_output")

# Create symbolic music generation agent
symbolic_music_agent = CodeAgent(
    tools=[notagen_tool],
    model=model,
    name="symbolic_music_agent",
    description="Generates symbolic music in ABC notation format with full conversion capabilities. Can create compositions based on musical periods, composers, and instrumentation. Returns a PDF of the score."
)

# audio_generation_tool = StableAudioTool(device="cuda:0")
# audio_generation_agent = CodeAgent(
#     tools=[audio_generation_tool],
#     model=model,
#     name="audio_generation_agent",
#     description="Generates audio from text descriptions. Use the 'prompt' argument to specify what you want to hear. Only generates once and then returns the audio"
#     )

# audio_analysis_tool = AudioAnalysisTool(
#     device="cuda:0", 
#     model="mradermacher/Qwen2-Audio-7B-i1-GGUF"
#     )vice="cuda:0")
# audio_analysis_agent = CodeAgent(
#     tools=[audio_analysis_tool],
#     model=model,
#     name="audio_analysis_agent",
#     description="Analyzes audio files using AI to provide detailed descriptions, musical analysis, and insights. Use 'audio_file' argument for the file path and 'analysis_type' for the type of analysis."
# )

# image_generation_tool = load_tool("m-ric/text-to-image", trust_remote_code=True, device="cuda:0")
# image_generation_agent = CodeAgent(
#     tools=[image_generation_tool],
#     model=model,
#     name="image_generation_agent",
#     description="Generates images from text descriptions. Use the 'prompt' argument to specify what you want to see."
# )

manager_agent = CodeAgent(
    tools=[conversational_tool], 
    model=model, 
    managed_agents=[
        web_agent, 
        symbolic_music_agent,
        # audio_generation_agent, 
        # audio_analysis_agent, 
        # image_generation_agent,        
        ],
    name="music_manager_agent",
    description="Manages music-related tasks, including web searches, symbolic music generation, audio generation, audio analysis when needed. If a task is not related to music, it will either be handled by the conversational tool or ask the user to privide a task related query.",
    add_base_tools=True,
    # planning_interval=3,
    stream_outputs=True,    
)

WeaveMuseInterface(
    agent=manager_agent,
    reset_agent_memory=False,
    file_upload_folder="/tmp/music_agent_uploads",
         ).launch()