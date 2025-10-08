from smolagents import (
        InferenceClientModel,
        CodeAgent,
        WebSearchTool,         
    )
from weavemuse.tools.stable_audio_tool import StableAudioTool, RemoteStableAudioTool
from weavemuse.tools.audio_analysis_tool import AudioAnalysisTool
from weavemuse.tools.notagen_tool import NotaGenTool, RemoteNotaGenTool
from weavemuse.tools.chat_musician_tool import ChatMusicianTool
from weavemuse.tools.audio_flamingo_tool import AudioFlamingoTool
import warnings


def create_web_agent(model):
    web_agent = CodeAgent(
        tools=[WebSearchTool()],
        model=model,
        name="web_search_agent",
        description="Runs web searches for you. Give it your query as an argument.",
        additional_authorized_imports=["os", "pathlib", "tempfile"]
    )    
    return web_agent


def create_symbolic_music_agent(model, device_map="auto", output_dir="/tmp/notagen_output", remote_only=False):
    # Create NotaGen tool for symbolic music generation
    if remote_only:
        notagen_tool = RemoteNotaGenTool()
    else:
        notagen_tool = NotaGenTool(device=device_map, output_dir=output_dir)

    symbolic_music_agent = CodeAgent(
        tools=[notagen_tool],
        model=model,
        name="symbolic_music_agent",
        description="Generates/Composes symbolic music in ABC notation format with full conversion capabilities. Can create compositions based on musical periods, composers, and instrumentation. Returns a PDF, XML, MIDI, and MP3 of the score. Example usage of this tool: Compose a piano piece in the style of Chopin.",
        max_steps=1
    )  
    return symbolic_music_agent


def create_audio_analysis_agent(model, device_map="auto", remote_only=False):
    # Create Audio Flamingo tool for advanced audio analysis via Gradio Client
    audio_flamingo_tool = AudioFlamingoTool()
    if not remote_only:
        # Create Audio Analysis tool (commented out to save VRAM)      
        audio_analysis_tool = AudioAnalysisTool(device=device_map)
        tl = [audio_flamingo_tool, audio_analysis_tool]
    else:
        tl = [audio_flamingo_tool]
    # Create Audio Flamingo agent for advanced audio analysis
    audio_analysis_agent = CodeAgent(
        tools=tl,
        model=model,
        name="audio_analysis_agent",
        description=(
            "Analyzes audio files using NVIDIA's Audio Flamingo model or other audio analysis tools. "
            "Can answer questions about audio content, describe musical elements, "
            "identify sounds, and provide detailed acoustic analysis. "
            "IMPORTANT: When users upload audio files, use the exact file path provided"
            "without checking if the file exists first. The tool will handle file validation internally."
            "Use first the Audio Flamingo tool, and if it fails, fall back to the Audio Analysis tool."
        ),
        additional_authorized_imports=["gradio_client", "os", "pathlib", "tempfile", "shutil", "posixpath"],
        max_steps=2
    )
    return audio_analysis_agent


def create_audio_generation_agent(model, device_map="auto", output_dir="/tmp/stable_audio", remote_only=False):
    # Create Audio Generation and Analysis agent (temporarily disabled to focus on NotaGen and ChatMusician)
    if remote_only:
        stable_audio_tool = RemoteStableAudioTool()
    else:
        stable_audio_tool = StableAudioTool(device=device_map, output_dir=output_dir)
    audio_generation_agent = CodeAgent(
        tools=[stable_audio_tool],
        model=model,
        name="audio_generation_agent",
        description="Generates audio from text descriptions. Use the 'prompt' argument to specify what you want to hear. Generates up to 47 seconds of high-quality audio.",
    )   
    return audio_generation_agent


def get_weavemuse_agents_and_tools(model=None, device_map="auto", notagen_output_dir="/tmp/notagen_output", stable_audio_output_dir="/tmp/stable_audio", tool_mode="hybrid"):
    """
    Returns all WeaveMuse agents and tools as a list for easy access and management.

    Args:
        model: The language model to be used by the agents.
        device_map (str): Device mapping for model deployment (default is "auto").
        notagen_output_dir (str): Output directory for NotaGen tool (default is "/tmp/notagen_output").
        stable_audio_output_dir (str): Output directory for Stable Audio tool (default
    
    """
    # If model is not provided, load a default InferenceClient model
    if model is None:
        # Load the default InferenceClient model, but produce warning if not specified
        warnings.warn("No model specified, using default InferenceClientModel.")
        model = InferenceClientModel(
            model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            provider="nebius"
        )
    remote_only = (tool_mode == "remote")       
    chat_musician_tool = ChatMusicianTool(device=device_map)
    symbolic_music_agent = create_symbolic_music_agent(model, device_map=device_map, output_dir=notagen_output_dir, remote_only=remote_only)
    audio_analysis_agent = create_audio_analysis_agent(model, device_map=device_map, remote_only=remote_only)
    audio_generation_agent = create_audio_generation_agent(model, device_map=device_map, output_dir=stable_audio_output_dir, remote_only=remote_only)
    web_agent = create_web_agent(model)
    tools = [] if remote_only else [chat_musician_tool]
    return [            
        symbolic_music_agent,
        audio_analysis_agent,
        audio_generation_agent,
        web_agent
    ], tools