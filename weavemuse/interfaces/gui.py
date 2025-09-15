import os, sys
from typing import List
from smolagents import Tool, CodeAgent


class WeaveMuseGUI:
    """Terminal-based interface for WeaveMuse."""
    
    def __init__(self):
        """Initialize the terminal interface with lazy loading."""
        self.conversation_history = []
        self.uploaded_files = []
        self.agent = None
        self.tools: List[Tool] = []
        self.vram_manager = None
        self.tools_initialized = False
        self.model_choice = None        
        print("⚡ WeaveMuse Terminal Interface - Fast Start Mode")
    
    def ask_model_choice(self) -> str:
        """Ask user to choose between local or HuggingFace model."""
        print("\n🤖 Choose your AI model:")
        print("1. Only Local Models (Requires more resources and loading time)")
        print("2. HuggingFace cloud-based agent (some local tools - faster startup)")
        print("3. All Remote (All models and Tools are remote - no resources needed)")
        
        while True:
            try:
                choice = input("\nEnter your choice (1/2/3): ").strip()
                if choice in ['1', '2', '3']:
                    return choice
                else:
                    print("❌ Please enter 1, 2, or 3")
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                sys.exit(0)
    
    def set_up_agents(self):                
        model_choice = self.ask_model_choice()

        # Import GPU detection utilities
        from weavemuse.utils.gpu_utils import detect_gpu_capabilities, print_gpu_summary, check_force_cpu_mode
        from weavemuse.agents.agents_as_tools import get_weavemuse_agents_and_tools

        # Detect GPU capabilities and get optimal model configuration
        force_cpu = check_force_cpu_mode()
        gpu_info = detect_gpu_capabilities(force_cpu=force_cpu)
        print_gpu_summary(gpu_info)

        if model_choice == '1':
            tool_mode = "hybrid"  # Some tools local, some remote
            model, gpu_info = self.setup_local_model(gpu_info=gpu_info)
        else:
            from smolagents import InferenceClientModel
            model = InferenceClientModel()
            if model_choice == '3':
                tool_mode = "remote"
            else:
                tool_mode = "hybrid"  # Some tools local, some remote

        print("⚡ Setting up WeaveMuse agents and tools...")         
        weavemuse_agents, weavemuse_tools = get_weavemuse_agents_and_tools(model=model, device_map=gpu_info.device_map, notagen_output_dir="/tmp/notagen_output", stable_audio_output_dir="/tmp/stable_audio")

        # Main manager agent
        manager_agent = CodeAgent(
            tools=weavemuse_tools, 
            model=model,
            managed_agents=weavemuse_agents,
            name="music_manager_agent",
            description=(
                "Manages music-related tasks, including web searches, advanced music analysis, "
                "symbolic music generation, audio music generation, and audio analysis using Audio Flamingo. "
                "When users upload audio files for analysis, forward the EXACT file path to the "
                "audio_flamingo_agent without checking if files exist first. "
                "If a task is not related to music, ask the user to provide a music-related query."
            ),
            add_base_tools=True,
            stream_outputs=True,
            max_steps=3,      
            additional_authorized_imports=[]
        )
        return manager_agent

    def setup_local_model(self, gpu_info):                
        from weavemuse.agents.models import TransformersModel        
        from weavemuse.utils.gpu_utils import get_quantization_config


        # Extract configuration from GPU info
        device_mode = gpu_info.device_map
         
        # The cache path is USER_HOME/.cache/huggingface/hub/
        CACHE_PATH = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
        CACHE_PATH = os.path.join(CACHE_PATH, "hub")

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
        return model, gpu_info

    def set_up_interface(self):
        """Set up the WeaveMuse interface."""
        from weavemuse.interfaces.gradio_interface import WeaveMuseInterface

        manager_agent = self.set_up_agents()
        
        # Launch the WeaveMuse interface
        demo = WeaveMuseInterface(
            agent=manager_agent,
            reset_agent_memory=False,
            file_upload_folder="/tmp/music_agent_uploads",
        )
        return demo

    def launch_interface(self, share=False):
        """Launch the WeaveMuse interface."""
        demo = self.set_up_interface()
        demo.launch(share=share)