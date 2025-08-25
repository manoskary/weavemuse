"""
Configuration management for the Music Agent Framework.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MusicAgentConfig:
    """Configuration class for the Music Agent Framework."""
    
    # Model configurations
    chatmusician_model_id: str = "m-a-p/ChatMusician"
    notagen_model_path: str = "./models/notagen"
    stable_audio_model_id: str = "stabilityai/stable-audio-open-1.0"
    default_llm_model: str = "meta-llama/Llama-2-70b-chat-hf"
    
    # Device configuration
    device: str = "auto"
    torch_dtype: str = "float16"
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 7860
    debug: bool = False
    
    # Hugging Face configuration
    hf_token: Optional[str] = None
    
    # API keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Gradio configuration
    gradio_share: bool = False
    gradio_auth: bool = False
    gradio_auth_username: str = "admin"
    gradio_auth_password: str = "password"
    
    # Logging
    log_level: str = "INFO"
    
    # Tool-specific configurations
    tool_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        self._load_from_env()
        self._validate_config()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Model configurations
        self.chatmusician_model_id = os.getenv(
            "CHATMUSICIAN_MODEL_ID", self.chatmusician_model_id
        )
        self.notagen_model_path = os.getenv(
            "NOTAGEN_MODEL_PATH", self.notagen_model_path
        )
        self.stable_audio_model_id = os.getenv(
            "STABLE_AUDIO_MODEL_ID", self.stable_audio_model_id
        )
        self.default_llm_model = os.getenv(
            "DEFAULT_LLM_MODEL", self.default_llm_model
        )
        
        # Device configuration
        self.device = os.getenv("DEVICE", self.device)
        self.torch_dtype = os.getenv("TORCH_DTYPE", self.torch_dtype)
        
        # Server configuration
        self.host = os.getenv("HOST", self.host)
        self.port = int(os.getenv("PORT", str(self.port)))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # API keys
        self.hf_token = os.getenv("HF_TOKEN")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Gradio configuration
        self.gradio_share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
        self.gradio_auth = os.getenv("GRADIO_AUTH", "false").lower() == "true"
        self.gradio_auth_username = os.getenv(
            "GRADIO_AUTH_USERNAME", self.gradio_auth_username
        )
        self.gradio_auth_password = os.getenv(
            "GRADIO_AUTH_PASSWORD", self.gradio_auth_password
        )
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
    
    def _validate_config(self):
        """Validate the configuration."""
        # Create necessary directories
        model_dir = Path(self.notagen_model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set Hugging Face token if available
        if self.hf_token:
            os.environ["HF_TOKEN"] = self.hf_token
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """Get configuration for a specific tool."""
        return self.tool_configs.get(tool_name, {})
    
    def set_tool_config(self, tool_name: str, config: Dict[str, Any]):
        """Set configuration for a specific tool."""
        self.tool_configs[tool_name] = config
    
    @classmethod
    def from_file(cls, config_path: str) -> "MusicAgentConfig":
        """Load configuration from a file."""
        import json
        import yaml
        
        config_path = Path(config_path)
        
        if config_path.suffix == ".json":
            with open(config_path, "r") as f:
                config_data = json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(**config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
