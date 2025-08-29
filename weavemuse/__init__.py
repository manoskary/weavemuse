"""
Music Agent Framework

A comprehensive music agent framework built on smolagents, integrating 
state-of-the-art music AI models for understanding, generation, and interaction.
"""

__version__ = "0.1.0"
__author__ = "Music Agent Team"
__email__ = "team@musicagent.ai"

# Lazy loading - only import when actually needed
# Use: from weavemuse import MusicAgent
# Or: from weavemuse.agents.music_agent import MusicAgent

def __getattr__(name):
    """Lazy import of heavy components."""
    if name == "MusicAgent":
        from .agents.music_agent import MusicAgent
        return MusicAgent
    elif name == "ChatMusicianTool":
        from .tools.chat_musician_tool import ChatMusicianTool
        return ChatMusicianTool
    elif name == "NotaGenTool":
        from .tools.notagen_tool import NotaGenTool
        return NotaGenTool
    elif name == "StableAudioTool":
        from .tools.stable_audio_tool import StableAudioTool
        return StableAudioTool
    elif name == "AudioAnalysisTool":
        from .tools.audio_analysis_tool import AudioAnalysisTool
        return AudioAnalysisTool
    elif name == "VerovioTool":
        from .tools.verovio_tool import VerovioTool
        return VerovioTool
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
