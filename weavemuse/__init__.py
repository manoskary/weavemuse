"""
Music Agent Framework

A comprehensive music agent framework built on smolagents, integrating 
state-of-the-art music AI models for understanding, generation, and interaction.
"""

__version__ = "0.1.0"
__author__ = "Music Agent Team"
__email__ = "team@musicagent.ai"

from .agents.music_agent import MusicAgent
from .tools import (
    ChatMusicianTool,
    NotaGenTool, 
    StableAudioTool,
    AudioAnalysisTool,
    VerovioTool,
)

__all__ = [
    "MusicAgent",
    "ChatMusicianTool",
    "NotaGenTool",
    "StableAudioTool", 
    "AudioAnalysisTool",
    "VerovioTool",
]
