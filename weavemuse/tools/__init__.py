"""
Music Agent Tools - Collection of specialized music AI tools.
"""

# Import tools with graceful fallback for missing dependencies
from .music_theory_tool import MusicTheoryTool

# Try to import heavy tools, but don't fail if dependencies are missing
try:
    from .chat_musician_tool import ChatMusicianTool
except ImportError:
    ChatMusicianTool = None

try:
    from .notagen_tool import NotaGenTool
except ImportError:
    NotaGenTool = None

try:
    from .stable_audio_tool import StableAudioTool
except ImportError:
    StableAudioTool = None

try:
    from .audio_analysis_tool import AudioAnalysisTool
except ImportError:
    AudioAnalysisTool = None

try:
    from .verovio_tool import VerovioTool
except ImportError:
    VerovioTool = None

# Import utility classes for memory-efficient tools
try:
    from .utils import (
        GPUOnDemandTool,
        MemoryEfficientStableAudioTool,
        MemoryEfficientAudioAnalysisTool,
        ConversationalTool
    )
except ImportError:
    GPUOnDemandTool = None
    MemoryEfficientStableAudioTool = None
    MemoryEfficientAudioAnalysisTool = None
    ConversationalTool = None

# Export available tools
__all__ = [
    'MusicTheoryTool',
    'ChatMusicianTool',
    'NotaGenTool', 
    'StableAudioTool',
    'AudioAnalysisTool',
    'VerovioTool',
    'GPUOnDemandTool',
    'MemoryEfficientStableAudioTool',
    'MemoryEfficientAudioAnalysisTool',
    'ConversationalTool'
]