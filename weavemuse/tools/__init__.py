"""
Music Agent Tools - Collection of specialized music AI tools.

All tools use lazy loading to improve startup performance.
Import tools directly when needed:
    from weavemuse.tools.notagen_tool import NotaGenTool
    from weavemuse.tools.chat_musician_tool import ChatMusicianTool
"""

def __getattr__(name):
    """Lazy import of all tools."""
    if name == "MusicTheoryTool":
        from .music_theory_tool import MusicTheoryTool
        return MusicTheoryTool
    elif name == "ChatMusicianTool":
        try:
            from .chat_musician_tool import ChatMusicianTool
            return ChatMusicianTool
        except ImportError:
            return None
    elif name == "NotaGenTool":
        try:
            from .notagen_tool import NotaGenTool
            return NotaGenTool
        except ImportError:
            return None
    elif name == "StableAudioTool":
        try:
            from .stable_audio_tool import StableAudioTool
            return StableAudioTool
        except ImportError:
            return None
    elif name == "AudioAnalysisTool":
        try:
            from .audio_analysis_tool import AudioAnalysisTool
            return AudioAnalysisTool
        except ImportError:
            return None
    elif name == "VerovioTool":
        try:
            from .verovio_tool import VerovioTool
            return VerovioTool
        except ImportError:
            return None
    elif name == "AudioFlamingoTool":
        try:
            from .audio_flamingo_tool import AudioFlamingoTool
            return AudioFlamingoTool
        except ImportError:
            return None
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")