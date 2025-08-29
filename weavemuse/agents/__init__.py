"""
Agents module for the Music Agent Framework.
"""

def __getattr__(name):
    """Lazy import of agents."""
    if name == "MusicAgent":
        from .music_agent import MusicAgent
        return MusicAgent
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
