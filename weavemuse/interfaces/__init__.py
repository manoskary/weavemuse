"""
Interfaces module for the Music Agent Framework.
"""

def __getattr__(name):
    """Lazy import of interfaces."""
    if name == "WeaveMuseInterface":
        from .gradio_interface import WeaveMuseInterface
        return WeaveMuseInterface
    elif name == "WeaveMuseTerminal":
        from .terminal_interface import WeaveMuseTerminal
        return WeaveMuseTerminal
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
