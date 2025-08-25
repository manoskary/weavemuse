from weavemuse.interfaces.gradio_interface import WeaveMuseInterface
from smolagents import CodeAgent
from smolagents.models import Model
from smolagents import ChatMessage, MessageRole
from typing import Any, List, Optional


class DummyLLMModel(Model):
    """A dummy model for testing purposes."""
    
    def __init__(self):
        super().__init__()
        self.name = "DummyModel"

    def __call__(
        self,
        messages: List[ChatMessage],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[Any] = None,
        **kwargs
    ) -> ChatMessage:
        """Generate a dummy response."""
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="This is a dummy response from the test model."
        )
    
    def generate(
        self,
        messages: List[ChatMessage],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[Any] = None,
        **kwargs
    ) -> ChatMessage:
        """Generate a dummy response."""
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="This is a dummy response from the test model."
        )


agent = CodeAgent(
    model=DummyLLMModel(),
    tools=[],
    additional_authorized_imports=["json"],
    name="DummyAgent",
    description="A dummy agent for testing purposes"
)


interface = WeaveMuseInterface(agent=agent)
interface.launch(share=True)