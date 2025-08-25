from smolagents.models import Model, remove_stop_sequences
import warnings
from typing import Any
from collections.abc import Generator
from threading import Thread
import logging
from smolagents import (
    ChatMessage,
    TokenUsage,
    ChatMessageStreamDelta,
    MessageRole,
    Tool,    
)
from transformers.generation import StoppingCriteriaList

logger = logging.getLogger(__name__)


class TransformersModel(Model):
    """A class that uses Hugging Face's Transformers library for language model interaction.

    This model allows you to load and use Hugging Face's models locally using the Transformers library. It supports features like stop sequences and grammar customization.

    > [!TIP]
    > You must have `transformers` and `torch` installed on your machine. Please run `pip install smolagents[transformers]` if it's not the case.

    Parameters:
        model_id (`str`):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
            For example, `"Qwen/Qwen2.5-Coder-32B-Instruct"`.
        device_map (`str`, *optional*):
            The device_map to initialize your model with.
        torch_dtype (`str`, *optional*):
            The torch_dtype to initialize your model with.
        trust_remote_code (bool, default `False`):
            Some models on the Hub require running remote code: for this model, you would have to set this flag to True.
        kwargs (dict, *optional*):
            Any additional keyword arguments that you want to use in model.generate(), for instance `max_new_tokens` or `device`.
        **kwargs:
            Additional keyword arguments to pass to `model.generate()`, for instance `max_new_tokens` or `device`.
    Raises:
        ValueError:
            If the model name is not provided.

    Example:
    ```python
    >>> engine = TransformersModel(
    ...     model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    ...     device="cuda",
    ...     max_new_tokens=5000,
    ... )
    >>> messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str | None = None,
        device_map: str | None = None,
        torch_dtype: str | None = None,
        trust_remote_code: bool = False,        
        **kwargs,
    ):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,                
                AutoTokenizer,
                TextIteratorStreamer,
            )
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'transformers' extra to use 'TransformersModel': `pip install 'smolagents[transformers]'`"
            )

        if not model_id:
            warnings.warn(
                "The 'model_id' parameter will be required in version 2.0.0. "
                "Please update your code to pass this parameter to avoid future errors. "
                "For now, it defaults to 'HuggingFaceTB/SmolLM2-1.7B-Instruct'.",
                FutureWarning,
            )
            model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

        default_max_tokens = 4096
        max_new_tokens = kwargs.get("max_new_tokens", None)        
        if not max_new_tokens:
            kwargs["max_new_tokens"] = default_max_tokens
            logger.warning(
                f"`max_new_tokens` not provided, using this default value for `max_new_tokens`: {default_max_tokens}"
            )        
        
        if device_map is None:
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device_map}")
        self._is_vlm = False
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,                
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                gguf_file=kwargs.get("gguf_file", None),  # Use gguf_file if provided
                quantization_config=kwargs.pop("quantization_config", None),
                offload_buffers=kwargs.pop("offload_buffers", True),
                low_cpu_mem_usage=kwargs.pop("low_cpu_mem_usage", True)
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, gguf_file=kwargs.get("gguf_file", None))
            self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore
            # Pop used kwargs after loading

        except Exception as e:
            raise ValueError(f"Failed to load tokenizer and model for {model_id=}: {e}") from e
        super().__init__(flatten_messages_as_text=not self._is_vlm, model_id=model_id, **kwargs)

    def make_stopping_criteria(self, stop_sequences: list[str], tokenizer) -> "StoppingCriteriaList":
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnStrings(StoppingCriteria):
            def __init__(self, stop_strings: list[str], tokenizer):
                self.stop_strings = stop_strings
                self.tokenizer = tokenizer
                self.stream = ""

            def reset(self):
                self.stream = ""

            def __call__(self, input_ids, scores, **kwargs):
                generated = self.tokenizer.decode(input_ids[0][-1], skip_special_tokens=True)
                self.stream += generated
                if any([self.stream.endswith(stop_string) for stop_string in self.stop_strings]):
                    return True
                return False

        return StoppingCriteriaList([StopOnStrings(stop_sequences, tokenizer)])

    def _prepare_completion_args(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            **kwargs,
        )

        messages = completion_kwargs.pop("messages")
        stop_sequences = completion_kwargs.pop("stop", None)
        tools = completion_kwargs.pop("tools", None)

        max_new_tokens = (
            kwargs.get("max_new_tokens")
            or kwargs.get("max_tokens")
            or self.kwargs.get("max_new_tokens")
            or self.kwargs.get("max_tokens")
            or 1024
        )
        prompt_tensor = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        prompt_tensor = prompt_tensor.to(self.model.device)  # type: ignore
        if hasattr(prompt_tensor, "input_ids"):
            prompt_tensor = prompt_tensor["input_ids"]

        model_tokenizer = self.tokenizer
        stopping_criteria = (
            self.make_stopping_criteria(stop_sequences, tokenizer=model_tokenizer) if stop_sequences else None
        )
        completion_kwargs["max_new_tokens"] = max_new_tokens
        return dict(
            inputs=prompt_tensor,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            **completion_kwargs,
        )

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if response_format is not None:
            raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
        generation_kwargs = self._prepare_completion_args(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore
        out = self.model.generate(
            **generation_kwargs,
        )
        generated_tokens = out[0, count_prompt_tokens:]        
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if stop_sequences is not None:
            output_text = remove_stop_sequences(output_text, stop_sequences)

        self._last_input_token_count = count_prompt_tokens
        self._last_output_token_count = len(generated_tokens)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={
                "out": output_text,
                "completion_kwargs": {key: value for key, value in generation_kwargs.items() if key != "inputs"},
            },
            token_usage=TokenUsage(
                input_tokens=count_prompt_tokens,
                output_tokens=len(generated_tokens),
            ),
        )

    def generate_stream(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        if response_format is not None:
            raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
        generation_kwargs = self._prepare_completion_args(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )

        # Get prompt token count once
        count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore
        self._last_input_token_count = count_prompt_tokens

        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs={"streamer": self.streamer, **generation_kwargs})
        thread.start()

        # Process streaming output
        is_first_token = True
        count_generated_tokens = 0
        for new_text in self.streamer:
            count_generated_tokens += 1
            # Only include input tokens in the first yielded token
            input_tokens = count_prompt_tokens if is_first_token else 0
            is_first_token = False
            yield ChatMessageStreamDelta(
                content=new_text,
                tool_calls=None,
                token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=1),
            )
            count_prompt_tokens = 0
        thread.join()

        # Update final output token count
        self._last_output_token_count = count_generated_tokens