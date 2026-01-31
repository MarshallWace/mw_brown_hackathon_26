"""LLM client abstraction for swappable backends."""

import json
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def completion(self, messages: list[dict]) -> str:
        """Call the LLM with a list of messages and return the response text.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            The LLM response text.
        """
        pass

    def structured_completion(self, messages: list[dict], output_cls: type[T]) -> T:
        """Call the LLM and parse the response into a Pydantic model.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            output_cls: Pydantic model class to parse the response into.

        Returns:
            Parsed Pydantic model instance.
        """
        modified_messages = self._add_json_schema_instruction(messages, output_cls)
        response = self.completion(modified_messages)
        return self._parse_json_response(response, output_cls)

    async def acompletion(self, messages: list[dict]) -> str:
        """Async version of completion. Subclasses should override for true async.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            The LLM response text.
        """
        # Default implementation falls back to sync
        return self.completion(messages)

    async def astructured_completion(self, messages: list[dict], output_cls: type[T]) -> T:
        """Async version of structured_completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            output_cls: Pydantic model class to parse the response into.

        Returns:
            Parsed Pydantic model instance.
        """
        modified_messages = self._add_json_schema_instruction(messages, output_cls)
        response = await self.acompletion(modified_messages)
        return self._parse_json_response(response, output_cls)

    def _add_json_schema_instruction(self, messages: list[dict], output_cls: type[T]) -> list[dict]:
        """Add JSON schema instruction to messages."""
        schema_json = json.dumps(output_cls.model_json_schema(), indent=2)
        json_instruction = (
            f"\n\nYou MUST respond with valid JSON that matches this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Respond ONLY with the JSON object, no other text."
        )

        modified_messages = messages.copy()
        if modified_messages:
            last_msg = modified_messages[-1].copy()
            last_msg["content"] = last_msg["content"] + json_instruction
            modified_messages[-1] = last_msg

        return modified_messages

    def _parse_json_response(self, response: str, output_cls: type[T]) -> T:
        """Parse JSON from LLM response into Pydantic model."""
        json_str = response.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()

        return output_cls.model_validate_json(json_str)


def create_llm_client(
    backend: str = "litellm",
    model: str | None = None,
    application_name: str = "mw-hackathon-brown",
) -> BaseLLMClient:
    """Factory function to create an LLM client.

    Args:
        backend: Which backend to use ("litellm").
        model: Model name. Defaults vary by backend.
        application_name: App name for cost tracking.

    Returns:
        An LLM client instance.

    Raises:
        ValueError: If an unknown backend is specified.
    """
    if backend == "litellm":
        from mw_hackathon_brown.litellm_client import LiteLLMClient

        return LiteLLMClient(model=model)

        if model is None:
            model = "gemini-2.5-flash"
    else:
        raise ValueError(f"Unknown backend: {backend}.")
