"""Gemini client implementation using google-generativeai SDK."""

import os

from mw_hackathon_brown.llm_client import BaseLLMClient


class LiteLLMClient(BaseLLMClient):
    """LLM client using Google's generativeai SDK directly."""

    def __init__(self, model: str | None = None):
        """Initialize the Gemini client.

        Args:
            model: Model name (e.g., "gemini-2.0-flash", "gemini-2.5-flash").
                   Defaults to GEMINI_MODEL env var or "gemini-2.0-flash".
        """
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)

        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._model = genai.GenerativeModel(self.model_name)

    def _build_prompt(self, messages: list[dict]) -> str:
        """Convert OpenAI-style messages to a single prompt for Gemini."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)
        return "\n\n".join(prompt_parts)

    def completion(self, messages: list[dict]) -> str:
        """Call the LLM with a list of messages and return the response text."""
        prompt = self._build_prompt(messages)
        response = self._model.generate_content(prompt)
        return response.text

    async def acompletion(self, messages: list[dict]) -> str:
        """Async call to the LLM."""
        prompt = self._build_prompt(messages)
        response = await self._model.generate_content_async(prompt)
        return response.text
