import abc

"""
This module provides a standardized interface for interacting with various Large Language Model (LLM) providers such as OpenAI, Gemini, and Claude. It includes abstract base classes, specific client implementations for each provider, and a factory class to create the appropriate client based on the provider name.

Classes:
    LLMResponse: A dataclass to standardize the response format across different LLM providers.
    BaseLLMClient: An abstract base class defining the interface for LLM clients.
    OpenAIClient: A client implementation for interacting with OpenAI's GPT models.
    GeminiClient: A client implementation for interacting with Gemini's generative models.
    ClaudeClient: A client implementation for interacting with Claude's models.
    LLMClientFactory: A factory class to create appropriate LLM client instances based on the provider.
    LLMConfig: A configuration class to hold provider-specific settings.

Functions:
    create_llm_client: Creates an LLM client instance based on the provided configuration.
"""
from dataclasses import dataclass
from typing import Any, Optional

import google.generativeai as genai
import openai
from anthropic import Anthropic


@dataclass
class LLMResponse:
    """Standardized response format across different LLM providers"""

    content: str
    model_name: str
    raw_response: Any  # Store the original response for debugging


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM clients"""

    @abc.abstractmethod
    def generate_response(self, prompt: str, temperature: float = 0.7) -> LLMResponse:
        """Generate a response from the LLM"""
        pass


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str, temperature: float = 0.7) -> LLMResponse:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                model_name=self.model_name,
                raw_response=response,
            )
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def generate_response(self, prompt: str, temperature: float = 0.7) -> LLMResponse:
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature),
            )
            return LLMResponse(
                content=response.text, model_name=self.model_name, raw_response=response
            )
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")


class ClaudeClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str = "claude-3-opus-20240229"):
        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name

    def generate_response(self, prompt: str, temperature: float = 0.7) -> LLMResponse:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,  # Added max_tokens parameter
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return LLMResponse(
                content=response.content[0].text,
                model_name=self.model_name,
                raw_response=response,
            )
        except Exception as e:
            raise Exception(f"Claude API error: {str(e)}")


class LLMClientFactory:
    """Factory class to create appropriate LLM client based on provider"""

    @staticmethod
    def create_client(
        provider: str, api_key: str, model_name: Optional[str] = None
    ) -> BaseLLMClient:
        if provider.lower() == "openai":
            return OpenAIClient(api_key, model_name or "gpt-3.5-turbo")
        elif provider.lower() == "gemini":
            return GeminiClient(api_key, model_name or "gemini-pro")
        elif provider.lower() == "claude":
            return ClaudeClient(api_key, model_name or "claude-3-opus-20240229")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


# Example configuration class
class LLMConfig:
    def __init__(self, provider: str, api_key: str, model_name: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name


# Usage example for your script
def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    return LLMClientFactory.create_client(
        provider=config.provider, api_key=config.api_key, model_name=config.model_name
    )
