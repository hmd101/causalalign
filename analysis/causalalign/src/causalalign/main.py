import os

"""
This script sets up and runs an experiment using various language model providers.

It loads API keys from a .env file, configures the providers (OpenAI, Gemini, and Claude) 
based on the available keys, and then runs the experiment using the specified configurations.

Classes:
    ExperimentRunner: Manages the execution of the experiment.
    LLMConfig: Holds configuration details for each language model provider.

Functions:
    load_dotenv: Loads environment variables from a .env file.

Usage:
    Ensure that the .env file contains the necessary API keys for the desired providers.
    Run the script to execute the experiment with the configured providers.
"""

from dotenv import load_dotenv
from causalalign.data.experiment import ExperimentRunner
from causalalign.llm.client import LLMConfig

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    # Configure providers you want to use
    provider_configs = {}

    # Add OpenAI if key exists
    if openai_key:
        provider_configs["openai"] = LLMConfig(
            # provider="openai", api_key=openai_key, model_name="gpt-4o"
            provider="openai",
            api_key=openai_key,
            model_name="gpt-3.5-turbo",
        )

    # Add Gemini if key exists
    if google_key:
        provider_configs["gemini"] = LLMConfig(
            provider="gemini", api_key=google_key, model_name="gemini-pro"
        )

    # Add Claude if key exists
    if anthropic_key:
        provider_configs["claude"] = LLMConfig(
            provider="claude",
            api_key=anthropic_key,
            model_name="claude-3-opus-20240229",
        )

    if not provider_configs:
        raise ValueError(
            "No API keys found. Please set at least one API key in .env file"
        )

    runner = ExperimentRunner(
        provider_configs=provider_configs, version="1_v", cot=False, n_times=5
    )

    runner.run(
        sub_folder_xs=["vanilla"], temperature_value_xs=[0.5, 0.7, 0.3]
    )  # other temperature values ran: 0.5, 0.7, 0.3 -> for gpt40 and gemini-pro
