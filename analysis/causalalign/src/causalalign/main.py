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

# from analysis.causalalign.src.causalalign.experiments.call_apis import ExperimentRunner
from causalalign.experiments.call_apis import ExperimentRunner
from causalalign.llm.client import LLMConfig
from dotenv import load_dotenv

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
            model_name="gpt-4o",  # "gpt-4.5" "gpt-3.5-turbo",
        )

    # Add Gemini if key exists
    if google_key:
        provider_configs["gemini"] = LLMConfig(
            provider="gemini",
            api_key=google_key,
            model_name="gemini-2.0-pro-exp-02-05",  # "gemini-1.5-pro", "gemini-pro"
        )

    # Add Claude if key exists
    if anthropic_key:
        provider_configs["claude"] = LLMConfig(
            provider="claude",
            api_key=anthropic_key,
            model_name="claude-3-opus-20240229",  # "claude-3-7-sonnet-20250219","claude-3-5-sonnet-20240620", "claude-3-5-haiku-20241022"
        )

    if not provider_configs:
        raise ValueError(
            "No API keys found. Please set at least one API key in .env file"
        )

    runner = ExperimentRunner(
        provider_configs=provider_configs, version="2_v", cot=False, n_times=4
    )

    runner.run(
        # sub_folder_xs=["vanilla"], temperature_value_xs=[0.5, 0.7, 0.3, 0.0, 1.0]
        input_path="datasets/17_rw",
        output_path="../../results/17_rw",
        sub_folder_xs=["prompts_for_LLM_api"],
        temperature_value_xs=[0.0, 0.3, 0.5, 0.7, 1.0],
    )  # other temperature values ran: 0.5, 0.7, 0.3 -> for gpt40 and gemini-pro
