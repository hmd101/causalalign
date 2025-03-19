import os

"""
ExperimentRunner is a class designed to manage and execute experiments involving 
large language models (LLMs). It handles the setup of experiment configurations, 
processes input files, generates responses using specified LLMs, logs initial 
responses, and saves the results.

Experiment runner can be executed in main.py file under causalalign/src/causalalign/main.py

Attributes:
    provider_configs (Dict[str, LLMConfig]): A dictionary containing configurations 
        for different LLM providers.
    version (str): The version identifier for the experiment.
    cot (bool): A flag indicating whether to use chain-of-thought (CoT) prompting.
    n_times (int): The number of times to repeat the prompts.
    run_id (str): A unique identifier for the experiment run.
    version_flag (str): A flag indicating the version of the experiment.
    log_file (str): The file path for logging initial responses.
    init_log (pd.DataFrame): A DataFrame to store initial responses.
    input_path (str): The path to the input files.
    results_folder (str): The path to the folder where results will be saved.

Methods:
    _setup_results_folder() -> str:
        Sets up the results folder based on the version and CoT flag.
    
    _load_or_create_log() -> pd.DataFrame:
        Loads the log file if it exists, otherwise creates a new log DataFrame.
    
    process_file(input_file: str, subfolder: str, llm_client: BaseLLMClient, temperature_value: float):
        Processes a single input file, generates responses using the specified LLM, 
        and logs the results.
    
    _log_response(input_file: str, init_response: str, model_name: str, temperature: float, subfolder: str, prompt_type: str):
        Logs the initial response for a given input file.
    
    _save_results(results: list, cnt_cond: str, model_name: str, subfolder: str, temperature: float, input_file: str):
        Saves the generated responses to a CSV file.
    
    run(sub_folder_xs: list, temperature_value_xs: list):
        Runs the experiment for the specified subfolders and temperature values.
"""
import time
import uuid
from datetime import datetime
from typing import Dict

import pandas as pd

from ..llm.client import BaseLLMClient, LLMConfig, create_llm_client


# In causalalign/src/causalalign/data/experiment.py
class ExperimentRunner:
    def __init__(
        self,
        provider_configs: Dict[str, LLMConfig],
        version: str = "2_v",
        cot: bool = False,
        combine_cnt_cond: bool = False,
        n_times: int = 4,
        input_path: str = None,  # Allow passing input path
        output_path: str = None,  # Allow passing output path
    ):
        self.provider_configs = provider_configs
        self.version = version
        self.cot = cot
        self.n_times = n_times
        self.combine_cnt_cond = combine_cnt_cond
        self.run_id = str(uuid.uuid4())
        self.version_flag = "1_experiment"
        self.log_file = "init_responses_log.csv"
        self.init_log = self._load_or_create_log()

        # Use provided input_path, otherwise default to "17_rw/prompts_only"
        if input_path:
            self.input_path = input_path
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.input_path = os.path.join(current_dir, "17_rw", "prompts_only")

        # Use provided output_path, otherwise default to "results/{version}"
        self.results_folder = (
            output_path if output_path else self._setup_results_folder()
        )

    def _setup_results_folder(self) -> str:
        base_folder = os.path.join(
            os.path.dirname(self.input_path), "results", self.version
        )
        if self.cot:
            base_folder += "_cot"
        os.makedirs(base_folder, exist_ok=True)
        return base_folder

    def _load_or_create_log(self) -> pd.DataFrame:
        if os.path.exists(self.log_file):
            return pd.read_csv(self.log_file, sep=";")
        return pd.DataFrame(columns=["file_name", "init_response"])

    def process_file(
        self,
        input_file: str,
        subfolder: str,
        llm_client: BaseLLMClient,
        temperature_value: float,
    ):
        print(
            f"Processing {input_file} with model {llm_client.model_name} at temperature {temperature_value}"
        )

        cnt_cond = None
        if self.combine_cnt_cond:
            try:
                cnt_cond = input_file.split("_")[2]
            except IndexError:
                print(
                    f"Warning: Could not extract cnt_cond from {input_file}. Using None."
                )

        # prompt_type = input_file.split("_")[3]
        # prompt_type = input_file["prompt_category"].unique()[0]
        prompt_type = "single_numeric_response"

        unique_prompt = ""
        # unique_prompt = (
        #     "In the following, you will be presented with different cause and effect relationships, "
        #     "a set of observations, and an inference task. Your task is to estimate the exact likelihood "
        #     "between 0 and 100 (where 0 means very unlikely and 100 means very likely) based on the "
        #     "cause-effect relationships and observations you are presented with."
        # )

        if self.cot:
            unique_prompt += (
                " Explain your reasoning step by step tags <cot> </cot> and state all the assumptions "
                "you're making. For example, you could say: 'I think that the likelihood of the question "
                "is ... because ...'. Within the tags <cot> </cot>, introduce fitting tags such as for "
                "assumptions <assumptions> </assumptions>."
            )

        try:
            init_response = llm_client.generate_response(
                unique_prompt, temperature_value
            ).content
            print(f"Initialization response for {input_file}: {init_response}")
        except Exception as e:
            print(f"Error sending initialization prompt for file {input_file}: {e}")
            init_response = f"Error: {e}"

        self._log_response(
            input_file,
            init_response,
            llm_client.model_name,
            temperature_value,
            subfolder,
            prompt_type,
        )

        input_file_path = os.path.join(self.input_path, subfolder)
        df = pd.read_csv(os.path.join(input_file_path, input_file), delimiter=",")

        # Repeat prompts for `n_times`
        df = pd.concat([df] * self.n_times, ignore_index=True)

        results = []
        for index, row in df.iterrows():
            try:
                response = llm_client.generate_response(
                    row["prompt"], temperature_value
                )
                results.append(
                    {
                        "id": row["id"],
                        "response": response.content,
                        "prompt": row["prompt"],  # Include prompt
                        "prompt_category": row["prompt_category"],  # New
                        "graph": row["graph"],  # New
                        "domain": row["domain"],
                        "task": row["task"],
                        "cntbl_cond": row["cntbl_cond"],  # New
                    }
                )
                print(f"Processed prompt {index + 1}/{len(df)} from file {input_file}")
                # every 5 prompts print the model and temperature
                if index % 10 == 0:
                    print(
                        f"Model: {llm_client.model_name}, Temperature: {temperature_value}"
                    )

            except Exception as e:
                print(f"Error at index {index} in file {input_file}: {e}")
                results.append(
                    {
                        "id": row["id"],
                        "response": f"Error: {e}",
                        # "prompt": row["prompt"],
                        "prompt_category": row["prompt_category"],
                        "graph": row["graph"],
                        "cntbl_cond": row["cntbl_cond"],
                    }
                )
            time.sleep(1)

        self._save_results(
            results,
            cnt_cond,
            llm_client.model_name,
            subfolder,
            temperature_value,
            input_file,
        )

    def _log_response(
        self,
        input_file: str,
        init_response: str,
        model_name: str,
        temperature: float,
        subfolder: str,
        prompt_type: str,
    ):
        log_entry = pd.DataFrame(
            [
                {
                    "file_name": input_file,
                    "init_response": init_response,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "run_id": self.run_id,
                    "version_flag": self.version_flag,
                    "model": model_name,
                    "temperature": temperature,
                    "type": subfolder,
                    "prompt_type": prompt_type,
                }
            ]
        )
        self.init_log = pd.concat([self.init_log, log_entry], ignore_index=True)
        self.init_log.to_csv(self.log_file, index=False, sep=";")

    def _save_results(
        self,
        results: list,
        cnt_cond: str,
        model_name: str,
        subfolder: str,
        temperature: float,
        input_file: str,
    ):
        # folder_path = os.path.join(self.results_folder, model_name, subfolder)
        folder_path = os.path.join(self.results_folder, model_name)

        os.makedirs(folder_path, exist_ok=True)

        file_name = f"{self.version}_"
        if self.combine_cnt_cond and cnt_cond is not None:
            file_name += f"{cnt_cond}_"
        file_name += f"{model_name}_{temperature}_temp"

        if self.cot:
            file_name += "_cot"

        file_name += ".csv"
        file_path = os.path.join(folder_path, file_name)

        response_df = pd.DataFrame(results)
        response_df["subject"] = model_name
        response_df["temperature"] = temperature

        response_df.to_csv(file_path, index=False, sep=";")
        print(f"Responses saved to {file_path}")

    def run(
        self,
        input_path: str = None,
        output_path: str = None,
        sub_folder_xs: list = None,
        temperature_value_xs: list = None,
    ):
        """
        Runs the experiment, processing input files and generating results.

        Parameters:
        -----------
        input_path: str
            Path to the folder containing input CSV files. If None, uses the default.
        output_path: str
            Path to store results. If None, uses the default results folder.
        sub_folder_xs: list
            List of subfolders to process.
        temperature_value_xs: list
            List of temperature values for LLM generation.
        """
        # Override paths if specified
        if input_path:
            self.input_path = input_path
        if output_path:
            self.results_folder = output_path

        if sub_folder_xs is None or temperature_value_xs is None:
            raise ValueError("sub_folder_xs and temperature_value_xs must be provided.")

        for subfolder in sub_folder_xs:
            subfolder_path = os.path.join(self.input_path, subfolder)
            if not os.path.exists(subfolder_path):
                print(
                    f"Warning: Subfolder {subfolder_path} does not exist. Skipping..."
                )
                continue

            input_files = [f for f in os.listdir(subfolder_path) if f.endswith(".csv")]

            for input_file in input_files:
                for temperature_value in temperature_value_xs:
                    for provider, config in self.provider_configs.items():
                        llm_client = create_llm_client(config)
                        self.process_file(
                            input_file, subfolder, llm_client, temperature_value
                        )
