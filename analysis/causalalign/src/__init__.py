# src/causalalign/__init__.py
# from .dataset_generation.dataset_gen import generate_dataset

# __all__ = ["generate_dataset"]


from .domain import create_domain_dict
from .processing import expand_df_by_task_queries, expand_domain_to_dataframe
from .prompt import generate_prompt_dataframe
from .utils import append_dfs
from .verbalization import (
    verbalize_causal_mechanism,
    verbalize_domain_intro,
    verbalize_inference_task,
)

# Define public API
__all__ = [
    "create_domain_dict",
    "expand_domain_to_dataframe",
    "expand_df_by_task_queries",
    "verbalize_domain_intro",
    "verbalize_causal_mechanism",
    "verbalize_inference_task",
    "generate_prompt_dataframe",
    "append_dfs",
]
