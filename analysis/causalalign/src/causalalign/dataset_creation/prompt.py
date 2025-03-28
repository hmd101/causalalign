from causalalign.dataset_creation.processing import (
    expand_df_by_task_queries,
    expand_domain_to_dataframe,
)
from causalalign.dataset_creation.verbalization import (
    verbalize_causal_mechanism,
    verbalize_domain_intro,
    verbalize_inference_task,
)


def generate_prompt_dataframe(
    domain_dict,
    inference_tasks,
    graph_type,
    graph_structures,
    prompt_type="Please provide only a numeric response and no additional information",
    prompt_category="single_numeric_response",
    counterbalance_enabled=True,
):
    """
    Expand the DataFrame to include full prompt verbalization and graph structure.

    Parameters:
    -----------
    domain_dict : dict
        A dictionary containing domain information.
    inference_tasks : dict
        A dictionary containing inference tasks.
    graph_type : str
        The type of graph structure (e.g., "collider", "fork", "chain").
    graph_structures : dict
        A dictionary containing graph structure templates.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with expanded prompts and graph structure information.
    """

    # Expand the domain dictionary into a DataFrame
    # containing all unique combinations of variable values
    # (including all 8 counterbalance conditions, subset the desired ones (only 4 used in Rehder 17))
    # and subsequently add the inference task information
    df = expand_df_by_task_queries(
        expand_domain_to_dataframe(domain_dict), inference_tasks
    )

    # Extract the domain introduction text
    domain_intro = verbalize_domain_intro(domain_dict)

    # Construct the causal mechanism statement
    # causal_mechanism = verbalize_causal_mechanism(
    #     domain_dict, df, graph_type, graph_structures
    # )
    # extract graph description in node and edge notation
    # graph_description = graph_structures[graph_type]["description"] if graph_type in graph_structures else ""

    # Extract the graph description in text
    graph_description = graph_type

    # Add the graph description to the DataFrame
    df["graph"] = graph_description

    # Generate the full prompt by combining domain introduction, causal mechanism, and inference task
    # df["prompt"] = df.apply(
    #     lambda row: domain_intro
    #     + causal_mechanism
    #     + verbalize_inference_task(
    #         row, nested_dict=domain_dict, prompt_type=prompt_type
    #     ),
    #     axis=1,
    # )

    df["prompt"] = df.apply(
        lambda row: domain_intro
        + verbalize_causal_mechanism(domain_dict, row, graph_type, graph_structures)
        + verbalize_inference_task(
            row, nested_dict=domain_dict, prompt_type=prompt_type
        ),
        axis=1,
    )

    df["prompt_category"] = prompt_category
    return df


# how to use:
# test_df = generate_prompt_dataframe(economy_domain, inference_tasks, "collider", graph_structures)
