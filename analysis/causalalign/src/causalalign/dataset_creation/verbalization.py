def verbalize_domain_intro(domain_dict):
    """
    Extract and return the domain introduction text.

    Parameters:
    -----------
    domain_dict : dict
        A dictionary containing domain information.

    Returns:
    --------
    str
        The introduction text of the domain.
    """
    return domain_dict["domain"]["introduction"]


def verbalize_causal_mechanism(domain_dict, df, graph_type, graph_structures):
    """
    Construct a causal mechanism statement based on the selected graph structure.

    Parameters:
    -----------
    domain_dict : dict
        A dictionary containing domain information.
    df : pd.DataFrame
        DataFrame containing expanded domain data.
    graph_type : str
        The type of graph structure (e.g., "collider", "fork", "chain").
    graph_structures : dict
        A dictionary containing graph structure templates.

    Returns:
    --------
    str
        The verbalized causal mechanism statement.
    """
    if graph_type in graph_structures:
        template = graph_structures[graph_type]["causal_template"]
        causal_text = template.format(
            c1_sense=df["C1_sense"].iloc[0],
            c1_name=df["C1"].iloc[0],
            c2_sense=df["C2_sense"].iloc[0],
            c2_name=df["C2"].iloc[0],
            e_sense=df["E_sense"].iloc[0],
            e_name=df["E"].iloc[0],
        )
        return " Assume you live in a world that works like this: " + causal_text + "."
    else:
        return ""


def verbalize_inference_task(row):
    """
    Generate the verbalized inference task based on a DataFrame row.

    Parameters:
    -----------
    row : pd.Series
        A row from the DataFrame containing the inference task information.

    Returns:
    --------
    str
        The verbalized inference task statement.
    """
    # Split the observations into a list
    observations = row["observation"].split(", ")

    # Construct the observation text
    observation_text = (
        " Now suppose you observe the following: "
        + ", ".join(
            [
                f"{row[obs.split('=')[0] + '_sense']} {row[obs.split('=')[0]]}"
                for obs in observations
            ]
        )
        + "."
    )

    # Extract the query variable and its sense
    query_var = row["query_node"].split("=")[0]
    query_sense = row[f"{query_var}_sense"]

    # Construct the query text
    query_text = f" Given the observations and the causal mechanism, how likely on a scale from 0 to 100 is {query_sense} {row[query_var]}? 0 means definitely not likely and 100 means definitely likely. Please provide only a numeric response and no additional information."

    # Combine observation text and query text
    return observation_text + query_text
