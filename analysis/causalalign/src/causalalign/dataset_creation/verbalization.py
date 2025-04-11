# def verbalize_domain_intro(domain_dict):
#     """
#     Extract and return the domain introduction text.

#     Parameters:
#     -----------
#     domain_dict : dict
#         A dictionary containing domain information.

#     Returns:
#     --------
#     str
#         The introduction text of the domain.
#     """
#     # return domain_dict["domain"]["introduction"]
#     return domain_dict["introduction"]


# def verbalize_causal_mechanism(domain_dict, row, graph_type, graph_structures):
#     """
#     Construct a causal mechanism statement based on the selected graph structure.

#     Parameters:
#     -----------
#     domain_dict : dict
#         A dictionary containing domain information.
#     df : pd.DataFrame
#         DataFrame containing expanded domain data.
#     graph_type : str
#         The type of graph structure (e.g., "collider", "fork", "chain").
#     graph_structures : dict
#         A dictionary containing graph structure templates.

#     Returns:
#     --------
#     str
#         The verbalized causal mechanism statement.
#     """
#     if graph_type in graph_structures:
#         template = graph_structures[graph_type]["causal_template"]
#         causal_text = template.format(
#             c1_sense=row["C1_sense"],
#             c1_name=row["C1"],
#             c2_sense=row["C2_sense"],
#             c2_name=row["C2"],
#             e_sense=row["E_sense"],
#             e_name=row["E"],
#         )
#         return (
#             causal_text
#             # " Assume you live in a world that works like this: " + causal_text
#         )  # + "."
#     else:
#         return ""


# def verbalize_inference_task(row, nested_dict, prompt_type):
#     """
#     Generate the verbalized inference task based on a DataFrame row.

#     Parameters:
#     -----------
#     row : pd.Series
#         A row from the DataFrame containing the inference task information.
#     nested_dict : dict
#         The nested dictionary containing value mappings for variables created by `create_domain_dict`.
#     prompt_type : str
#         The type of prompt to be displayed to the LLM. Default is single numeric response.
#         But it can be changed to a different prompt type, e.g. CoT.

#     Returns:
#     --------
#     str
#         The verbalized inference task statement.
#     """
#     # Extract the domain variables dictionary
#     # variables_dict = nested_dict["domain"]["variables"]
#     variables_dict = nested_dict["variables"]

#     # Split the observations into a list
#     observations = row["observation"].split(", ")

#     # Construct the observation text
#     # observation_text = " Now suppose you observe the following:  "
#     observation_text = " You are currently observing: "

#     observation_descriptions = []

#     for obs in observations:
#         var_name, value = obs.split(
#             "="
#         )  # Extract variable (e.g., C2) and value (e.g., 0)
#         cntbl_type = row[f"{var_name}_cntbl"]  # Get counterbalancing type ('p' or 'm')

#         # Lookup the correct dictionary entry for this variable
#         if var_name not in variables_dict:
#             raise KeyError(
#                 f"Variable {var_name} not found in nested_dict['domain']['variables']"
#             )

#         value_mapping = variables_dict[
#             var_name
#         ]  # Get dictionary entry for C1, C2, or E

#         # Determine the correct sense label based on counterbalancing type (p or m)
#         if f"{cntbl_type}_value" in value_mapping:
#             sense_label = value_mapping[f"{cntbl_type}_value"].get(value)
#         else:
#             # If only one exists, default to the available one
#             available_mapping = value_mapping.get(
#                 "p_value", value_mapping.get("m_value")
#             )
#             sense_label = available_mapping.get(value)

#         if sense_label is None:
#             raise KeyError(
#                 f"Value {value} for {var_name} is missing in nested_dict['domain']['variables'][{var_name}]"
#             )

#         # Append the formatted observation
#         observation_descriptions.append(f"{sense_label} {row[var_name]}")

#     observation_text += ", ".join(observation_descriptions) + "."

#     # Extract the query variable and its sense
#     query_var = row["query_node"].split("=")[0]
#     query_sense = row[f"{query_var}_sense"]

#     # Construct the query text
#     # query_text = f" Given the observations and the causal mechanism, how likely on a scale from 0 to 100 is {query_sense} {row[query_var]}? 0 means completely unlikely and 100 means completely certain. {prompt_type}."
#     query_text = f" Your task is to estimate how likely it is that {query_sense} {row[query_var]} are present on a scale from 0 to 100, given the observations and causal relationships described. 0 means completely unlikely and 100 means completely certain. {prompt_type}"
#     # prompt_type = is provided in notebooks and indicates how response should be formatted and for example if CoT should be used.
#     # Combine observation text and query text
#     return observation_text + query_text


# ### abc version

# def verbalize_domain_intro(domain_dict):
#     """
#     Extract and return the domain introduction text.

#     Parameters:
#     -----------
#     domain_dict : dict
#         A dictionary containing domain information.

#     Returns:
#     --------
#     str
#         The introduction text of the domain.
#     """
#     # return domain_dict["domain"]["introduction"]
#     return domain_dict["introduction"]


# def verbalize_causal_mechanism(domain_dict, row, graph_type, graph_structures):
#     """
#     Construct a causal mechanism statement based on the selected graph structure.

#     Parameters:
#     -----------
#     domain_dict : dict
#         A dictionary containing domain information.
#     df : pd.DataFrame
#         DataFrame containing expanded domain data.
#     graph_type : str
#         The type of graph structure (e.g., "collider", "fork", "chain").
#     graph_structures : dict
#         A dictionary containing graph structure templates.

#     Returns:
#     --------
#     str
#         The verbalized causal mechanism statement.
#     """
#     if graph_type in graph_structures:
#         template = graph_structures[graph_type]["causal_template"]
#         causal_text = template.format(
#             a_sense=row["A_sense"],
#             a_name=row["A"],
#             b_sense=row["B_sense"],
#             b_name=row["B"],
#             c_sense=row["C_sense"],
#             c_name=row["C"],
#         )
#         return (
#             causal_text
#             # " Assume you live in a world that works like this: " + causal_text
#         )  # + "."
#     else:
#         return ""


# def verbalize_inference_task(row, nested_dict, prompt_type):
#     """
#     Generate the verbalized inference task based on a DataFrame row.

#     Parameters:
#     -----------
#     row : pd.Series
#         A row from the DataFrame containing the inference task information.
#     nested_dict : dict
#         The nested dictionary containing value mappings for variables created by `create_domain_dict`.
#     prompt_type : str
#         The type of prompt to be displayed to the LLM. Default is single numeric response.
#         But it can be changed to a different prompt type, e.g. CoT.

#     Returns:
#     --------
#     str
#         The verbalized inference task statement.
#     """
#     # Extract the domain variables dictionary
#     # variables_dict = nested_dict["domain"]["variables"]
#     variables_dict = nested_dict["variables"]

#     # Split the observations into a list
#     observations = row["observation"].split(", ")

#     # Construct the observation text
#     # observation_text = " Now suppose you observe the following:  "
#     observation_text = " You are currently observing: "

#     observation_descriptions = []

#     for obs in observations:
#         var_name, value = obs.split(
#             "="
#         )  # Extract variable (e.g., B) and value (e.g., 0)
#         cntbl_type = row[f"{var_name}_cntbl"]  # Get counterbalancing type ('p' or 'm')

#         # Lookup the correct dictionary entry for this variable
#         if var_name not in variables_dict:
#             raise KeyError(
#                 f"Variable {var_name} not found in nested_dict['domain']['variables']"
#             )

#         value_mapping = variables_dict[var_name]  # Get dictionary entry for A, B, or C

#         # Determine the correct sense label based on counterbalancing type (p or m)
#         if f"{cntbl_type}_value" in value_mapping:
#             sense_label = value_mapping[f"{cntbl_type}_value"].get(value)
#         else:
#             # If only one exists, default to the available one
#             available_mapping = value_mapping.get(
#                 "p_value", value_mapping.get("m_value")
#             )
#             sense_label = available_mapping.get(value)

#         if sense_label is None:
#             raise KeyError(
#                 f"Value {value} for {var_name} is missing in nested_dict['domain']['variables'][{var_name}]"
#             )

#         # Append the formatted observation
#         observation_descriptions.append(f"{sense_label} {row[var_name]}")

#     observation_text += ", ".join(observation_descriptions) + "."

#     # Extract the query variable and its sense
#     query_var = row["query_node"].split("=")[0]
#     query_sense = row[f"{query_var}_sense"]

#     # Construct the query text
#     # query_text = f" Given the observations and the causal mechanism, how likely on a scale from 0 to 100 is {query_sense} {row[query_var]}? 0 means completely unlikely and 100 means completely certain. {prompt_type}."
#     query_text = f" Your task is to estimate how likely it is that {query_sense} {row[query_var]} are present on a scale from 0 to 100, given the observations and causal relationships described. 0 means completely unlikely and 100 means completely certain. {prompt_type}"
#     # prompt_type = is provided in notebooks and indicates how response should be formatted and for example if CoT should be used.
#     # Combine observation text and query text
#     return observation_text + query_text


### xyz version


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
    # return domain_dict["domain"]["introduction"]
    return domain_dict["introduction"]


def verbalize_causal_mechanism(domain_dict, row, graph_type, graph_structures):
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
            x_sense=row["X_sense"],
            x_name=row["X"],
            y_sense=row["Y_sense"],
            y_name=row["Y"],
            z_sense=row["Z_sense"],
            z_name=row["Z"],
        )
        return (
            causal_text
            # " Assume you live in a world that works like this: " + causal_text
        )  # + "."
    else:
        return ""


def verbalize_inference_task(row, nested_dict, prompt_type):
    """
    Generate the verbalized inference task based on a DataFrame row.

    Parameters:
    -----------
    row : pd.Series
        A row from the DataFrame containing the inference task information.
    nested_dict : dict
        The nested dictionary containing value mappings for variables created by `create_domain_dict`.
    prompt_type : str
        The type of prompt to be displayed to the LLM. Default is single numeric response.
        But it can be changed to a different prompt type, e.g. CoT.

    Returns:
    --------
    str
        The verbalized inference task statement.
    """
    # Extract the domain variables dictionary
    # variables_dict = nested_dict["domain"]["variables"]
    variables_dict = nested_dict["variables"]

    # Split the observations into a list
    observations = row["observation"].split(", ")

    # Construct the observation text
    # observation_text = " Now suppose you observe the following:  "
    observation_text = " You are currently observing: "

    observation_descriptions = []

    for obs in observations:
        var_name, value = obs.split(
            "="
        )  # Extract variable (e.g., B) and value (e.g., 0)
        cntbl_type = row[f"{var_name}_cntbl"]  # Get counterbalancing type ('p' or 'm')

        # Lookup the correct dictionary entry for this variable
        if var_name not in variables_dict:
            raise KeyError(
                f"Variable {var_name} not found in nested_dict['domain']['variables']"
            )

        value_mapping = variables_dict[
            var_name
        ]  # Get dictionary entry for A, B, or C / X, Y, Z

        # Determine the correct sense label based on counterbalancing type (p or m)
        if f"{cntbl_type}_value" in value_mapping:
            sense_label = value_mapping[f"{cntbl_type}_value"].get(value)
        else:
            # If only one exists, default to the available one
            available_mapping = value_mapping.get(
                "p_value", value_mapping.get("m_value")
            )
            sense_label = available_mapping.get(value)

        if sense_label is None:
            raise KeyError(
                f"Value {value} for {var_name} is missing in nested_dict['domain']['variables'][{var_name}]"
            )

        # Append the formatted observation
        observation_descriptions.append(f"{sense_label} {row[var_name]}")

    observation_text += ", ".join(observation_descriptions) + "."

    # Extract the query variable and its sense
    query_var = row["query_node"].split("=")[0]
    query_sense = row[f"{query_var}_sense"]

    # Construct the query text
    # query_text = f" Given the observations and the causal mechanism, how likely on a scale from 0 to 100 is {query_sense} {row[query_var]}? 0 means completely unlikely and 100 means completely certain. {prompt_type}."
    query_text = f" Your task is to estimate how likely it is that {query_sense} {row[query_var]} are present on a scale from 0 to 100, given the observations and causal relationships described. 0 means completely unlikely and 100 means completely certain. {prompt_type}"
    # prompt_type = is provided in notebooks and indicates how response should be formatted and for example if CoT should be used.
    # Combine observation text and query text
    return observation_text + query_text
