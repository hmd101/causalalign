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
#             x_sense=row["X_sense"],
#             x_name=row["X"],
#             y_sense=row["Y_sense"],
#             y_name=row["Y"],
#             z_sense=row["Z_sense"],
#             z_name=row["Z"],
#         )
#         return (
#             causal_text
#             # " Assume you live in a world that works like this: " + causal_text
#         )  # + "."
#     else:
#         return ""


# New
def verbalize_causal_mechanism(domain_dict, row, graph_type, graph_structures):
    """
    Create a detailed description of causal relationships with explanations.

    Parameters:
    -----------
    domain_dict : dict
        The domain dictionary
    row : pd.Series
        Row from the expanded dataframe with counterbalance information
    graph_type : str
        Type of causal graph (collider, fork, chain)
    graph_structures : dict
        Graph structure templates

    Returns:
    --------
    str
        Formatted text describing causal relationships with explanations
    """
    if graph_type not in graph_structures:
        return ""

    domain_name = domain_dict["domain_name"].upper()
    causal_text = (
        " "  # f"\n{domain_name}\n********** CAUSAL RELATIONSHIPS **********\n"
    )

    # Get variables and their counterbalance conditions
    x_cntbl = row["X_cntbl"]
    y_cntbl = row["Y_cntbl"]
    z_cntbl = row["Z_cntbl"]

    # Get variable senses
    x_sense = row["X_sense"]
    y_sense = row["Y_sense"]
    z_sense = row["Z_sense"]

    # Get variable names
    x_name = row["X"]
    y_name = row["Y"]
    z_name = row["Z"]

    # Build causal relationships based on graph type
    if graph_type == "collider":
        # X → Z relationship with explanation
        x_z_relation = f"{x_sense} {x_name} cause {z_sense} {z_name}. "

        # Get X → Z explanation
        x_z_key = f"{x_cntbl}_{z_cntbl}"
        x_z_explanation = ""
        if (
            "explanations" in domain_dict["variables"]["X"]
            and x_z_key in domain_dict["variables"]["X"]["explanations"]
        ):
            x_z_explanation = (
                " " + domain_dict["variables"]["X"]["explanations"][x_z_key]
            )

        # Y → Z relationship with explanation
        y_z_relation = f"{y_sense} {y_name} cause {z_sense} {z_name}."

        # Get Y → Z explanation
        y_z_key = f"{y_cntbl}_{z_cntbl}"
        y_z_explanation = ""
        if (
            "explanations" in domain_dict["variables"]["Y"]
            and y_z_key in domain_dict["variables"]["Y"]["explanations"]
        ):
            y_z_explanation = (
                " " + domain_dict["variables"]["Y"]["explanations"][y_z_key]
            )

        # Combine relationships and explanations
        causal_text += f"{x_z_relation}{x_z_explanation} "
        causal_text += f"{y_z_relation}{y_z_explanation} "

    # Add similar handling for fork and chain graphs if needed

    return causal_text


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
#     observation_text = " You are currently observing: "   # TODO: make this domain dependent like this for weahter:  "Suppose that there is a weather system that is known to have...", if the domain is economy, replace it with economy. if it's sociology replace it with society.

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

#         value_mapping = variables_dict[
#             var_name
#         ]  # Get dictionary entry for A, B, or C / X, Y, Z

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

#     observation_text += "and ".join(observation_descriptions) + "."

#     # Extract the query variable and its sense
#     query_var = row["query_node"].split("=")[0]
#     query_sense = row[f"{query_var}_sense"]

#     # Construct the query text
#     # query_text = f" Given the observations and the causal mechanism, how likely on a scale from 0 to 100 is {query_sense} {row[query_var]}? 0 means completely unlikely and 100 means completely certain. {prompt_type}."
#     query_text = f" Your task is to estimate how likely it is that {query_sense} {row[query_var]} are present on a scale from 0 to 100, given the observations and causal relationships described. 0 means completely unlikely and 100 means completely certain. {prompt_type}"
#     # prompt_type = is provided in notebooks and indicates how response should be formatted and for example if CoT should be used.
#     # Combine observation text and query text
#     return observation_text + query_text


# New
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
        The type of prompt to be displayed to the LLM.

    Returns:
    --------
    str
        The verbalized inference task statement.
    """
    variables_dict = nested_dict["variables"]
    observations = row["observation"].split(", ")

    # Set domain-specific observation intro
    domain = row["domain"].lower()
    if domain == "weather":
        observation_text = (
            "Suppose that there is a weather system that is known to have "
        )
    elif domain == "economy":
        observation_text = "Suppose that the economy is currently known to have "
    elif domain == "sociology":
        observation_text = (
            "Suppose that the society you live in currently exhibits the following: "
        )
    else:
        observation_text = "You are currently observing: "

    observation_descriptions = []

    for obs in observations:
        var_name, value = obs.split("=")
        cntbl_type = row[f"{var_name}_cntbl"]

        if var_name not in variables_dict:
            raise KeyError(f"Variable {var_name} not found in nested_dict['variables']")

        value_mapping = variables_dict[var_name]

        if f"{cntbl_type}_value" in value_mapping:
            sense_label = value_mapping[f"{cntbl_type}_value"].get(value)
        else:
            available_mapping = value_mapping.get(
                "p_value", value_mapping.get("m_value")
            )
            sense_label = available_mapping.get(value)

        if sense_label is None:
            raise KeyError(
                f"Value {value} for {var_name} is missing in nested_dict['variables'][{var_name}]"
            )

        observation_descriptions.append(f"{sense_label} {row[var_name]}")

    observation_text += " and ".join(observation_descriptions) + "."

    # Extract the query variable and construct the query
    query_var = row["query_node"].split("=")[0]
    query_sense = row[f"{query_var}_sense"]
    query_text = f" Your task is to estimate how likely it is that {query_sense} {row[query_var]} are present on a scale from 0 to 100, given the observations and causal relationships described. 0 means completely unlikely and 100 means completely certain. {prompt_type}"

    return observation_text + query_text


# New
def verbalize_variables_section(domain_dict, row):
    """
    Create a comprehensive description of all variables based on counterbalance conditions.

    Parameters:
    -----------
    domain_dict : dict
        The domain dictionary containing variable information
    row : pd.Series
        Row from the expanded dataframe with counterbalance information

    Returns:
    --------
    str
        Formatted text describing all variables with their counterbalanced values
    """
    domain_name = domain_dict["domain_name"]

    # replace domain name such that it fits into the sentence
    if domain_name == "economy":
        domain_name = "economies"

    elif domain_name == "sociology":
        domain_name = "societies"

    elif domain_name == "weather":
        domain_name = "weather systems"

    variables_text = " "  # f"\n{domain_name}\n********** VARIABLES **********\n"

    for var_key, var_details in domain_dict["variables"].items():
        name = var_details[f"{var_key}_name"]
        detailed = var_details[f"{var_key}_detailed"]

        # Get appropriate value based on counterbalance
        cntbl = row[f"{var_key}_cntbl"]
        value_dict = var_details["p_value"] if cntbl == "p" else var_details["m_value"]

        # Create description with on/off values
        value_1 = value_dict["1"]
        value_0 = value_dict["0"]

        # Format the description
        variables_text += f"{detailed} Some {domain_name} have {value_1} {name}. Others have {value_0} {name}. "  # this should be different -> needed to reproduce exact RW17 prompts, but not needed for abstract prompts
        # else, for abstract prompts, it should be:
        # variables_text += f"{detailed} Sometimes {name} is {value_1} and sometimes{name} is {value_0}. "

    return variables_text
