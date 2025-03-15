import itertools

import pandas as pd


def expand_domain_to_dataframe(domain_dict):
    """
    Expand a domain dictionary into a DataFrame with 8 unique counterbalance condition combinations.

    Parameters:
    -----------
    domain_dict : dict
        A dictionary created using create_domain_dict() function.

    Returns:
    --------
    pd.DataFrame
        DataFrame where each row represents a unique combination of variable values.
    """
    domain_name = domain_dict["domain"]["domain_name"]
    variables = domain_dict["domain"]["variables"]

    # Prepare data storage
    data = []
    var_keys = []  # Stores variable keys (C1, C2, E)
    var_names = []  # Stores actual variable names (e.g., "interest rates")
    var_p_values = []
    var_m_values = []

    # Extract variable details and prepare p/m values separately
    for var, details in variables.items():
        var_name = details[f"{var.lower()}_name"]  # Fetch actual variable name
        var_detailed = details[f"{var.lower()}_detailed"]

        p_values = [("p", key, val) for key, val in details["p_value"].items()]
        m_values = (
            [("m", key, val) for key, val in details["m_value"].items()]
            if "m_value" in details
            else p_values  # Default to p_values if m_values are absent
        )

        var_keys.append(var)
        var_names.append(var_name)  # Store the actual variable name
        var_p_values.append(p_values)
        var_m_values.append(m_values)

    # Construct only the 2^3 = 8 unique counterbalance combinations
    cntbl_conditions = list(itertools.product(["p", "m"], repeat=3))

    for cntbl_combo in cntbl_conditions:
        row = {"domain": domain_name}
        cntbl_cond = ""  # Counterbalance condition initialization

        for var, var_name, cntbl, p_vals, m_vals in zip(
            var_keys, var_names, cntbl_combo, var_p_values, var_m_values
        ):
            chosen_values = p_vals if cntbl == "p" else m_vals
            var_cntbl, var_value, var_sense = chosen_values[
                0
            ]  # Select the first key-value pair

            row[f"{var}"] = var_name  # Assign actual variable name
            row[f"{var}_values"] = var_value
            row[f"{var}_cntbl"] = var_cntbl
            row[f"{var}_sense"] = var_sense
            row[f"{var}_detailed"] = variables[var][f"{var.lower()}_detailed"]

            cntbl_cond += var_cntbl  # Concatenate all counterbalance conditions

        row["cntbl_cond"] = cntbl_cond
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    return df


def expand_df_by_task_queries(df, inference_tasks):
    """
    Expand the dataframe with verbalized inference tasks.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame created using expand_domain_to_dataframe.
    inference_tasks : dict
        Dictionary of inference tasks mapping.

    Returns:
    --------
    pd.DataFrame
        DataFrame where each row represents a unique combination of variable values and an inference task.
    """
    expanded_rows = []

    for _, row in df.iterrows():
        for task, task_info in inference_tasks.items():
            # For tasks a-h and j, do both (C1, C2) and (C2, C1)
            if task in {"a", "b", "c", "d", "e", "f", "g", "h", "j"}:
                for Ci, Cj in [("C1", "C2"), ("C2", "C1")]:
                    new_row = row.copy()
                    new_row["task"] = task
                    new_row["query_node"] = task_info["query_node"].replace("Ci", Ci)
                    new_row["observation"] = (
                        task_info["observation"].replace("Ci", Ci).replace("Cj", Cj)
                    )
                    new_row["query"] = (
                        task_info["query"].replace("Ci", Ci).replace("Cj", Cj)
                    )
                    expanded_rows.append(new_row)

            # For tasks i and k, only do one assignment
            else:
                new_row = row.copy()
                new_row["task"] = task
                new_row["query_node"] = (
                    task_info["query_node"].replace("Ci", "C1").replace("Cj", "C2")
                )
                new_row["observation"] = (
                    task_info["observation"].replace("Ci", "C1").replace("Cj", "C2")
                )
                new_row["query"] = (
                    task_info["query"].replace("Ci", "C1").replace("Cj", "C2")
                )
                expanded_rows.append(new_row)

    # Create expanded DataFrame
    expanded_df = pd.DataFrame(expanded_rows)

    return expanded_df
