# ########## C1, C2  E version ########


# def create_domain_dict(
#     domain,
#     introduction,
#     C1_name,
#     C1_detailed,
#     C1_values,
#     C2_name,
#     C2_detailed,
#     C2_values,
#     E_name,
#     E_detailed,
#     E_values,
#     counterbalance_enabled=False,
#     enforce_zero_label=False,
#     zero_label="normal",  # Default to 'normal' but configurable
# ):
#     """
#     Create a domain dictionary with a configurable zero-value verbalization.

#     Parameters:
#     -----------
#     domain : str
#         Domain name (e.g., "economy", "biology").
#     introduction : str
#         Domain introduction text.
#     C1_name, C2_name, E_name : str
#         Names of variables C1, C2, and E.
#     C1_detailed, C2_detailed, E_detailed : str
#         Detailed descriptions of variables.
#     C1_values, C2_values, E_values : dict
#         Mapping of "1" and "0" to their verbal descriptions.
#         e.g., {"1": "high", "0": "low"}.
#     counterbalance_enabled : bool
#         Whether to enable counterbalancing (p/m conditions).
#     enforce_zero_label : bool
#         Whether to enforce the specified zero_label on the '0' value.
#     zero_label : str
#         The verbalization to be assigned to the "0" value across all variables.

#     Returns:
#     --------
#     dict
#         Domain dictionary in the required format.
#     """

#     def validate_values(values):
#         """Ensure that required keys exist in the value dictionary."""
#         if not isinstance(values, dict):
#             raise ValueError(
#                 "Expected a dictionary for variable values, but received a different type."
#             )
#         if "0" not in values:
#             raise KeyError(
#                 "Missing '0' key in values dictionary. Ensure '0' is explicitly defined."
#             )
#         if "1" not in values:
#             raise KeyError(
#                 "Missing '1' key in values dictionary. Ensure '1' is explicitly defined."
#             )
#         return values

#     # Validate input value dictionaries
#     C1_values = validate_values(C1_values)
#     C2_values = validate_values(C2_values)
#     E_values = validate_values(E_values)

#     # Build the domain dictionary
#     domain_dict = {
#         "domain_name": domain,
#         "introduction": introduction,
#         "variables": {
#             "C1": {
#                 "C1_name": C1_name,
#                 "C1_detailed": C1_detailed,
#                 "p_value": C1_values.copy(),
#             },
#             "C2": {
#                 "C2_name": C2_name,
#                 "C2_detailed": C2_detailed,
#                 "p_value": C2_values.copy(),
#             },
#             "E": {
#                 "E_name": E_name,
#                 "E_detailed": E_detailed,
#                 "p_value": E_values.copy(),
#             },
#         },
#     }

#     # Handle counterbalance vs. simple value assignment
#     if counterbalance_enabled:
#         domain_dict["variables"]["C1"]["m_value"] = {
#             "1": C1_values["0"],
#             "0": C1_values["1"],
#         }
#         domain_dict["variables"]["C2"]["m_value"] = {
#             "1": C2_values["0"],
#             "0": C2_values["1"],
#         }
#         domain_dict["variables"]["E"]["m_value"] = {
#             "1": E_values["0"],
#             "0": E_values["1"],
#         }

#     # Enforce the zero label **after** counterbalancing to preserve logic
#     # if enforce_zero_label:
#     #     domain_dict["variables"]["C1"]["p_value"]["0"] = zero_label
#     #     # domain_dict["variables"]["C1"]["m_value"]["1"] = zero_label
#     #     domain_dict["variables"]["C2"]["p_value"]["0"] = zero_label
#     #     # domain_dict["variables"]["C2"]["m_value"]["1"] = zero_label
#     #     domain_dict["variables"]["E"]["p_value"]["0"] = zero_label
#     #     # domain_dict["variables"]["E"]["m_value"]["1"] = zero_label

#     if enforce_zero_label:
#         for var in ["C1", "C2", "E"]:
#             domain_dict["variables"][var]["p_value"]["0"] = zero_label
#             domain_dict["variables"][var]["m_value"]["0"] = (
#                 zero_label  # Both p and m should have "0" as normal
#             )

#     return domain_dict


# ########## A, B   C version ########


# def create_domain_dict(
#     domain,
#     introduction,
#     A_name,
#     A_detailed,
#     A_values,
#     B_name,
#     B_detailed,
#     B_values,
#     C_name,
#     C_detailed,
#     C_values,
#     counterbalance_enabled=False,
#     enforce_zero_label=False,
#     zero_label="normal",  # Default to 'normal' but configurable
# ):
#     """
#     Create a domain dictionary with a configurable zero-value verbalization.

#     Parameters:
#     -----------
#     domain : str
#         Domain name (e.g., "economy", "biology").
#     introduction : str
#         Domain introduction text.
#     A_name, B_name, C_name : str
#         Names of variables A, B, and E.
#     A_detailed, B_detailed, C_detailed : str
#         Detailed descriptions of variables.
#     A_values, B_values, C_values : dict
#         Mapping of "1" and "0" to their verbal descriptions.
#         e.g., {"1": "high", "0": "low"}.
#     counterbalance_enabled : bool
#         Whether to enable counterbalancing (p/m conditions).
#     enforce_zero_label : bool
#         Whether to enforce the specified zero_label on the '0' value.
#     zero_label : str
#         The verbalization to be assigned to the "0" value across all variables.

#     Returns:
#     --------
#     dict
#         Domain dictionary in the required format.
#     """

#     def validate_values(values):
#         """Ensure that required keys exist in the value dictionary."""
#         if not isinstance(values, dict):
#             raise ValueError(
#                 "Expected a dictionary for variable values, but received a different type."
#             )
#         if "0" not in values:
#             raise KeyError(
#                 "Missing '0' key in values dictionary. Ensure '0' is explicitly defined."
#             )
#         if "1" not in values:
#             raise KeyError(
#                 "Missing '1' key in values dictionary. Ensure '1' is explicitly defined."
#             )
#         return values

#     # Validate input value dictionaries
#     A_values = validate_values(A_values)
#     B_values = validate_values(B_values)
#     C_values = validate_values(C_values)

#     # Build the domain dictionary
#     domain_dict = {
#         "domain_name": domain,
#         "introduction": introduction,
#         "variables": {
#             "A": {
#                 "A_name": A_name,
#                 "A_detailed": A_detailed,
#                 "p_value": A_values.copy(),
#             },
#             "B": {
#                 "B_name": B_name,
#                 "B_detailed": B_detailed,
#                 "p_value": B_values.copy(),
#             },
#             "C": {
#                 "C_name": C_name,
#                 "C_detailed": C_detailed,
#                 "p_value": C_values.copy(),
#             },
#         },
#     }

#     # Handle counterbalance vs. simple value assignment
#     if counterbalance_enabled:
#         domain_dict["variables"]["A"]["m_value"] = {
#             "1": A_values["0"],
#             "0": A_values["1"],
#         }
#         domain_dict["variables"]["B"]["m_value"] = {
#             "1": B_values["0"],
#             "0": B_values["1"],
#         }
#         domain_dict["variables"]["C"]["m_value"] = {
#             "1": C_values["0"],
#             "0": C_values["1"],
#         }

#     # Enforce the zero label **after** counterbalancing to preserve logic
#     # if enforce_zero_label:
#     #     domain_dict["variables"]["A"]["p_value"]["0"] = zero_label
#     #     # domain_dict["variables"]["A"]["m_value"]["1"] = zero_label
#     #     domain_dict["variables"]["B"]["p_value"]["0"] = zero_label
#     #     # domain_dict["variables"]["B"]["m_value"]["1"] = zero_label
#     #     domain_dict["variables"]["C"]["p_value"]["0"] = zero_label
#     #     # domain_dict["variables"]["C"]["m_value"]["1"] = zero_label

#     if enforce_zero_label:
#         for var in ["A", "B", "C"]:
#             domain_dict["variables"][var]["p_value"]["0"] = zero_label
#             domain_dict["variables"][var]["m_value"]["0"] = (
#                 zero_label  # Both p and m should have "0" as normal
#             )

#     return domain_dict


########## X, Y, Z version ########


# def create_domain_dict(
#     domain,
#     introduction,
#     X_name,
#     X_detailed,
#     X_values,
#     Y_name,
#     Y_detailed,
#     Y_values,
#     Z_name,
#     Z_detailed,
#     Z_values,
#     counterbalance_enabled=False,
#     enforce_zero_label=False,
#     zero_label="normal",  # Default to 'normal' but configurable
# ):
#     """
#     Create a domain dictionary with a configurable zero-value verbalization.

#     Parameters:
#     -----------
#     domain : str
#         Domain name (e.g., "economy", "biology").
#     introduction : str
#         Domain introduction text.
#     X_name, Y_name, Z_name : str
#         Names of variables A, B, and E.
#     X_detailed, Y_detailed, Z_detailed : str
#         Detailed descriptions of variables.
#     X_values, Y_values, Z_values : dict
#         Mapping of "1" and "0" to their verbal descriptions.
#         e.g., {"1": "high", "0": "low"}.
#     counterbalance_enabled : bool
#         Whether to enable counterbalancing (p/m conditions).
#     enforce_zero_label : bool
#         Whether to enforce the specified zero_label on the '0' value.
#     zero_label : str
#         The verbalization to be assigned to the "0" value across all variables.

#     Returns:
#     --------
#     dict
#         Domain dictionary in the required format.
#     """

#     def validate_values(values):
#         """Ensure that required keys exist in the value dictionary."""
#         if not isinstance(values, dict):
#             raise ValueError(
#                 "Expected a dictionary for variable values, but received a different type."
#             )
#         if "0" not in values:
#             raise KeyError(
#                 "Missing '0' key in values dictionary. Ensure '0' is explicitly defined."
#             )
#         if "1" not in values:
#             raise KeyError(
#                 "Missing '1' key in values dictionary. Ensure '1' is explicitly defined."
#             )
#         return values

#     # Validate input value dictionaries
#     X_values = validate_values(X_values)
#     Y_values = validate_values(Y_values)
#     Z_values = validate_values(Z_values)

#     # Build the domain dictionary
#     domain_dict = {
#         "domain_name": domain,
#         "introduction": introduction,
#         "variables": {
#             "X": {
#                 "X_name": X_name,
#                 "X_detailed": X_detailed,
#                 "p_value": X_values.copy(),
#             },
#             "Y": {
#                 "Y_name": Y_name,
#                 "Y_detailed": Y_detailed,
#                 "p_value": Y_values.copy(),
#             },
#             "Z": {
#                 "Z_name": Z_name,
#                 "Z_detailed": Z_detailed,
#                 "p_value": Z_values.copy(),
#             },
#         },
#     }

#     # Handle counterbalance vs. simple value assignment
#     if counterbalance_enabled:
#         domain_dict["variables"]["X"]["m_value"] = {
#             "1": X_values["0"],
#             "0": X_values["1"],
#         }
#         domain_dict["variables"]["Y"]["m_value"] = {
#             "1": Y_values["0"],
#             "0": Y_values["1"],
#         }
#         domain_dict["variables"]["Z"]["m_value"] = {
#             "1": Y_values["0"],
#             "0": Y_values["1"],
#         }

#     # Enforce the zero label **after** counterbalancing to preserve logic
#     # if enforce_zero_label:
#     #     domain_dict["variables"]["A"]["p_value"]["0"] = zero_label
#     #     # domain_dict["variables"]["A"]["m_value"]["1"] = zero_label
#     #     domain_dict["variables"]["B"]["p_value"]["0"] = zero_label
#     #     # domain_dict["variables"]["B"]["m_value"]["1"] = zero_label
#     #     domain_dict["variables"]["C"]["p_value"]["0"] = zero_label
#     #     # domain_dict["variables"]["C"]["m_value"]["1"] = zero_label

#     if enforce_zero_label:
#         for var in ["X", "Y", "Z"]:
#             domain_dict["variables"][var]["p_value"]["0"] = zero_label
#             domain_dict["variables"][var]["m_value"]["0"] = (
#                 zero_label  # Both p and m should have "0" as normal
#             )

#     return domain_dict


def create_domain_dict(
    domain_name, introduction, variables_config, graph_type="collider"
):
    """
    Create a domain dictionary with full support for explanations and counterbalance conditions.

    Parameters:
    -----------
    domain_name : str
        Domain name (e.g., "economy", "sociology")

    introduction : str
        Domain introduction text

    variables_config : dict
        Complete configuration for all variables (X, Y, Z):
        {
            "X": {
                "name": "interest rates",
                "detailed": "Interest rates are the rates banks charge...",
                "p_value": {"1": "low", "0": "normal"},
                "m_value": {"1": "high", "0": "normal"},
                "explanations": {
                    "p_p": "Low interest rates stimulate economic growth...",
                    "p_m": "The good economic times produced by...",
                    "m_p": "The high interest rates result in high yields...",
                    "m_m": "A lot of people are making large monthly interest..."
                }
            },
            "Y": {...},
            "Z": {...}
        }

    graph_type : str
        Type of causal graph (collider, fork, chain)

    Returns:
    --------
    dict
        Domain dictionary in the required format
    """
    # Start with basic structure
    domain_dict = {
        "domain_name": domain_name,
        "introduction": introduction,
        "variables": {},
        "graph_type": graph_type,
    }

    # Process each variable
    for var_key, config in variables_config.items():
        # Validate required fields
        required_fields = ["name", "detailed", "p_value"]
        for field in required_fields:
            if field not in config:
                raise ValueError(
                    f"Missing required field '{field}' for variable {var_key}"
                )

        # Create variable entry
        var_entry = {
            f"{var_key}_name": config["name"],
            f"{var_key}_detailed": config["detailed"],
            "p_value": config["p_value"].copy(),
        }

        # Add m_value if provided, otherwise use opposite of p_value
        if "m_value" in config:
            var_entry["m_value"] = config["m_value"].copy()
        else:
            # Default behavior: swap 0/1 values from p_value
            var_entry["m_value"] = {
                "1": config["p_value"]["0"],
                "0": config["p_value"]["1"],
            }

        # Add explanations if provided
        if "explanations" in config:
            var_entry["explanations"] = config["explanations"].copy()

        # Add to domain_dict
        domain_dict["variables"][var_key] = var_entry

    return domain_dict
