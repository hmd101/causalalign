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
# ):
#     """
#     Create a domain dictionary with minimal required inputs.

#     Parameters:
#     -----------
#     name : str
#         Domain name (e.g., "economy", "biology")
#     introduction : str
#         Domain introduction text
#     C1_name, C2_name, E_name : str
#         Names of variables C1, C2, and E
#     C1_detailed, C2_detailed, E_detailed : str
#         Detailed descriptions of variables
#     C1_values, C2_values, E_values : dict
#         Mapping of "1" and "0" to their verbal descriptions
#         e.g., {"1": "high", "0": "low"}
#     counterbalance_enabled : bool
#         Whether to enable counterbalancing (p/m conditions)

#     Returns:
#     --------
#     dict
#         Domain dictionary in the required format
#     """
#     # Build the domain dictionary
#     domain_dict = {
#         # "domain": {
#         "domain_name": domain,
#         "introduction": introduction,
#         "variables": {
#             "C1": {
#                 "C1_name": C1_name,
#                 "C1_detailed": C1_detailed,
#                 "p_value": C1_values,
#             },
#             "C2": {
#                 "C2_name": C2_name,
#                 "C2_detailed": C2_detailed,
#                 "p_value": C2_values,
#             },
#             "E": {
#                 "E_name": E_name,
#                 "E_detailed": E_detailed,
#                 "p_value": E_values,
#             },
#         },
#     }
#     # }

#     # Handle counterbalance vs. simple value assignment
#     if counterbalance_enabled:
#         # Add m_values by inverting the p_values
#         # domain_dict["domain"]["variables"]["C1"]["m_value"] = {
#         domain_dict["variables"]["C1"]["m_value"] = {
#             "1": C1_values["0"],
#             "0": C1_values["1"],
#         }
#         # domain_dict["domain"]["variables"]["C2"]["m_value"] = {
#         domain_dict["variables"]["C2"]["m_value"] = {
#             "1": C2_values["0"],
#             "0": C2_values["1"],
#         }
#         # domain_dict["domain"]["variables"]["E"]["m_value"] = {
#         domain_dict["variables"]["E"]["m_value"] = {
#             "1": E_values["0"],
#             "0": E_values["1"],
#         }

#     return domain_dict


##########
def create_domain_dict(
    domain,
    introduction,
    C1_name,
    C1_detailed,
    C1_values,
    C2_name,
    C2_detailed,
    C2_values,
    E_name,
    E_detailed,
    E_values,
    counterbalance_enabled=False,
    enforce_zero_label=False,
    zero_label="normal",  # Default to 'normal' but configurable
):
    """
    Create a domain dictionary with a configurable zero-value verbalization.

    Parameters:
    -----------
    domain : str
        Domain name (e.g., "economy", "biology").
    introduction : str
        Domain introduction text.
    C1_name, C2_name, E_name : str
        Names of variables C1, C2, and E.
    C1_detailed, C2_detailed, E_detailed : str
        Detailed descriptions of variables.
    C1_values, C2_values, E_values : dict
        Mapping of "1" and "0" to their verbal descriptions.
        e.g., {"1": "high", "0": "low"}.
    counterbalance_enabled : bool
        Whether to enable counterbalancing (p/m conditions).
    enforce_zero_label : bool
        Whether to enforce the specified zero_label on the '0' value.
    zero_label : str
        The verbalization to be assigned to the "0" value across all variables.

    Returns:
    --------
    dict
        Domain dictionary in the required format.
    """

    def validate_values(values):
        """Ensure that required keys exist in the value dictionary."""
        if not isinstance(values, dict):
            raise ValueError(
                "Expected a dictionary for variable values, but received a different type."
            )
        if "0" not in values:
            raise KeyError(
                "Missing '0' key in values dictionary. Ensure '0' is explicitly defined."
            )
        if "1" not in values:
            raise KeyError(
                "Missing '1' key in values dictionary. Ensure '1' is explicitly defined."
            )
        return values

    # Validate input value dictionaries
    C1_values = validate_values(C1_values)
    C2_values = validate_values(C2_values)
    E_values = validate_values(E_values)

    # Build the domain dictionary
    domain_dict = {
        "domain_name": domain,
        "introduction": introduction,
        "variables": {
            "C1": {
                "C1_name": C1_name,
                "C1_detailed": C1_detailed,
                "p_value": C1_values.copy(),
            },
            "C2": {
                "C2_name": C2_name,
                "C2_detailed": C2_detailed,
                "p_value": C2_values.copy(),
            },
            "E": {
                "E_name": E_name,
                "E_detailed": E_detailed,
                "p_value": E_values.copy(),
            },
        },
    }

    # Handle counterbalance vs. simple value assignment
    if counterbalance_enabled:
        domain_dict["variables"]["C1"]["m_value"] = {
            "1": C1_values["0"],
            "0": C1_values["1"],
        }
        domain_dict["variables"]["C2"]["m_value"] = {
            "1": C2_values["0"],
            "0": C2_values["1"],
        }
        domain_dict["variables"]["E"]["m_value"] = {
            "1": E_values["0"],
            "0": E_values["1"],
        }

    # Enforce the zero label **after** counterbalancing to preserve logic
    # if enforce_zero_label:
    #     domain_dict["variables"]["C1"]["p_value"]["0"] = zero_label
    #     # domain_dict["variables"]["C1"]["m_value"]["1"] = zero_label
    #     domain_dict["variables"]["C2"]["p_value"]["0"] = zero_label
    #     # domain_dict["variables"]["C2"]["m_value"]["1"] = zero_label
    #     domain_dict["variables"]["E"]["p_value"]["0"] = zero_label
    #     # domain_dict["variables"]["E"]["m_value"]["1"] = zero_label

    if enforce_zero_label:
        for var in ["C1", "C2", "E"]:
            domain_dict["variables"][var]["p_value"]["0"] = zero_label
            domain_dict["variables"][var]["m_value"]["0"] = (
                zero_label  # Both p and m should have "0" as normal
            )

    return domain_dict


#############

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
#             raise ValueError("Expected a dictionary for variable values, but received a different type.")
#         if "0" not in values:
#             raise KeyError("Missing '0' key in values dictionary. Ensure '0' is explicitly defined.")
#         if "1" not in values:
#             raise KeyError("Missing '1' key in values dictionary. Ensure '1' is explicitly defined.")
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
#             "1": C1_values["1"],  # Keep 1 value same
#             "0": C1_values["0"],  # Keep 0 value same initially
#         }
#         domain_dict["variables"]["C2"]["m_value"] = {
#             "1": C2_values["1"],
#             "0": C2_values["0"],
#         }
#         domain_dict["variables"]["E"]["m_value"] = {
#             "1": E_values["1"],
#             "0": E_values["0"],
#         }

#     # Enforce the zero label **after** counterbalancing to preserve logic
#     if enforce_zero_label:
#         for var in ["C1", "C2", "E"]:
#             domain_dict["variables"][var]["p_value"]["0"] = zero_label
#             domain_dict["variables"][var]["m_value"]["0"] = zero_label  # Both p and m should have "0" as normal

#     return domain_dict
