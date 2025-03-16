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
):
    """
    Create a domain dictionary with minimal required inputs.

    Parameters:
    -----------
    name : str
        Domain name (e.g., "economy", "biology")
    introduction : str
        Domain introduction text
    C1_name, C2_name, E_name : str
        Names of variables C1, C2, and E
    C1_detailed, C2_detailed, E_detailed : str
        Detailed descriptions of variables
    C1_values, C2_values, E_values : dict
        Mapping of "1" and "0" to their verbal descriptions
        e.g., {"1": "high", "0": "low"}
    counterbalance_enabled : bool
        Whether to enable counterbalancing (p/m conditions)

    Returns:
    --------
    dict
        Domain dictionary in the required format
    """
    # Build the domain dictionary
    domain_dict = {
        "domain": {
            "domain_name": domain,
            "introduction": introduction,
            "variables": {
                "C1": {
                    "C1_name": C1_name,
                    "C1_detailed": C1_detailed,
                    "p_value": C1_values,
                },
                "C2": {
                    "C2_name": C2_name,
                    "C2_detailed": C2_detailed,
                    "p_value": C2_values,
                },
                "E": {
                    "E_name": E_name,
                    "E_detailed": E_detailed,
                    "p_value": E_values,
                },
            },
        }
    }

    # Handle counterbalance vs. simple value assignment
    if counterbalance_enabled:
        # Add m_values by inverting the p_values
        domain_dict["domain"]["variables"]["C1"]["m_value"] = {
            "1": C1_values["0"],
            "0": C1_values["1"],
        }
        domain_dict["domain"]["variables"]["C2"]["m_value"] = {
            "1": C2_values["0"],
            "0": C2_values["1"],
        }
        domain_dict["domain"]["variables"]["E"]["m_value"] = {
            "1": E_values["0"],
            "0": E_values["1"],
        }

    return domain_dict
