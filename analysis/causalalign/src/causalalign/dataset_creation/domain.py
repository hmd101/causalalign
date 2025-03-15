def create_domain_dict(
    domain,
    introduction,
    c1_name,
    c1_detailed,
    c1_values,
    c2_name,
    c2_detailed,
    c2_values,
    e_name,
    e_detailed,
    e_values,
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
    c1_name, c2_name, e_name : str
        Names of variables C1, C2, and E
    c1_detailed, c2_detailed, e_detailed : str
        Detailed descriptions of variables
    c1_values, c2_values, e_values : dict
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
                    "c1_name": c1_name,
                    "c1_detailed": c1_detailed,
                    "p_value": c1_values,
                },
                "C2": {
                    "c2_name": c2_name,
                    "c2_detailed": c2_detailed,
                    "p_value": c2_values,
                },
                "E": {
                    "e_name": e_name,
                    "e_detailed": e_detailed,
                    "p_value": e_values,
                },
            },
        }
    }

    # Handle counterbalance vs. simple value assignment
    if counterbalance_enabled:
        # Add m_values by inverting the p_values
        domain_dict["domain"]["variables"]["C1"]["m_value"] = {
            "1": c1_values["0"],
            "0": c1_values["1"],
        }
        domain_dict["domain"]["variables"]["C2"]["m_value"] = {
            "1": c2_values["0"],
            "0": c2_values["1"],
        }
        domain_dict["domain"]["variables"]["E"]["m_value"] = {
            "1": e_values["0"],
            "0": e_values["1"],
        }

    return domain_dict
