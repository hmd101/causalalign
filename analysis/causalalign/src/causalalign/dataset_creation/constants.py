# Contains default domain components used in rehder & Waldmann, 2017 (RW17) and graph structures


rw_17_domain_components = {
    "economy": {
        "domain_name": "economy",
        "variables": {
            "C1": {
                "name": "interest rates",
                "detailed": "Interest rates are the rates banks charge to loan money.",
                "p_value": {"1": "low", "0": "high"},
                "m_value": {"1": "high", "0": "low"},
                "explanations": {
                    "p_p": "Low interest rates stimulate economic growth, leading to greater prosperity overall, and allowing more money to be saved for retirement in particular.",
                    "p_m": "The good economic times produced by the low interest rates leads to greater confidence and less worry about the future, so people are less concerned about retirement.",
                    "m_p": "The high interest rates result in high yields on government bonds, which are especially attractive for retirement savings because they are such a safe investment.",
                    "m_m": "A lot of people are making large monthly interest payments on credit card debt, and they have no money left to save for retirement.",
                },
            },
            "C2": {
                "name": "trade deficits",
                "detailed": "A country's trade deficit is the difference between the value of the goods that a country imports and the value of the goods that a country exports.",
                "p_value": {"1": "small", "0": "large"},
                "m_value": {"1": "large", "0": "small"},
                "explanations": {
                    "p_p": "When the economy is good, people can cover their basic expenses and so have enough money left over to contribute to their retirement accounts.",
                    "p_m": "When the economy is good, people are optimistic and so spend rather than save.",
                    "m_p": "People become nervous when their economy is no longer competitive enough in the world economy to export products, and begin saving for retirement as a result.",
                    "m_m": "The loss of local manufacturing jobs means that there are people out of work, and contributions to retirement accounts decreases.",
                },
            },
            "E": {
                "name": "retirement savings",
                "detailed": "Retirement savings is the money people save for their retirement.",
                "p_value": {"1": "high", "0": "low"},
                "m_value": {"1": "low", "0": "high"},
            },
        },
        "introduction": "Economists seek to describe and predict the regular patterns of economic fluctuation. To do this, they study some important variables or attributes of economies. They also study how these attributes are responsible for producing or causing one another.",
    },
    "sociology": {
        "name": "sociology",
        "variables": {
            "C1": {
                "name": "urbanization",
                "detailed": "Urbanization is the degree to which the members of a society live in urban environments (i.e., cities) versus rural environments.",
                "p_value": {"1": "high", "0": "low"},
                "m_value": {"1": "low", "0": "high"},
                "explanations": {
                    "p_p": "Big cities provide many opportunities for financial and social improvement.",
                    "p_m": "In big cities many people are competing for the same high-status jobs and occupations.",
                    "m_p": 'People in rural areas are rarely career oriented, and so take time off from working and switch frequently between different "temp" jobs.',
                    "m_m": "The low density of people prevents the dynamic economic expansion needed for people to get ahead.",
                },
            },
            "C2": {
                "name": "interest in religion",
                "detailed": "Interest in religion is the degree to which the members of a society show a curiosity in religion issues or participate in organized religions.",
                "p_value": {"1": "low", "0": "high"},
                "m_value": {"1": "high", "0": "low"},
                "explanations": {
                    "p_p": "Without the restraint of religion-based morality, the impulse toward greed dominates and people tend to accumulate material wealth.",
                    "p_m": "Many religions reinforce a strong work ethic; without this motivation, workers become complacent at their jobs.",
                    "m_p": "Religion fosters communal care, and those of the same religion tend to support each other with jobs, financial favors, and so on.",
                    "m_m": "The spiritualism induced by religion works to reduce the desire for material wealth.",
                },
            },
            "E": {
                "name": "socio-economic mobility",
                "detailed": "Socioeconomic mobility is the degree to which the members of a society are able to improve their social and economic status.",
                "p_value": {"1": "high", "0": "low"},
                "m_value": {"1": "low", "0": "high"},
            },
        },
        "introduction": "Sociologists seek to describe and predict the regular patterns of societal interactions. To do this, they study some important variables or attributes of societies. They also study how these attributes are responsible for producing or causing one another.",
    },
    "weather": {
        "name": "weather",
        "variables": {
            "C1": {
                "name": "ozone levels",
                "detailed": "Ozone is a gaseous allotrope of oxygen (O3) and is formed by exposure to UV radiation.",
                "p_value": {"1": "high", "0": "low"},
                "m_value": {"1": "low", "0": "high"},
                # Note: this is graph dependent. the explanations below are based on the collider graph
                "explanations": {
                    "p_p": "Ozone attracts extra oxygen atoms from water molecules, creating a concentration of water vapor in that region.",
                    "p_m": "Ozone accepts extra oxygen atoms, decreasing the amount of oxygen available to form water molecules. With fewer water molecules, there is lower humidity.",
                    "m_p": "The oxygen atoms that would normally be part of ozone molecules are free to combine with hydrogen atoms instead, creating water molecules.",
                    "m_m": "The low amount of ozone allows a large number of ultra-violet (UV) rays to enter the atmosphere, and the UV rays break up water molecules, resulting in low humidity.",
                },
            },
            "C2": {
                "name": "air pressure",
                "detailed": "Air pressure is force exerted due to concentrations of air molecules.",
                "p_value": {"1": "low", "0": "high"},
                "m_value": {"1": "high", "0": "low"},
                # Note: this is graph dependent. the explanations below are based on the collider graph
                "explanations": {
                    "p_p": "When pressure does not force water vapor to break into oxygen and hydrogen atoms, water vapor remains in abundance.",
                    "p_m": "Low air pressure poorly facilitates condensation; as a result, there are less water molecules in the air.",
                    "m_p": "The higher pressure means that the components of water molecules (hydrogen and oxygen) tend to not dissocciate from one another. Because there are more water molecules, humidity is higher.",
                    "m_m": "When air pressure is high, water vapor condenses into liquid water (rain), and the atmosphere is left with little moisture.",
                },
            },
            "E": {
                "name": "humidity",
                "detailed": "Humidity is the degree to which the atmosphere contains water molecules.",
                "p_value": {"1": "high", "0": "low"},
                "m_value": {"1": "low", "0": "high"},
            },
        },
        "introduction": "Meteorologists seek to describe and predict the regular patterns that govern weather systems. To do this, they study some important variables or attributes of weather systems. They also study how these attributes are responsible for producing or causing one another.",
    },
}


#  graph structures (currently only collider in dataset, but expandable), RW17 used collider and fork.
graph_structures = {
    "collider": {
        "description": "C1→E←C2",
        "causal_template": "{c1_sense} {c1_name} cause {e_sense} {e_name}. Also, {c2_sense} {c2_name} cause {e_sense} {e_name}.",
    },
    "fork": {
        "description": "C1←E→C2",
        "causal_template": "{e_sense} {e_name} causes {c1_sense} {c1_name}. Also, {e_sense} {e_name} causes {c2_sense} {c2_name}.",
    },
    "chain": {
        "description": "C1→C2→E",
        "causal_template": "{c1_sense} {c1_name} causes {c2_sense} {c2_name}. Also, {c2_sense} {c2_name} causes {e_sense} {e_name}.",
    },
}


# Add inference task mapping to the structure
inference_tasks_rw17 = {
    "a": {
        "query_node": "Ci=1",
        "observation": "E=1, Cj=1",
        "query": "p(Ci=1|E=1, Cj=1)",
    },
    "b": {"query_node": "Ci=1", "observation": "E=1", "query": "p(Ci=1|E=1)"},
    "c": {
        "query_node": "Ci=1",
        "observation": "E=1, Cj=0",
        "query": "p(Ci=1|E=1, Cj=0)",
    },
    "d": {"query_node": "Ci=1", "observation": "Cj=1", "query": "p(Ci=1|Cj=1)"},
    "e": {"query_node": "Ci=1", "observation": "Cj=0", "query": "p(Ci=1|Cj=0)"},
    "f": {
        "query_node": "Ci=1",
        "observation": "E=0, Cj=1",
        "query": "p(Ci=1|E=0, Cj=1)",
    },
    "g": {"query_node": "Ci=1", "observation": "E=0", "query": "p(Ci=1|E=0)"},
    "h": {
        "query_node": "Ci=1",
        "observation": "E=0, Cj=0",
        "query": "p(Ci=1|E=0, Cj=0)",
    },
    "i": {
        "query_node": "E=1",
        "observation": "Ci=0, Cj=0",
        "query": "p(E=1|Ci=0, Cj=0)",
    },
    "j": {
        "query_node": "E=1",
        "observation": "Ci=0, Cj=1",
        "query": "p(E=1|Ci=0, Cj=1)",
    },
    "k": {
        "query_node": "E=1",
        "observation": "Ci=1, Cj=1",
        "query": "p(E=1|Ci=1, Cj=1)",
    },
}
