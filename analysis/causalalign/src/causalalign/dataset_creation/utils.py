import pandas as pd


# Append the dataframes together
def append_dfs(*dfs):
    """
    Append multiple dataframes together after checking that they have the same columns and dtypes.

    Parameters:
    -----------
    *dfs : list of pd.DataFrame
        DataFrames to be appended.

    Returns:
    --------
    pd.DataFrame
        A single DataFrame resulting from appending the input DataFrames.
    """
    # Check that all dataframes have the same columns and dtypes
    columns = dfs[0].columns
    dtypes = dfs[0].dtypes

    for df in dfs[1:]:
        assert df.columns.equals(columns), "DataFrames have different columns"
        assert df.dtypes.equals(dtypes), "DataFrames have different dtypes"

    # Concatenate the dataframes
    all_domains_df = pd.concat(dfs, ignore_index=True)
    return all_domains_df
