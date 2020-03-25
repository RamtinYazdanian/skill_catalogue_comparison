import pandas as pd

def make_two_cols_map(df, col_key, col_val):
    """
    Takes a dataframe and two of its columns, returns a dictionary mapping the keys to the values.
    :param df: The dataframe.
    :param col_key: The column containing the keys. Should have unique values.
    :param col_val: The column containing the values.
    :return: A dictionary.
    """
    return {df[col_key].values[i]:df[col_val].values[i] for i in range(df.shape[0])}


def make_indexed_columns(df, col_names, reverse=None):
    """
    Given a dataframe where multiple non-unique columns need to be bound together, uses the numerical index
    as a uniting id.
    :param df: The dataframe
    :param col_names: List of column names to be bound together
    :param reverse: Whether the mapping should be from column values to the index. Either a list or None.
    :return: A list of dictionaries, each mapping the index to one of the columns.
    """
    result_list = list()
    df = df.reset_index()
    for i in range(len(col_names)):
        colname = col_names[i]
        if reverse is None or reverse[i]:
            result_list.append(make_two_cols_map(df, 'index', colname))
        else:
            result_list.append(make_two_cols_map(df, colname, 'index'))
    return result_list