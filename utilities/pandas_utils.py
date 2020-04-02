import pandas as pd
from functools import reduce
from scipy.sparse import csr_matrix
import numpy as np
from utilities.common_utils import *

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

def find_top_words(df, colname, tfidf_model, top_n=10, lower_bound=0, modify_df=True):
    vocab = invert_dict(tfidf_model.vocabulary_)
    if isinstance(df.iloc[0][colname], csr_matrix):
        df[colname] = df[colname].apply(lambda x: np.array(x.todense()).flatten())
    if lower_bound is None:
        new_col = df[colname].apply(lambda x: ([vocab[y] for y in
                            np.argsort(x)[-top_n:]])[::-1])
    else:
        new_col = df[colname].apply(lambda x: ([vocab[y] for y in
                            np.argsort(x)[-top_n:] if x[y] > lower_bound])[::-1])
    if modify_df:
        df['top_words_'+colname.split('_')[1]] = new_col
        return df
    else:
        return new_col

def aggregate_top_words(df, colname, col_to_aggregate_on, tfidf_model, top_n=10, raw_tf_idf=False):
    if not raw_tf_idf:
        return find_top_words(df[[col_to_aggregate_on, colname]].\
                groupby(col_to_aggregate_on).sum(),
                colname, tfidf_model, top_n=top_n)[['top_words_'+colname.split('_')[1]]]
    else:
        return df[[col_to_aggregate_on, colname]].\
                groupby(col_to_aggregate_on).sum()

def reduce_sum(series):
    return reduce(lambda x, y: x + y, series)