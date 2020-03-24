import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from itertools import chain
from utilities.common_utils import *
from utilities.text_utils import *
from utilities.dataset_constants import *
from functools import reduce
import argparse

def append_all_titles(df, main_title, alternate_titles=None):
    """
    Takes all the titles of each job, cleans them, and appends them together.
    :param df: The job dataframe
    :param main_title: Main title column. Cannot be None.
    :param alternate_titles: Alternative title column. Can be None, in which case only the Main title is handled.
    :return: A copy of the dataframe with a new column, appended_titles.
    """
    df = df.copy()
    # TODO: More advanced stuff involving the generation of alternatives using synonyms
    df['appended_titles'] = df.apply(lambda x:
                ([' '.join(tokenise_stem_punkt_and_stopword(x[main_title], stopword_set=None))]) +
                ([' '.join(tokenise_stem_punkt_and_stopword(y, stopword_set=None))
                 for y in x[alternate_titles].split('\n')] if alternate_titles is not None and
                                     isinstance(x[alternate_titles], str) else []),
                                    axis=1)
    return df

def explode_appended_titles(df, id_col):
    """

    :param df: Dataframe output from append_all_titles
    :param id_col: The id column of the dataframe, which will be one of the two columns of the final result.
    :return: A dataframe with two columns: id_col and 'title_simple'.
    """
    return df.set_index(id_col)['appended_titles'].apply(pd.Series).stack().\
                    reset_index(level=0).rename(columns={0:'title_simple'})

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

def find_exact_matches(dfs, main_titles, alt_titles, id_cols):
    """
    Given a list of dataframes, matches the job titles together directly (using both main and alt titles).
    :param dfs: List of dataframes
    :param main_titles: List of main title column names
    :param alt_titles: List of alt column names. Can contain Nones.
    :param id_cols: List of id column names.
    :return: A dataframe with all the exact match jobs matched together, and a list of dictionaries binding their
    ids together using the numerical index of the df.
    """
    modified_title_dfs = [append_all_titles(dfs[i], main_titles[i], alt_titles[i]) for i in range(len(dfs))]
    exploded_title_dfs = [explode_appended_titles(modified_title_dfs[i], id_cols[i]) for i in range(len(dfs))]
    all_exact_match_jobs = reduce(lambda left,right: pd.merge(left,right,on='title_simple'), exploded_title_dfs)
    # TODO: Cleaning phase for the titles that shouldn't really have been matched together
    for i in range(len(dfs)):
        all_exact_match_jobs = pd.merge(all_exact_match_jobs, dfs[i], on=id_cols[i])
    return all_exact_match_jobs, make_indexed_columns(all_exact_match_jobs,
                                                        id_cols+['title_simple'], None)

def get_skills_to_investigate_direct_match(skills_and_relations, id_cols, job_titles):
    """
    Takes matched job titles from multiple datasets, and retrieves their respective skills and returns
    :param skills_and_relations: A list of dataframes, each being skill-job relations for one dataset.
    :param id_cols: A list of id column names, one for each dataset.
    :param job_titles: A list of dictionaries, mapping indices to id column values for each dataset,
        with its last element mapping indices to simplified matching titles.
    :return:
    """

    skills_to_investigate = [skills_and_relations[i].loc[
                        skills_and_relations[i][id_cols[i]].apply(lambda x: x in job_titles[i].values())].copy()
                        for i in range(len(skills_and_relations))]
    result_list = list()
    for i in range(len(skills_to_investigate)):
        df = skills_to_investigate[i]
        job_title_keys = list(job_titles[i])
        id_df = pd.DataFrame({'common_id': job_title_keys, id_cols[i]: [job_titles[i][x] for x in job_title_keys]})
        common_key_df = pd.DataFrame(
                        {'common_id': job_title_keys,
                         'common_key': [job_titles[len(job_titles)-1][x] for x in job_title_keys]})
        all_ids_df = pd.merge(id_df, common_key_df)
        df = pd.merge(df, all_ids_df)
        result_list.append(df)
    return result_list

def calculate_tfidf_for_col(dfs_and_colnames, do_stem=True, min_df=1,
                            count_vec=False, return_sum_all=False, dense=False):
    """
    Calculates TF-IDF or word count vectors for each row in a column.
    :param dfs_and_colnames: List of tuples; each tuple is (dataframe, colname).
    :param do_stem: Whether to do stemming on the contents of the column or not.
    :param min_df: Minimum doc frequency of a word, words with lower df are eliminated
    :param count_vec: Whether to count vectorise, or calculate TF-IDF (default).
    :param return_sum_all: Whether to also return a single vector which is the sum of the whole column's vectors.
    :param dense: Whether to return dense or sparse results.
    :return: A list of either numpy 1d arrays or single row sparse matrices, optionally plus the vector that is the
            sum of them all.
    """
    cleaned_text = [df[col_name].apply(lambda x: ' '.join(tokenise_stem_punkt_and_stopword(x, do_stem=do_stem))).values
                    for df, col_name in dfs_and_colnames]
    if count_vec:
        vec_model = CountVectorizer(tokenizer=lambda x: x.split(' '), min_df=min_df)
    else:
        vec_model = TfidfVectorizer(tokenizer=lambda x: x.split(' '), min_df=min_df)
    vec_model.fit(list(chain.from_iterable(cleaned_text)))
    vec_matrices = [vec_model.transform(text) for text in cleaned_text]
    if not return_sum_all:
        if dense:
            return [[np.array(vec_matrix[i,:].todense()).flatten()
                     for i in range(vec_matrix.shape[0])] for vec_matrix in vec_matrices], vec_model
        else:
            return [[vec_matrix[i,:] for i in range(vec_matrix.shape[0])] for vec_matrix in vec_matrices], vec_model
    else:
        if dense:
            return [[np.array(vec_matrix[i,:].todense()).flatten()
                for i in range(vec_matrix.shape[0])] for vec_matrix in vec_matrices], vec_model, \
                sum([vec_matrix.sum(axis=0) for vec_matrix in vec_matrices])
        else:
            return [[vec_matrix[i,:] for i in range(vec_matrix.shape[0])] for vec_matrix in vec_matrices], vec_model, \
                sum([vec_matrix.sum(axis=0) for vec_matrix in vec_matrices])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='Root dir containing dataframes.')
    parser.add_argument('--jobs_filenames', type=str, required=True, help=
                                            'Names of pickled job dataframe files, comma separated.')
    parser.add_argument('--skills_filenames', type=str, required=True, help=
                                            'Names of pickled skills and relations dataframe files, comma separated.')
    parser.add_argument('--datasets', type=str, required=True, help=
                                                    'Names of datasets, comma separated, '
                                                            'same order as filenames. Options are esco and onet.')
    parser.add_argument('--countvec', action='store_true')
    parser.add_argument('--tfidf', action='store_true')
    args = parser.parse_args()

    if args.countvec and args.tfidf:
        print('Only one of the two options (TF-IDF and CountVec) is possible!')
        return

    output_dir = os.path.join(args.root_dir, 'matched')

    job_df_names = [os.path.join(args.root_dir, fn) for fn in args.jobs_filenames.split(',')]
    skill_df_names = [os.path.join(args.root_dir, fn) for fn in args.skills_filenames.split(',')]
    job_dfs = [pickle.load(open(df_name, 'rb')) for df_name in job_df_names]
    skill_dfs = [pickle.load(open(df_name, 'rb')) for df_name in skill_df_names]
    dataset_names = args.datasets.split(',')
    all_exact_match_jobs, job_titles_index = find_exact_matches(job_dfs,
                                                                [main_titles[x] for x in dataset_names],
                                                                [alternative_titles[x] for x in dataset_names],
                                                                [job_id_cols[x] for x in dataset_names])

    skills_to_investigate = get_skills_to_investigate_direct_match(skill_dfs,
                                                           [skill_relations_id_cols[x] for x in dataset_names],
                                                           job_titles_index)

    if args.countvec:
        vec_list, model = calculate_tfidf_for_col([(skills_to_investigate[i], skill_labels[dataset_names[i]])
                                                   for i in range(len(skills_to_investigate))], do_stem=True,
                                                   count_vec=True, return_sum_all=False, dense=False)
    elif args.tfidf:
        vec_list, model = calculate_tfidf_for_col([(skills_to_investigate[i], skill_labels[dataset_names[i]])
                                                   for i in range(len(skills_to_investigate))], do_stem=True,
                                                   count_vec=False, return_sum_all=False, dense=False)
    else:
        vec_list = None
        model = None

    for i in range(len(skills_to_investigate)):
        with open(os.path.join(output_dir, dataset_names[i]+'_skills.pkl'), 'wb') as f:
            pickle.dump(skills_to_investigate[i], f)
    with open(os.path.join(output_dir, 'matches_df.pkl'), 'wb') as f:
        pickle.dump(all_exact_match_jobs, f)
    if vec_list is not None:
        with open(os.path.join(output_dir, 'vec_list.pkl'), 'wb') as f:
            pickle.dump(vec_list, f)
        with open(os.path.join(output_dir, 'tfidf_model.pkl'), 'wb') as f:
            pickle.dump(model, f)

if __name__ == '__main__':
    main()