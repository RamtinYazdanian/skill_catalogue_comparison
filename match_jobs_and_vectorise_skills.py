import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from itertools import chain
from utilities.common_utils import *
from utilities.pandas_utils import make_indexed_columns
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
    if alt_titles is not None:
        modified_title_dfs = [append_all_titles(dfs[i], main_titles[i], alt_titles[i]) for i in range(len(dfs))]
    else:
        modified_title_dfs = [append_all_titles(dfs[i], main_titles[i], None) for i in range(len(dfs))]
    exploded_title_dfs = [explode_appended_titles(modified_title_dfs[i], id_cols[i]) for i in range(len(dfs))]
    all_exact_match_jobs = reduce(lambda left,right: pd.merge(left,right,on='title_simple'), exploded_title_dfs)
    # TODO: Cleaning phase for the titles that shouldn't really have been matched together
    for i in range(len(dfs)):
        all_exact_match_jobs = pd.merge(all_exact_match_jobs, dfs[i], on=id_cols[i])
    return all_exact_match_jobs, make_indexed_columns(all_exact_match_jobs,
                                                      id_cols + ['title_simple'], None)

def find_closest_matches(dfs, main_titles, alt_titles, w2v_model, id_cols, top_n=1):
    df_word_vectors = list()
    for i in range(len(dfs)):
        if alt_titles is None:
            df_word_vectors.append(get_df_word_vectors(dfs[i], [main_titles[i]], w2v_model))
        elif alt_titles[i] is None:
            df_word_vectors.append(get_df_word_vectors(dfs[i], [main_titles[i]], w2v_model))
        else:
            df_word_vectors.append(get_df_word_vectors(dfs[i], [main_titles[i], alt_titles[i]], w2v_model))
        print(len(df_word_vectors[i]))

    similarities = get_pairwise_similarities(df_word_vectors[0], df_word_vectors[1])
    if top_n == 1:
        matched_indices = np.argmax(similarities, axis=1)
        print(len(matched_indices))
        all_matched_jobs = pd.DataFrame({id_cols[0] + '_ind': list(range(len(matched_indices))),
                                         id_cols[1] + '_ind': matched_indices})
    else:
        matched_indices = np.argsort(similarities, axis=1).tolist()
        matched_indices = [x[-top_n:] for x in matched_indices]
        all_matched_jobs = pd.DataFrame({id_cols[0] + '_ind': list(range(len(matched_indices)*top_n)),
                                         id_cols[1] + '_ind': list(chain.from_iterable(matched_indices))})

    for i in range(len(dfs)):
        all_matched_jobs = pd.merge(all_matched_jobs, dfs[i], right_index=True, left_on=id_cols[i]+'_ind')
    all_matched_jobs = all_matched_jobs.drop(columns=[id_col+'_ind' for id_col in id_cols])
    print(all_matched_jobs.shape)
    return all_matched_jobs, make_indexed_columns(all_matched_jobs,
                                                      id_cols + [id_cols[0]], None)


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
                            count_vec=False, return_sum_all=False, dense=False, ngrams=1):
    """
    Calculates TF-IDF or word count vectors for each row in a column.
    :param dfs_and_colnames: List of tuples; each tuple is (dataframe, colname).
    :param do_stem: Whether to do stemming on the contents of the column or not.
    :param min_df: Minimum doc frequency of a word, words with lower df are eliminated
    :param count_vec: Whether to count vectorise, or calculate TF-IDF (default).
    :param return_sum_all: Whether to also return a single vector which is the sum of the whole column's vectors.
    :param dense: Whether to return dense or sparse results.
    :return: A list of n (n being the number of dfs) lists of either numpy 1d arrays or single row sparse matrices,
            optionally plus the vector that is the sum of them all.
    """
    cleaned_text = [df[col_name].apply(lambda x: ' '.join(tokenise_stem_punkt_and_stopword(x, do_stem=do_stem))).values
                    for df, col_name in dfs_and_colnames]
    if count_vec:
        vec_model = CountVectorizer(min_df=min_df, ngram_range=(1, ngrams))
    else:
        vec_model = TfidfVectorizer(min_df=min_df, ngram_range=(1, ngrams))
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
    parser.add_argument('--output_dir', type=str, required=True, help='Output dir.')
    parser.add_argument('--jobs_filenames', type=str, required=True, help=
                                            'Names of pickled job dataframe files, comma separated.')
    parser.add_argument('--skills_filenames', type=str, required=True, help=
                                            'Names of pickled skills and relations dataframe files, comma separated.')
    parser.add_argument('--datasets', type=str, required=True, help=
                                                    'Names of datasets, comma separated, '
                                                            'same order as filenames. '
                                                            'Options are ESCO, ONET, and SWISS.')
    parser.add_argument('--countvec', action='store_true')
    parser.add_argument('--tfidf', action='store_true')
    parser.add_argument('--no_alt_titles', action='store_true')
    parser.add_argument('--w2v', type=str, default=None)
    parser.add_argument('--ngrams', type=int, default=1)
    args = parser.parse_args()

    if args.countvec and args.tfidf:
        print('Only one of the two options (TF-IDF and CountVec) is possible!')
        return

    output_dir = args.output_dir

    job_df_names = [os.path.join(args.root_dir, fn) for fn in args.jobs_filenames.split(',')]
    skill_df_names = [os.path.join(args.root_dir, fn) for fn in args.skills_filenames.split(',')]
    job_dfs = [pickle.load(open(df_name, 'rb')) for df_name in job_df_names]
    skill_dfs = [pickle.load(open(df_name, 'rb')) for df_name in skill_df_names]
    dataset_names = args.datasets.split(',')
    if args.no_alt_titles:
        alt_titles = None
    else:
        alt_titles = [ALT_TITLES[x] for x in dataset_names]
    if args.w2v is not None:
        w2v_model = load_w2v_model(args.w2v)
        all_exact_match_jobs, job_titles_index = find_closest_matches(job_dfs,
                                                                      main_titles=[MAIN_TITLES[x] for x in dataset_names],
                                                                      alt_titles=alt_titles,
                                                                      w2v_model=w2v_model,
                                                                      id_cols=[JOB_ID_COLS[x] for x in dataset_names])
    else:
        all_exact_match_jobs, job_titles_index = find_exact_matches(job_dfs,
                                                                    [MAIN_TITLES[x] for x in dataset_names],
                                                                    alt_titles,
                                                                    [JOB_ID_COLS[x] for x in dataset_names])

    skills_to_investigate = get_skills_to_investigate_direct_match(skill_dfs,
                                                                   [SKILL_RELATIONS_ID_COLS[x] for x in dataset_names],
                                                                   job_titles_index)

    if args.countvec:
        vec_list, vec_model = calculate_tfidf_for_col([(skills_to_investigate[i], SKILL_LABELS[dataset_names[i]])
                                               for i in range(len(skills_to_investigate))], do_stem=True,
                                               count_vec=True, return_sum_all=False, dense=False, ngrams=args.ngrams)
    elif args.tfidf:
        vec_list, vec_model = calculate_tfidf_for_col([(skills_to_investigate[i], SKILL_LABELS[dataset_names[i]])
                                               for i in range(len(skills_to_investigate))], do_stem=True,
                                               count_vec=False, return_sum_all=False, dense=False, ngrams=args.ngrams)
    else:
        vec_list = None
        vec_model = None

    print(vec_model)

    if vec_list is not None:
        for i in range(len(skills_to_investigate)):
            skills_to_investigate[i]['O_'+dataset_names[i]] = vec_list[i]


    for i in range(len(skills_to_investigate)):
        with open(os.path.join(output_dir, dataset_names[i]+'_skills.pkl'), 'wb') as f:
            pickle.dump(skills_to_investigate[i], f)
    with open(os.path.join(output_dir, args.datasets.replace(',','_') + '_matches_df.pkl'), 'wb') as f:
        pickle.dump(all_exact_match_jobs, f)
    if vec_list is not None:
        with open(os.path.join(output_dir, args.datasets.replace(',','_') + '_tfidf_model.pkl'), 'wb') as f:
            pickle.dump(vec_model, f)

if __name__ == '__main__':
    main()