import pandas
import os
import argparse
from utilities.common_utils import *
from utilities.pandas_utils import *
import pickle

def associate_words_with_datasets(df, vocab, suffixes):
    for i in range(len(suffixes)):
        suffix = suffixes[i]
        other_suffix = suffixes[(i+1)%len(suffixes)]
        df['words_'+suffix] = df.apply(lambda x: [y for y in x['top_words'] if
                                                  x['O_'+suffix][1,vocab[y]] > x['O_'+other_suffix][1,vocab[y]]])
    return

def join_and_log_likelihood(dfs, df_suffixes, col_to_join_by, countvec_model,
                            aggregation_df=None, names_df=None, significant_only=False, top_n=20):
    """
    Only accepts two dfs. This is because the log likelihood ratio is designed for two corpora.
    """
    partially_aggregated_dfs = list()
    for i in range(len(dfs)):
        df = dfs[i]
        partially_aggregated_dfs.append(
            df[[col_to_join_by, 'O_' + df_suffixes[i]]].groupby(col_to_join_by).sum())
    print('Partial aggregation complete')

    current_joined_df = partially_aggregated_dfs[0]
    for i in range(1, len(partially_aggregated_dfs)):
        current_joined_df = current_joined_df.join(partially_aggregated_dfs[i])

    if aggregation_df is not None:
        aggregation_col = aggregation_df.columns.values[0]
        current_joined_df = current_joined_df.join(aggregation_df)
        current_joined_df = current_joined_df.groupby(aggregation_col).sum()

    print('Finished all the joins')

    # print(current_joined_df[current_joined_df.isnull().any(axis=1)])
    print(current_joined_df.shape)
    null_values = current_joined_df[current_joined_df.isnull().any(axis=1)].join(names_df)
    current_joined_df = current_joined_df.dropna(axis=0, how='any')

    print(current_joined_df.shape)
    for i in range(len(dfs)):
        current_joined_df['N_' + df_suffixes[i]] = current_joined_df.apply(lambda x:
                                                                           np.sum(x['O_' + df_suffixes[i]]), axis=1)
    print('N_i computed')

    for i in range(len(dfs)):
        current_joined_df['E_' + df_suffixes[i]] = current_joined_df.apply(lambda x:
                                                                           x['N_' + df_suffixes[i]] *
                                                                           sum([x['O_' + df_suffixes[j]] for j in
                                                                                range(len(dfs))]) /
                                                                           sum([x['N_' + df_suffixes[j]] for j in
                                                                                range(len(dfs))]), axis=1)
    print('E_i computed')

    current_joined_df['LL'] = current_joined_df.apply(lambda x: np.nan_to_num(np.array((2 * sum(
        [np.array(x['O_' + df_suffixes[j]].todense().flatten()) *
         np.log(np.array(x['O_' + df_suffixes[j]] / x['E_' + df_suffixes[j]]).flatten())
         for j in range(len(dfs))]
                    ))).flatten(), nan=-1e20), axis=1)
    print('LL computed')

    if not significant_only:
        current_joined_df['top_words'] = find_top_words(current_joined_df,
                                                        'LL', countvec_model, top_n=top_n, modify_df=False)
    else:
        current_joined_df['top_words'] = find_top_words(current_joined_df,
                                                        'LL', countvec_model, top_n=top_n, lower_bound=7.82,
                                                        modify_df=False)
    for i in range(len(dfs)):
        associate_words_with_datasets(current_joined_df, countvec_model.vocabulary_, df_suffixes)

    if names_df is not None:
        current_joined_df = current_joined_df.join(names_df)

    return current_joined_df, null_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skills_filenames', type=str, required=True, help=
                        'Names of pickled skills and relations dataframe files, comma separated.')
    parser.add_argument('--jobs_matched', type=str, required=True, help='Used for the names and the career clusters/'
                                                                        'pathways.')
    parser.add_argument('--datasets', type=str, required=True, help=
                                        'Names of datasets, comma separated, '
                                        'same order as filenames. Options are ESCO and ONET.')
    parser.add_argument('--countvec_model', type=str, required=True)
    parser.add_argument('--agg_col', type=str, choices=['Career_Cluster', 'Career_Pathway'], default=None)
    parser.add_argument('--significant_only', action='store_true')
    parser.add_argument('--top_n', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    skill_df_names = [os.path.join(args.root_dir, fn) for fn in args.skills_filenames.split(',')]
    skill_dfs = [pickle.load(open(df_name, 'rb')) for df_name in skill_df_names]
    dataset_names = args.datasets.split(',')
    names_df = pickle.load(open(args.jobs_matched, 'rb'))
    countvec_model = pickle.load(open(args.countvec_model, 'rb'))
    if args.agg_col is not None:
        # We get a single column dataframe; the index is the one used to index all the matches, i.e. common_id.
        agg_df = names_df[args.agg_col.replace('_', ' ')]
    else:
        agg_df = None
    jobs_ll, null_values = join_and_log_likelihood(skill_dfs, dataset_names, 'common_id',
                               countvec_model, aggregation_df=agg_df, names_df=names_df,
                               significant_only=args.significant_only, top_n=args.top_n)
    with open(os.path.join(args.output_dir, 'jobs_ll.pkl'), 'wb') as f:
        pickle.dump(jobs_ll, f)


if __name__ == '__main__':
    main()