import pandas as pd
import pickle
import argparse
import os
from utilities.common_utils import *
from functools import reduce


def join_series_with_char(series, char='\n'):
    return reduce(lambda x, y: x + char + y, series)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--esco_dir', type=str, required=True)
    parser.add_argument('--onet_dir', type=str, required=True)
    parser.add_argument('--wordvec', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    # Loading the data
    esco_jobs = pd.read_csv(os.path.join(args.esco_dir,'occupations_en.csv'))
    onet_jobs = pd.read_csv(os.path.join(args.onet_dir,'Occupation Data.txt'), sep='\t')
    onet_alternative_titles = pd.read_csv(os.path.join(args.onet_dir,'Alternate Titles.txt'),
                                          sep='\t')
    onet_career_clusters = pd.read_csv(os.path.join(args.onet_dir,'All_Career_Clusters.csv'),
                                       sep=',')
    if args.wordvec is not None:
        w2v_model = load_w2v_model(args.wordvec,
                               is_bin=True)
    else:
        w2v_model = None

    # Joining the O*NET data with each job's alternate labels.
    alternate_labels = onet_alternative_titles[['O*NET-SOC Code', 'Alternate Title']]. \
        groupby('O*NET-SOC Code').agg({'Alternate Title':
                                           join_series_with_char}).reset_index()
    onet_jobs = pd.merge(onet_jobs, alternate_labels, on='O*NET-SOC Code')
    onet_jobs = pd.merge(onet_jobs, onet_career_clusters[['Code', 'Career Cluster', 'Career Pathway']],
                         left_on='O*NET-SOC Code', right_on='Code').drop('Code')

    if w2v_model is not None:
        esco_jobs_word_vectors = get_df_word_vectors(esco_jobs, ['preferredLabel', 'altLabels'], w2v_model)
        onet_jobs_word_vectors = get_df_word_vectors(onet_jobs, ['Title', 'Alternate Title'], w2v_model)
    else:
        esco_jobs_word_vectors = None
        onet_jobs_word_vectors = None

    # Loading skills and relations data
    esco_skills = pd.read_csv(os.path.join(args.esco_dir,'skills_en.csv'))
    esco_job_skill_relations = pd.read_csv(
        os.path.join(args.esco_dir,'occupationSkillRelations.csv'))
    onet_skills_and_relations = pd.read_csv(os.path.join(args.onet_dir,'/Task Statements.txt'),
                                            sep='\t')

    # Preparing O*NET skills and relations data
    onet_skills_and_relations = pd.merge(onet_skills_and_relations,
                                         onet_jobs, on='O*NET-SOC Code')
    onet_skills_and_relations['is_essential'] = onet_skills_and_relations['Task Type'].apply(lambda x: x == 'Core')
    onet_skills_and_relations = pd.merge(onet_skills_and_relations,
                                         onet_career_clusters[['Code', 'Career Cluster', 'Career Pathway']],
                                                left_on='O*NET-SOC Code', right_on='Code').drop('Code')

    # Preparing ESCO skills and relations data
    esco_skills_and_relations = pd.merge(pd.merge(esco_skills, esco_job_skill_relations.drop(columns=['skillType']),
                                                  left_on='conceptUri', right_on='skillUri').drop(
        columns=['conceptUri']),
                                         esco_jobs[['conceptUri', 'preferredLabel', 'altLabels', 'description']],
                                         left_on='occupationUri', right_on='conceptUri', suffixes=('_skill', '_job'))[[
            'preferredLabel_skill', 'altLabels_skill', 'preferredLabel_job', 'altLabels_job', 'reuseLevel',
            'description_skill', 'description_job', 'relationType', 'occupationUri', 'skillType'
        ]]
    esco_skills_and_relations['is_essential'] = esco_skills_and_relations.relationType.apply(lambda x: x == 'essential')


    # Saving the results
    output_dir = args.output_dir
    with open(os.path.join(output_dir,'onet_jobs.pkl'), 'wb') as f:
        pickle.dump(onet_jobs, f)
    with open(os.path.join(output_dir,'esco_jobs.pkl'), 'wb') as f:
        pickle.dump(esco_jobs, f)
    with open(os.path.join(output_dir,'onet_skills_and_relations.pkl'), 'wb') as f:
        pickle.dump(onet_skills_and_relations, f)
    with open(os.path.join(output_dir,'esco_skills_and_relations.pkl'), 'wb') as f:
        pickle.dump(esco_skills_and_relations, f)

    if esco_jobs_word_vectors is not None:
        with open(os.path.join(output_dir,'esco_jobs_word_vectors.pkl'), 'wb') as f:
            pickle.dump(esco_jobs_word_vectors, f)

    if onet_jobs_word_vectors is not None:
        with open(os.path.join(output_dir,'onet_jobs_word_vectors.pkl'), 'wb') as f:
            pickle.dump(onet_jobs_word_vectors, f)



if __name__ == '__main__':
    main()