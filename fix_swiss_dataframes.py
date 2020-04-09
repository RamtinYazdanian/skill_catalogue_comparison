import pickle
import argparse
import os

SWISS_CORRECTIONS = {
    'Échafaudeur': 'Scaffolder',
    'Électroplaste': 'Electroplast',
    'Étancheur': 'Waterproofer',
    'Télématicien': 'Telematician',
    'Vannier créateur': 'Creative basket maker',
    'Vernisseur industriel': 'Industrial varnisher',
    'Employé de commerce': 'Commerce employee',
    'Formateur pour l enseignement des branches pratiques': 'Trainer for the teaching of practical branches',
    'Techniscéniste': 'Scene technician',
    'Pédagogue curatif UNI': 'Curative pedagogue',
    'Médiamaticien': 'Mediamatician',
    'Papetier': 'Paper maker',
    'Polygraphe': 'Lie detector',
    'Opérateur en informatique': 'Computer operator',
    'Professionnel du cheval': 'Horse professional',
    'Spécialiste en restauration': 'Catering specialist',
    'Spécialiste en restauration de système': 'System catering specialist',
    'Mécanicien d appareils à moteur': 'Mechanic of motorized devices',
    'Agent de détention BF': 'Detention officer',
    'Maître socioprofessionnel ES': 'Socioprofessional master',
    'Luthier': 'Stringed instrument maker',
    'Juriste UNI': 'jurist',
    'Menuisier ébénisterie': 'Cabinetmaker carpenter',
    'Esthéticien': 'Beautician',
    'Électronicien': 'Electronics specialist',
    'Électricien de réseau': 'Network electrician',
    'Facteur d instruments de musique': 'Musical instrument maker',
    'Facteur d instruments à vent': 'Wind instrument maker',
    'Électricien de montage': 'Electrician, fitter',
    'Conducteur de chiens': 'Dog handler',
    'Courtepointier': 'Quilter',
    'Carrossier peintre': 'Vehicle body shop painter',
    'Agent d exploitation': 'Operations officer',
    'Assistant socio éducatif': 'Socio-educative assistant',
    'Bottier orthopédiste': 'Orthopedic shoemaker',
    'Spécialiste en communication hôtelière': 'Hotel communication specialist',
    'Entraîneur de sport de performance BF': 'Performance sports coach',
    'Cuisinier en diététique': 'Diet cook',
    'Carrossier tôlier': 'Sheet metal vehicle body builder',
    'Criminaliste UNI': 'Criminalist',
    'Criminologue UNI': 'Criminologist'
}

SWISS_ABBREVIATIONS = {
    'HEC': 'School of Commerce',
    'HES': 'Specialized College',
    'ES': 'Graduate School',
    'UNI': 'University',
    'BF': "Federal Patent"
}

def remove_A(text):
    return ' '.join([word for word in text.split(' ') if word != 'A']).strip()

def capitalise_first_letter(text):
    return text[0].upper()+text[1:]

def all_caps(text):
    return text.upper() == text

def correct_translations(df, col_src, col_dest, correction_dict, abbreviations=None):
    df[col_dest] = df.apply(lambda x: correction_dict[x[col_src]]
                            if x[col_src] in correction_dict else x[col_dest], axis=1)
    df[col_dest] = df[col_dest].apply(lambda x: capitalise_first_letter(remove_A(x)))
    if abbreviations is not None:
        df[col_dest] = df.apply(lambda x: x[col_dest].replace(x[col_src].split(' ')[-1], '')
                            + ' ' + abbreviations[x[col_src].split(' ')[-1]]
                            if x[col_src].split(' ')[-1] in abbreviations else x[col_dest], axis=1)
    else:
        df[col_dest] = df.apply(lambda x: x[col_dest].replace(x[col_src].split(' ')[-1], '')
                                if all_caps(x[col_src].split(' ')[-1]) else x[col_dest], axis=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_df', type=str, required=True)
    parser.add_argument('--abbreviations', action='store_true')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    swiss_data = pickle.load(open(args.original_df, 'rb'))
    if args.abbreviations:
        correct_translations(swiss_data, 'job', 'job_en', SWISS_CORRECTIONS, abbreviations=SWISS_ABBREVIATIONS)
    else:
        correct_translations(swiss_data, 'job', 'job_en', SWISS_CORRECTIONS, abbreviations=None)
    swiss_skills = swiss_data.copy()
    swiss_jobs = swiss_data.drop(columns=['task', 'task_en']).drop_duplicates().reset_index().drop(columns=['index'])

    with open(os.path.join(args.output_dir, 'swiss_fixed_skills.pkl'), 'wb') as f:
        pickle.dump(swiss_skills, f)
    with open(os.path.join(args.output_dir, 'swiss_fixed_jobs.pkl'), 'wb') as f:
        pickle.dump(swiss_jobs, f)


if __name__ == '__main__':
    main()