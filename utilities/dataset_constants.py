job_id_cols = {'ONET': 'O*NET-SOC Code',
               'ESCO': 'conceptUri',
               'SWISS': 'job_en'}
skill_relations_id_cols = {'ESCO': 'occupationUri', 'ONET': 'O*NET-SOC Code', 'SWISS': 'job_en'}
main_titles = {'ONET': 'Title', 'ESCO': 'preferredLabel', 'SWISS': 'job_en'}
alternative_titles = {'ONET': None, 'ESCO': 'altLabels', 'SWISS': None}
skill_labels = {'ONET': 'Task', 'ESCO': 'description_skill', 'SWISS': 'task_en'}