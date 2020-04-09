# coding=utf-8
import os
import errno
import gensim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def remove_nones(l):
    return [x for x in l if x is not None]

def none_to_zero(l, d=300):
    return [x if x is not None else np.zeros(d) for x in l]

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST and exception.errno != errno.EPERM:
            raise

def invert_dict(d):
    return {d[k]: k for k in d}

def load_w2v_model(filename, is_bin=True):
    model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=is_bin)
    return model

def calculate_word_vec(s, w2v_model):
    if isinstance(s, str):
        if '\n' in s:
            return np.sum(none_to_zero([calculate_word_vec(x, w2v_model) for x in s.split('\n')]), axis=0)
        else:
            return np.sum([w2v_model.wv[x] if x in w2v_model.wv else np.zeros(w2v_model.vector_size) for x in s.split(' ')], axis=0)
    elif isinstance(s, list):
        return np.sum(none_to_zero([calculate_word_vec(x, w2v_model) for x in s]), axis=0)

def calculate_phrase_pair_similarity(w2v_model, word_1, word_2):
    v1 = calculate_word_vec(word_1, w2v_model)
    v2 = calculate_word_vec(word_2, w2v_model)
    return np.dot(v1,v2) / (np.sqrt(np.dot(v1,v1)*np.dot(v2,v2)))

def get_df_word_vectors(df, column_list, w2v_model):
    print(df.columns)
    print(column_list)
    if column_list is None:
        return None
    column_list = [col for col in column_list if col is not None]
    df_word_vectors = df.apply(lambda x: calculate_word_vec([x[y] for y in column_list], w2v_model), axis=1).values.tolist()
    df_word_vectors = np.array(df_word_vectors)
    return df_word_vectors

def get_pairwise_similarities(word_vecs_1, word_vecs_2):
    return cosine_similarity(word_vecs_1, word_vecs_2)

def find_one_title(title, df, df_word_vectors, w2v_model, top_n=10):
    title_vector = calculate_word_vec(title, w2v_model)
    title_vector = title_vector.reshape((1,title_vector.size))
    similarities = get_pairwise_similarities(title_vector, df_word_vectors).flatten()
    top_hit_indices = (np.argsort(similarities)[::-1])[:top_n]
    return df.iloc[top_hit_indices]