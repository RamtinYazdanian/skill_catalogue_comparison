from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from itertools import chain
from utilities.common_utils import *
import numpy as np
import pandas as pd

def form_word_clusters(df, cols, w2v, dataset_names, n_clusters=10, reduce_dim=None, normalise=False):
    dataset_words = {dataset_names[i]: set(chain.from_iterable(df[cols[i]].values.tolist())) for i in range(len(cols))}
    word_index = list(set(chain.from_iterable(dataset_words.values())))
    word_vector_matrix = np.vstack([calculate_word_vec(word, w2v) for word in word_index])
    if reduce_dim is not None:
        pca_model = PCA(n_components=reduce_dim)
        word_vector_matrix = word_vector_matrix - np.mean(word_vector_matrix, axis=0)
        word_vector_matrix = pca_model.fit_transform(word_vector_matrix)
    if normalise:
        word_vector_matrix = word_vector_matrix / np.reshape(np.linalg.norm(word_vector_matrix, axis=1),
                                                             newshape=(word_vector_matrix.shape[0], 1))
    if isinstance(n_clusters, int):
        cluster_model = KMeans(n_clusters=n_clusters)
        results = cluster_model.fit_predict(word_vector_matrix)
    elif isinstance(n_clusters, list):
        models = [KMeans(n_clusters=n) for n in n_clusters]
        predictions = list()
        silhouettes = list()
        for model in models:
            current_results = model.fit_predict(word_vector_matrix)
            predictions.append(current_results)
            silhouettes.append(silhouette_score(word_vector_matrix, current_results))
        print('Silhouette scores:')
        print({n_clusters[i]: silhouettes[i] for i in range(len(n_clusters))})
        results = predictions[np.argmax(silhouettes)]
    else:
        raise Exception('n_clusters must be either an integer or a list of integers')
    return dataset_words, {word_index[i]: results[i] for i in range(len(word_index))}, word_index, word_vector_matrix

def get_word_dataset_map(dataset_words, word_index):
    word_datasets = list()
    dataset_names = list(dataset_words.keys())
    word_sets = [dataset_words[ds_name] for ds_name in dataset_names]
    for word in word_index:
        done = False
        for i in range(len(dataset_names)):
            if word in word_sets[i] and word not in word_sets[(i+1)%len(dataset_names)]:
                word_datasets.append(dataset_names[i])
                done = True
                break
        if not done:
            word_datasets.append('Both')

    return word_datasets

def visualise_word_clusters(dataset_words, cluster_mappings, word_index, word_vector_matrix):
    pca_model = PCA(n_components=2)
    word_vector_matrix = word_vector_matrix - np.mean(word_vector_matrix, axis=0)
    transformed_vectors = pca_model.fit_transform(word_vector_matrix)
    word_datasets = get_word_dataset_map(dataset_words, word_index)
    word_clusters = [cluster_mappings[word] for word in word_index]
    return word_datasets, word_clusters, transformed_vectors

def return_cluster_word_sets(cluster_mappings, dataset_words):
    words = list(cluster_mappings.keys())
    word_datasets = get_word_dataset_map(dataset_words, words)
    word_clusters = [cluster_mappings[word] for word in words]
    return pd.DataFrame({'word': words, 'cluster': word_clusters, 'dataset': word_datasets})






