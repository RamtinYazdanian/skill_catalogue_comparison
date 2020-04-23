from utilities.common_utils import *
from utilities.text_utils import *
import difflib
from itertools import chain
from collections import Counter
from nltk import PorterStemmer
import os
import pickle
import argparse

def make_word_stem_map(dfs_and_cols, stems_to_keep=None, ngrams=1):
    all_words = chain.from_iterable(
            [df[col].apply(lambda x: ' '.join(
                tokenise_stem_punkt_and_stopword(x, do_stem=False, ngrams=ngrams))).values
                 for df, col in dfs_and_cols])
    word_counts = Counter(all_words)
    words_set = set(all_words)
    stemmer = PorterStemmer()
    word_stem_map = {word: ' '.join([stemmer.stem(subword) for subword in word.split(' ')])
                     for word in words_set}
    if stems_to_keep is not None:
        word_stem_map = {k:v for k,v in word_stem_map.items() if v in stems_to_keep}
    return word_stem_map, word_counts

def invert_word_stem_map(word_stem_map, word_counts):
    inverted_map = dict()
    for word, stem in word_stem_map.items():
        inverted_map[stem] = inverted_map.get(stem, [])
        inverted_map[stem].append(word)
    final_inverted = {stem: (inverted_map[stem])[np.argmax([word_counts[word] for word in inverted_map[stem]])]
                      for stem in inverted_map}
    return final_inverted

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--datasets', type=str, required=True)
    parser.add_argument('--stems_to_keep', type=str, default=None)
    args = parser.parse_args()

    dataset_names = args.datasets.split(',')
    col_names = ['O_'+ds for ds in dataset_names]
    skill_dfs = [pickle.load(open(os.path.join(args.root_dir, ds+'_skills.pkl'), 'rb')) for ds in dataset_names]
    if args.stems_to_keep is None:
        stems_to_keep = None
    else:
        stems_to_keep = pickle.load(open(args.stems_to_keep, 'rb'))
    word_stem_map, word_counts = make_word_stem_map([(skill_dfs[i], col_names[i]) for i in range(len(dataset_names))],
                                                    stems_to_keep, 3)
    final_map = invert_word_stem_map(word_stem_map, word_counts)
    with open(os.path.join(args.root_dir, 'stem_reverse_map.pkl'), 'wb') as f:
        pickle.dump(final_map, f)

if __name__ == '__main__':
    main()





