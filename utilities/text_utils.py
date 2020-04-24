from utilities.constants import *
from nltk import PorterStemmer


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def generate_n_grams(tokens, return_string=True, n=3, joining_char='_'):
    if n == 1 or tokens == []:
        return tokens
    if len(tokens) >= n:
        n_gram_list = [tokens[i:i+n] for i in range(len(tokens)-n+1)]
        if return_string:
            n_gram_list = [joining_char.join(x) for x in n_gram_list]
        return n_gram_list + generate_n_grams(tokens, return_string=return_string, n=n-1, joining_char=joining_char)
    else:
        return generate_n_grams(tokens, return_string=return_string, n=n-1, joining_char=joining_char)

def remove_punkt(text, punkt_to_remove=PUNKT):
    nopunkt = " ".join("".join([" " if ch in punkt_to_remove else ch for ch in text]).split())
    return nopunkt

def tokenise_stem_punkt_and_stopword(text, punkt_to_remove=PUNKT, remove_numbers=True
                                     , stopword_set=STOPWORDS, do_stem = True, ngrams=1):

    """
    Handles text and tokenises it. By default lowercases, removes all HTML tags and escaped characters such as
    &nbsp;, etc. Has options for removing punctuation, numbers, stopwords and also for stemming.
    """

    if (text is None):
        return []

    # Lowercases the whole text.
    text = text.lower()
    # Tokenises and removes punctuation.
    if punkt_to_remove is not None:
        tokenised = remove_punkt(text, punkt_to_remove).split()
    else:
        tokenised = text.split()

    # Removes stopwords and numbers.
    if remove_numbers:
        tokenised = [x for x in tokenised if not is_number(x)]
    if stopword_set is not None:
        tokenised = [x for x in tokenised if x not in stopword_set]
    if do_stem:
        stemmer = PorterStemmer()
        tokenised = [stemmer.stem(x) for x in tokenised]
    if ngrams > 1:
        return generate_n_grams(tokenised, return_string=True, n=ngrams, joining_char=' ')
    return tokenised