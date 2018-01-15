# Modified from sklearn
# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause
import os, pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from library import load_data, LemmaTokenizer

def print_top_words(model, feature_names, n_top_words):
    """
    Prints top words from lda and nmf models
    """
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

def do_NMF(data, n_features=100, n_components=100, n_top_words=10, \
            stop_words='english', verbose=True):
    """
    """
    if verbose:
        print('Starting vectorizer fits...')
        print('TFIDF...')
    # Use tf-idf features for NMF.
    tfidf_vectorizer = TfidfVectorizer(max_features=n_features, \
                                stop_words=stop_words, max_df=0.95, min_df=0.01)
    tfidf = tfidf_vectorizer.fit_transform(data)

    if verbose:
        print('Fitting NMF..')
    # Fit the NMF model(Frobenius norm) with tf-idf features
    nmf = NMF(n_components=n_components, random_state=1,
              alpha=.1, l1_ratio=.5)
    nmf.fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
    print_top_words(nmf, tfidf_feature_names, n_top_words)
    # Fit the NMF model (generalized Kullback-Leibler divergence)
    nmf = NMF(n_components=n_components, random_state=1,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5)
    nmf.fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
    print_top_words(nmf, tfidf_feature_names, n_top_words)


def do_LDA(data, tf_features=100, lda_components=10, n_top_words=20, \
            stop_words='english', max_iter=20, verbose=True, ngram_range=(1,1),
            tokenizer=LemmaTokenizer()):
    """
    Performs Latent Dirichlet Allocation
    ---------------
    INPUT
    ---------------
    n_features: int - number of vectorizer words
    n_components: int - number of LDA Topics

    """
    if verbose:
        print('Starting vectorizer fits...')
        print('CountVectorizer...')
    # Use tf (raw term count) features for LDA.
    tf_vectorizer = CountVectorizer(max_features=tf_features, \
                                stop_words=stop_words, ngram_range=ngram_range,\
                                tokenizer=tokenizer)
    tf = tf_vectorizer.fit_transform(data)
    lda = LatentDirichletAllocation(n_components=lda_components, max_iter=max_iter,
                                    learning_method='online',
                                    random_state=0)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    if verbose:
        print("\nTopics in LDA model:")
        print_top_words(lda, tf_feature_names, n_top_words)

    return lda

if __name__ == '__main__':
    pickle_path = '../ref_analysis/big_1000.pkl'

    stop_words = []
    with open('../ref_analysis/data/common-english-words.csv') as f:
        for word in f:
            stop_words.append(word.strip())
    #print(stop_words)

    docs = load_data(n_games=300, pickle_path=pickle_path)
    # data = []
    # for doc in docs:
    #     for word in doc.lower().split():
    #         if word in vocab:
    #             data.append(doc)
    #             break
    data = []
    for game in docs:
        for doc in game:
            data.append(doc[1])

    #do_NMF(data)
    do_LDA(data, n_features=1000, n_components=30)
