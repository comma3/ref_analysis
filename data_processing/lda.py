# From sklearn
# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause
import os, pickle
import numpy as np
import sqlite3

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import praw

from library import load_data


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

if __name__ == '__main__':

    n_features = 100
    n_components = 100
    n_top_words = 5
    pickle_path = '../ref_analysis/full.pkl'

    with open('../ref_analysis/data/manual_vocab.csv') as f:
        vocab = np.array([word.strip() for word in f])


    # Load the dataset and vectorize it.
    print("Loading dataset...")
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

    # Use tf-idf features for NMF.
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=n_features, stop_words='english')

    tfidf = tfidf_vectorizer.fit_transform(data)

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_features=n_features, stop_words='english')

    tf = tf_vectorizer.fit_transform(data)

    # Fit the NMF model
    print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
          "n_features={}".format(n_features))
    nmf = NMF(n_components=n_components, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)

    print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    # Fit the NMF model
    print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
          "tf-idf features\nn_features={}".format(n_features))
    nmf = NMF(n_components=n_components, random_state=1,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5).fit(tfidf)

    print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    print("Fitting LDA models with tf features, "
          "n_features={}".format(n_features))
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)
