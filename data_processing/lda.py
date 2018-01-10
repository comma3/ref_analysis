# Modified from sklearn
# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause
import os, pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from library import load_data

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


def do_lda_and_NMF(data, n_features=100, n_components=1000, n_top_words=10):
    """
    """
    # Use tf-idf features for NMF.
    tfidf_vectorizer = TfidfVectorizer(max_features=n_features, \
                                stop_words='english', max_df=0.95, min_df=0.01)
    tfidf = tfidf_vectorizer.fit_transform(data)
    # Use tf (raw term count) features for LDA.
    tf_vectorizer = CountVectorizer(max_features=n_features, \
                                stop_words='english', max_df=0.95, min_df=0.01)
    tf = tf_vectorizer.fit_transform(data)

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

    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    print("\nTopics in LDA model:")
    print_top_words(lda, tf_feature_names, n_top_words)

if __name__ == '__main__':
    pickle_path = '../ref_analysis/full.pkl'


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

    do_lda_and_NMF(data)
