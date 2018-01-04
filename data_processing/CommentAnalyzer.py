import sqlite3
import pickle, os
from time import time
import numpy as np
import pandas as pd
import scipy.stats as stats

import praw

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


# Want to return something of the format:
# team, rule, incorrect, strength, id
# UNC, offsides, yes, 10, off001

def collect_game_threads(db):
    """
    TODO
    """
    conn = sqlite3.connect(db)
    curr = conn.cursor()
    curr.execute("""SELECT
                    game_thread
                    FROM
                    games
                    LIMIT
                    50
                    """)
    games = curr.fetchall()
    conn.close()
    return games


def determine_reply_score(reply_forest):
    """
    Takes the replies from a top level comment and determines support for
    that comment. May just be better to use top level score.
    """
    pass

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

if __name__ == '__main__':
    if not os.path.isfile('working.pkl'):
        db = '/data/cfb_game_db.sqlite3'
        subreddit = 'cfb'
        bot_params = 'bot1' # These are collected from praw.ini
        reddit = praw.Reddit(bot_params)
        threads = collect_game_threads(db)
        print(threads)
        vocab = []
        for thread in threads:
            print('working on thread:', thread)
            submission = reddit.submission(id=thread[0])
            submission.comment_sort = 'new'
            comments = submission.comments
            comments.replace_more()
            for top_level_comment in comments:
                vocab.append(top_level_comment.body)
        pickle.dump(vocab, open('working.pkl', 'wb'))
    else:
        vocab = pickle.load(open('working.pkl', 'rb'))

    print(len(vocab))
    stop_words = []
    with open('common-english-words.csv') as f:
        for word in f:
            stop_words.append(word.strip())
    print(stop_words)
    n_samples = 2000
    n_features = 100
    n_components = 15
    n_top_words = 20
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_df=0.99, min_df=0.01, max_features=n_features, stop_words=stop_words)
    tf_vectorizer = CountVectorizer(ngram_range=(1, 3), max_df=0.99, min_df=0.01, max_features=n_features, stop_words=stop_words)
    # for NMF
    tfidf = tfidf_vectorizer.fit_transform(vocab)
    # for LatentDirichletAllocation
    tf = tf_vectorizer.fit_transform(vocab)
    # Fit the NMF model
    print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    t0 = time()
    nmf = NMF(n_components=n_components, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    # Fit the NMF model
    print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
          "tf-idf features, n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    t0 = time()
    nmf = NMF(n_components=n_components, random_state=1,
              beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
              l1_ratio=.5).fit(tfidf)
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    t0 = time()
    lda.fit(tf)
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)

    # Leave here to remember these
    # for top_level_comment in comments:
    #     #print(top_level_comment.body)
    #     #print(top_level_comment.score)
    #     #print(top_level_comment.author_flair_css_class)
    #     #print(top_level_comment.author_flair_text)
    #     #print(top_level_comment.gilded) # int
    #     #print(top_level_comment.replies) # None or comment forrest
    # #print(dir(top_level_comment.replies))
    # #print(dir(top_level_comment))
    # print('Len comments:', len(comments))
    # print(thread[0][0])
