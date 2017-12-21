import sqlite3
import pickle, os
from time import time
import numpy as np
import pandas as pd

import praw

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans

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


if __name__ == '__main__':
    if not os.path.isfile('working.pkl'):
        db = '/data/cfb_game_db.sqlite3'
        subreddit = 'cfb'
        bot_params = 'bot1' # These are collected from praw.ini
        reddit = praw.Reddit(bot_params)
        threads = collect_game_threads(db)
        print(threads)
        documents = []
        times = []
        for thread in threads:
            print('working on thread:', thread)
            submission = reddit.submission(id=thread[0])
            submission.comment_sort = 'new'
            comments = submission.comments
            comments.replace_more()
            for top_level_comment in comments:
                documents.append(top_level_comment.body)
                times.append(top_level_comment.created_utc)
        pickle.dump(documents, open('working.pkl', 'wb'))
    else:
        documents = pickle.load(open('working.pkl', 'rb'))

    print(len(documents))

    vocab = []
    with open('manual_vocab.csv') as f:
        for word in f:
            vocab.append(word.strip())
    print(vocab)
    n_features = 10

    tfidf_vectorizer = TfidfVectorizer(vocabulary=vocab)
    tf_vectorizer = CountVectorizer(vocabulary=vocab)

    X = tf_vectorizer.fit_transform(documents)
    print(X)
    kmeans = KMeans()
    kmeans.fit(X)
