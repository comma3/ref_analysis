import sqlite3, os, pickle, random
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import praw

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import euclidean


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
                    1
                    """)
    games = curr.fetchall()
    conn.close()
    return games

def load_data():
    pickle_path = '../ref_analysis/working.pkl'
    if not os.path.isfile(pickle_path):
        db = '/data/cfb_game_db.sqlite3'
        subreddit = 'cfb'
        bot_params = 'bot1' # These are collected from praw.ini
        reddit = praw.Reddit(bot_params)
        threads = collect_game_threads(db)
        print(threads)
        documents = []
        for thread in threads:
            print('working on thread:', thread)
            submission = reddit.submission(id=thread[0])
            submission.comment_sort = 'new'
            comments = submission.comments
            comments.replace_more()
            for top_level_comment in comments:
                documents.append([top_level_comment.created_utc, top_level_comment.body])
        pickle.dump(documents, open(pickle_path, 'wb'))
    else:
        documents = pickle.load(open(pickle_path, 'rb'))

    docs = []
    times = []
    for time, doc in documents:
        docs.append(doc)
        times.append(time)

    return times, docs

def replace_many(to_replace, replace_with, string):
    """
    Replace all of the items in the to_replace list with replace_with.
    """
    for s in to_replace:
        string.replace(s, replace_with)
    return string

def categorize(string):
    string = string.lower()
    string = replace_many(['pass interference', 'opi', 'dpi', 'interference'],'pi' ,string)
    string = replace_many(['holding'],'hold' ,string)
    string = replace_many(['pass interference', 'opi', 'dpi', 'interference'],'pi' ,string)







if __name__ == '__main__':

    times, docs = load_data()
    with open('../ref_analysis/data/manual_vocab.csv') as f:
        vocab = np.array([word.strip() for word in f])
    #tf_vectorizer = TfidfVectorizer()
    tf_vectorizer = CountVectorizer(vocabulary=vocab)
    ref_docs = []
    ref_times = []
    for time, doc in zip(times,docs):
        #TODO: Lemmatize
        for word in doc.lower().split():
            if word in vocab:
                ref_docs.append(doc)
                ref_times.append(time)
                break
    time_stamped_vectors = list(zip(ref_times, tf_vectorizer.fit_transform(ref_docs), ref_docs))

    #ref_docs = [tsv for tsv in time_stamped_vectors if tsv[1].nnz]
    print(len(time_stamped_vectors))
    clusters = KernelDensity(kernel='epanechnikov', bandwidth=0.5,leaf_size=10).fit(np.array(ref_times)[:, np.newaxis])
    print(clusters.get_params())
    plt.hist(ref_times, bins=100, normed=True, color='r')

    X_plot = np.linspace(min(ref_times), max(ref_times), 500)[:, np.newaxis]
    #print(X_plot.shape)
    #print(ref_times.shape)
    log_dens = clusters.score_samples(X_plot)
    print(log_dens)
    plt.plot(X_plot[:, 0], np.exp(log_dens), ls='-', color='b', alpha=0.3)


    plt.show()

    # with open('review.txt', 'w') as f:
    #     for tsv in ref_docs:
    #         f.write(tsv[2])
    #         f.write('\n========\n')
