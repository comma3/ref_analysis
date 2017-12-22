import sqlite3, os, pickle, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

import praw

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix
from collections import defaultdict
from sklearn.metrics import silhouette_score



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

def make_hashable(csr):
    """
    Couldn't hash csr. Convert to string.
    """
    string = ''
    for i in csr:
        string += str(i)
    return string


def to_csr(hashable):
    """
    Converts
    """
    hashable = hashable.replace('(', '').replace(')', '').replace(',', '')
    data = np.array([list(map(float, item.split())) for item in hashable.split('\n')])
    return csr_matrix((data[:, 2], (data[:, 0], data[:, 1])))


def k_means(time_stamped_vectors, k=10, max_iter=100, time_scale_factor=0.001, threshold=0.1):
    """Performs k means

    Args:
    - time_stamped_vectors - feature matrix of (timestamp, tf_vector)
    - k - number of clusters
    - max_iter - maximum iterations

    Returns:
    - clusters - dict mapping cluster centers to observations
    """
    centers = [tuple(pt) for pt in random.sample(list(time_stamped_vectors), k)]
    for i in range(max_iter):
        print('Iteration:', i)
        clusters = defaultdict(list)
        for time, datapoint in time_stamped_vectors:
            distances = [(euclidean(datapoint.todense(), center[1].todense()) + euclidean(time, center[0]) * time_scale_factor) for center in centers]
            center = centers[np.argmin(distances)]
            center = center[0], make_hashable(center[1])
            clusters[center].append((time,datapoint))

        new_centers = []
        for center, pts in clusters.items():
            time = 0
            matrix = np.zeros(pts[0][1].shape[1]).T
            for pt in pts:
                time += pt[0]
                matrix += pt[1]
            new_center = (time/len(pts), csr_matrix(matrix/len(pts)))
            new_centers.append(tuple(new_center))

        dist = 0
        for center, new_center in zip(centers, new_centers):
            dist += euclidean(new_center[1].todense(), center[1].todense()) + euclidean(new_center[0], center[0]) * time_scale_factor
        print('Centroid movement:', dist)
        if dist < threshold:
            print('Converged!')
            break

        centers = new_centers
    return clusters


def load_data():
    if not os.path.isfile('working.pkl'):
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
        pickle.dump(documents, open('working.pkl', 'wb'))
    else:
        documents = pickle.load(open('working.pkl', 'rb'))

    docs = []
    times = []
    for time, doc in documents:
        docs.append(doc)
        times.append(time)


    return times, docs

if __name__ == '__main__':

    times, docs = load_data()
    with open('manual_vocab.csv') as f:
        vocab = np.array([word.strip() for word in f])
    tf_vectorizer = TfidfVectorizer(vocabulary=vocab)
    #tf_vectorizer = CountVectorizer(vocabulary=vocab)

    time_stamped_vectors = zip(times, tf_vectorizer.fit_transform(docs))

    ref_related = [tsv for tsv in time_stamped_vectors if tsv[1].nnz]

    clusters = k_means(ref_related)
    for key in clusters.keys():
        print(len(clusters[key]))
    for num, center in enumerate(clusters):
        print('Centroid:', num)
        print('Time:', center[0])
        c_vec = to_csr(center[1]).todense().argsort()
        #t = to_csr(centroid[1]).todense().T
        #print(t[c_vec[:,-1:-11:-1]])
        print('Top Words:', vocab[c_vec[:,-1:-11:-1]])
        plt.scatter(center[0], 0)
        times = [pt[0] for pt in clusters[center]]
        plt.hist(times)
    plt.show()
