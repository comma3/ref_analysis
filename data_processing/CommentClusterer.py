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

from library import load_data, replace_many

class CommentClusterer(object):

    """docstring for ."""
    def __init__(self, vectorizer='count', vocab=None, verbose=True, print_figs=False):

        self.print_figs = print_figs
        self.verbose = verbose
        self.vocab = vocab

        if vectorizer == 'count':
            self.tf_vectorizer = CountVectorizer()
        else:
            self.tf_vectorizer = TfidfVectorizer()

        self.clusters = defaultdict(list)
        self.game_vectors = []
        self.documents = None


    def _make_hashable(self, csr):
        """
        Couldn't hash csr. Convert to pickle.
        """
        return pickle.dumps(csr)

    def _to_csr(self, hashable):
        """
        Converts pickle back to csr
        """
        return pickle.loads(hashable)


    def k_means(time_stamped_vectors, k=10, max_iter=100, time_scale_factor=0.001, threshold=0.1):
        """Performs k means

        Args:
        - time_stamped_vectors - feature matrix of (timestamp, tf_vector)
        - k - number of clusters
        - max_iter - maximum iterations

        Returns:
        - self.clusters - dict mapping cluster centers to observations
        """
        centers = [tuple(pt) for pt in random.sample(list(time_stamped_vectors), k)]
        for i in range(max_iter):

            for time, datapoint, doc in time_stamped_vectors:
                distances = [(euclidean(datapoint.todense(), \
                            center[1].todense()) + euclidean(time, center[0]) \
                             * time_scale_factor) for center in centers]
                center = centers[np.argmin(distances)]
                center = center[0], self._make_hashable(center[1])
                self.clusters[center].append((time,datapoint,doc))
            new_centers = []
            for center, pts in self.clusters.items():
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
            centers = new_centers
            if self.verbose:
                print('Iteration:', i)
                print('Centroid movement:', dist)
                if dist < threshold:
                    print('Converged!')
                    break

    def add_tf_vectors(self):
        """
        Finds
        """
        for game in self.documents:
            ref_related = []
            ref_times = []
            for doc in game:
                for word in doc[1].lower().split():
                    if word in vocab:
                        ref_related.append(doc)
                        ref_times.append(time)
                        break
            self.game_vectors.append(list(zip(ref_times, self.tf_vectorizer.fit_transform(ref_related), ref_related)))

    def loop_kmeans(self):
        """
        """
        game_self.clusters = []
        for game in game_vectors:
            self.clusters = k_means(time_stamped_vectors)


    def get_cluster_docs(self):
        """
        """
        docs = []
        for num, center in enumerate(self.clusters):
            times = [pt[0] for pt in self.clusters[center]]
            docs.append(' '.join([pt[2] for pt in self.clusters[center]]))
            if self.verbose:
                c_vec = self._to_csr(center[1]).todense().argsort()
                features = self.tf_vectorizer.get_feature_names()
                f_list = np.array(c_vec[:,-1:-11:-1])[0].tolist()
                top_words = [features[i] for i in f_list]
                print('Centroid:', num)
                print('Time:', center[0])
                print('Top Words:', top_words)
            if self.print_figs:
                plt.scatter(center[0], 0)
                plt.hist(times)
        if self.print_figs:
            plt.show()

    def fit(self, documents):
        """
        """
        self.documents = documents
        self.add_tf_vectors()
        self.k_means



if __name__ == '__main__':
    pickle_path = '../ref_analysis/working.pkl'
    with open('../ref_analysis/data/manual_vocab.csv') as f:
        vocab = np.array([word.strip() for word in f])

    clusterer = CommentClusterer(vocab)

    documents = load_data(pickle_path=pickle_path)
    clusterer.fit(documents)
