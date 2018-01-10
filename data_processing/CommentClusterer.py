import sqlite3, os, pickle, random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score

from scipy.spatial.distance import euclidean, cosine
from scipy.sparse import csr_matrix

from library import load_data
from lda import do_LDA, do_NMF


class CommentClusterer(object):
    """
    """

    def __init__(self, vocab=None, vectorizer='count', distance=euclidean, \
                k=10,max_iter=100, time_scale_factor=0.001, threshold=0.1, \
                verbose=True, print_figs=False):

        self.print_figs = print_figs
        self.verbose = verbose
        self.vectorizer = vectorizer

        self.distance=distance
        self.vocab = vocab
        self.k = k
        self.max_iter = max_iter
        self.time_scale_factor = time_scale_factor
        self.threshold = threshold

        self.game_clusters = []
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


    def _k_means(self, game_vector):
        """Performs custom k means

        Args:
        - game - feature matrix of (timestamp, tf_vector)
        - k - number of clusters
        - max_iter - maximum iterations

        Returns:
        - clusters - dict mapping cluster centers to observations
        """

        game_vector = list(game_vector) # zip iterator gets used up

        try:
            centers = [tuple(pt) for pt in random.sample(game_vector, self.k)]
        except ValueError:
            return
        for i in range(self.max_iter):
            clusters = defaultdict(list)
            for time, tfs, doc in game_vector:
                distances = [(self.distance(tfs.todense(), center[1].todense())\
                            + self.distance(time, center[0]) * \
                            self.time_scale_factor) for center in centers]
                center = centers[np.argmin(distances)]
                center = center[0], self._make_hashable(center[1])
                clusters[center].append((time,tfs,doc))
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
                dist += self.distance(new_center[1].todense(), \
                        center[1].todense()) + self.distance(new_center[0], \
                        center[0]) * self.time_scale_factor
            centers = new_centers
            if self.verbose:
                print('Iteration:', i)
                print('Centroid movement:', dist)
                if dist < self.threshold:
                    print('Converged!')
                    break

        return clusters

    def add_tf_vectors(self):
        """
        Adds term frequency vector to game documents.

        Optionally will filter only officiating related comments if a vocabulary
        is provided.
        """

        for game in self.documents:
            if self.vectorizer == 'count':
                tf_vectorizer = CountVectorizer()
            else:
                tf_vectorizer = TfidfVectorizer()

            ref_related = []
            ref_times = []
            for time, doc in game:
                for word in doc.lower().split():
                    if any(self.vocab):
                        if word in self.vocab:
                            ref_related.append(doc)
                            ref_times.append(time)
                            break
                    else:
                        ref_related.append(doc)
                        ref_times.append(time)
            try:
                self.game_vectors.append((zip(ref_times, tf_vectorizer.fit_transform(ref_related), ref_related), tf_vectorizer))
            except ValueError:
                print(game)
                print('Game had no comments that passed vocab filter!')

    def loop_k_means(self):
        """
        """

        for game_vector, tf_vectorizer in self.game_vectors:
            clusters = self._k_means(game_vector)
            if clusters:
                self.game_clusters.append((clusters, tf_vectorizer))

    def print_clusters(self):
        """
        """
        game_num = 0
        for clusters, tf_vectorizer in self.game_clusters:
            for num, center in enumerate(clusters):
                times = [pt[0] for pt in clusters[center]]
                if self.verbose:
                    c_vec = self._to_csr(center[1]).todense().argsort()
                    features = tf_vectorizer.get_feature_names()
                    f_list = np.array(c_vec[:,-1:-11:-1])[0].tolist()
                    top_words = [features[i] for i in f_list]
                    print('Game: {}\tCentroid:{}'.format(game_num, num))
                    print('Time:', center[0])
                    print('Top Words:', top_words)
                if self.print_figs:
                    plt.scatter(center[0], 0)
                    plt.hist(times)
            if self.print_figs:
                plt.show()
            game_num += 1

    def get_cluster_docs(self):
        """
        Combines all of the documents in a cluster into a single document,
        with the goal of having a single document for each call
        """
        docs = []
        for clusters, _ in self.game_clusters:
            for num, center in enumerate(clusters):
                docs.append(' '.join([pt[2] for pt in clusters[center]]))
        return docs

    def fit(self, documents):
        """
        """
        self.documents = documents
        self.add_tf_vectors()
        self.loop_k_means()
        if self.verbose or self.print_figs:
            self.print_clusters()



if __name__ == '__main__':
    pickle_path = '../ref_analysis/full.pkl'
    with open('../ref_analysis/data/manual_vocab.csv') as f:
        vocab = np.array([word.strip() for word in f])
    documents = load_data(pickle_path=pickle_path, n_games=10)
    clusterer = CommentClusterer(vocab=vocab)
    clusterer.fit(documents)

    grouped_docs = clusterer.get_cluster_docs()
    do_LDA(grouped_docs, n_features=500, n_components=30)
