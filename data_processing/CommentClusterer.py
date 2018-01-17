import sqlite3, os, pickle, random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score

from scipy.spatial.distance import euclidean, cosine
from scipy.sparse import csr_matrix

from LemmaTokenizer import LemmaTokenizer
from library import *
from lda import do_LDA, do_NMF


class CommentClusterer(object):
    """
    """

    def __init__(self, vocab=None, vectorizer='tfidf', distance=euclidean, \
                max_iter=100, time_scale_factor=0.005, threshold=0.1, \
                verbose=True, print_figs=False, stop_words='english', \
                ngram_range=(1,1), tokenizer=LemmaTokenizer()):

        self.print_figs = print_figs
        self.verbose = verbose

        self.vectorizer = vectorizer
        self.tokenizer = tokenizer

        self.ngram_range = ngram_range

        self.distance=distance
        self.vocab = vocab
        self.stop_words=stop_words
        self.max_iter = max_iter
        self.time_scale_factor = time_scale_factor
        self.threshold = threshold

        self.game_clusters = []
        self.game_vector = []
        self.documents = None
        self.flairs = set()

        # Store (scored_clusters, corresponding clusters)
        self.scored_clusters = []

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


    def _add_tf_vectors(self):
        """
        Adds term frequency vector to game documents.

        Optionally will filter only officiating related comments if a vocabulary is provided.
        """

        if self.vectorizer.lower() == 'count':
            self.tf_vectorizer = CountVectorizer(stop_words=self.stop_words, tokenizer=self.tokenizer)
        else:
            self.tf_vectorizer = TfidfVectorizer(stop_words=self.stop_words, tokenizer=self.tokenizer)

        ref_related = []
        if any(self.vocab):
            for comment in self.documents:
                for word in comment.body.lower().split():
                    if word in self.vocab:
                        ref_related.append(comment)
                        break
        else:
            ref_related = [comment for comment in self.documents]

        # Fit the vectorizer using all of the words in a game_documents
        #print([comment.body for comment in ref_related])
        try:
            self.tf_vectorizer.fit([comment.body for comment in ref_related])
        except ValueError:
            # Game didn't contain any ref related events
            return None
        tfs_and_comments = []
        for comment in ref_related:
            # Transform wants a list
            tfs_and_comments.append((self.tf_vectorizer.transform([comment.body]),
                                comment))

        # Save the comment objects, tf vectors and vectorizer by game
        self.game_vector = tfs_and_comments


    def _get_silhouette_score(self, clusters):
        """
        Determines sil_score for clustering
        """
        # Careful with names here: points is the points from the clusters and data is the combined points
        # Label the points in each cluster and add all of the points to a single list for distance calculation
        labels = []
        combined_points = []
        for label, cluster_points in enumerate(clusters.values()):
            for cluster_point in cluster_points:
                combined_points.append(cluster_point)
                labels.append(label)

        # I bet there is a np way to do this better
        # Seems decently fast though...
        # Needs to be reculculated for every k because the points get shuffled
        # by kmeans so the distance matrix will change with k.
        distances = [[self.distance(p1[0].todense(), p2[0].todense()) + \
                    self.distance(p1[1].created_utc, p2[1].created_utc) * \
                    self.time_scale_factor for p2 in combined_points]
                    for p1 in combined_points]

        return silhouette_score(distances, labels, metric="precomputed")


    def label_distance(self, test, center):
        """
        """
        for a, b in zip(test, center):
            pass


    def _k_means(self, k, dist_type='old'):
        """
        Performs custom k means clustering

        Args:
        - game - feature matrix of (tf_vector, praw comment object)
        - k - number of clusters
        - max_iter - maximum iterations

        Returns:
        - clusters - dict mapping cluster centers to observations
        """

            # Initialize centers and get time from comment -> needs to be float
        centers = [(pt[0], pt[1].created_utc, pt[2]) for pt in random.sample(self.game_vector, k)]


        for i in range(self.max_iter):
            clusters = defaultdict(list)
            # Calculate the distance of each point to each center and assignment
            # the point to the cluster with that center.
            for tfs, comment, labels in self.game_vector:
                if dist_type == 'old':
                    distances = [(self.distance(tfs.todense(), center[0].todense())\
                            + self.distance(comment.created_utc, center[1]) * \
                            self.time_scale_factor) for center in centers]
                else:
                    # Maybe cosine similarity makes sense?
                    distances = [(self.distance(tfs.todense(), center[0].todense())\
                            + self.distance(comment.created_utc, center[1]) * \
                            self.time_scale_factor) for center in centers]
                # Determine which center is closest
                center = centers[np.argmin(distances)]
                # Collect the data for the center to use as the key
                center_key = self._make_hashable(center[0]), center[1]
                # Add point to the center that is closest
                clusters[center_key].append((tfs, comment, labels))
            new_centers = []
            # Reculaate the centers of the clusters based on the points assigned
            # to each cluster
            for pts in clusters.values():
                time = 0
                matrix = np.zeros(pts[0][0].shape[1]).T
                for pt in pts:
                    time += pt[1].created_utc
                    matrix += pt[0]
                new_center = csr_matrix(matrix/len(pts)), time/len(pts)
                new_centers.append(tuple(new_center))
            dist = 0
            for center, new_center in zip(centers, new_centers):
                dist += self.distance(new_center[0].todense(), \
                        center[0].todense()) + self.distance(new_center[1], \
                        center[1]) * self.time_scale_factor
            centers = new_centers
            if True: #self.verbose:
                print('Iteration:', i)
                print('Centroid movement:', dist)
            if dist < self.threshold:
                if False: #self.verbose:
                    print('Converged!')
                break

        return clusters

    def _loop_k_means(self, max_k=3):
        """
        """
        # Can't fit if there are fewer clusters than data points
        # Probably don't want to put much stock into games with fewer than
        # max_k comments anyway
        if len(self.game_vector) <= max_k:
            print('Very few commments. Skipping.')
            return True
        for i in range(2, max_k): # Is there potential for multithreading here?
            clusters = self._k_means(k=i)
            sil_score = self._get_silhouette_score(clusters)
            # Add the silhouette score and clustering to a list
            self.scored_clusters.append((sil_score, clusters, i))
        # Find the k with the lowest silhouette_score and add the clusters
        # to a list. len(clusters) will give k, so its not stored)
        self.scored_clusters.sort(reverse=True)
        if self.verbose:
            print('Best Silhouette Score: {:.3f}'.format(self.scored_clusters[-1][0]))

    def print_silhouette_plot(self):
        """
        """
        pass

    def print_clusters(self, only_best=True):
        """
        """

        for score, clusters, k in self.scored_clusters:
            print('K = {}'.format(len(clusters)))
            for num, center in enumerate(clusters):
                times = [pt[1].created_utc for pt in clusters[center]]
                #if self.verbose:
                    # c_vec = self._to_csr(center[0]).todense().argsort()
                    # features = self.tf_vectorizer.get_feature_names()
                    # f_list = np.array(c_vec[:,-1:-11:-1])[0].tolist()
                    # top_words = [features[i] for i in f_list]
                    # print('Centroid Number: {}'.format(num))
                    # print('Time:', center[1])
                    # print('Top Words:', top_words)
                if self.print_figs:
                    plt.scatter(center[1], 0)
                    plt.hist(times)
                    plt.title('K: {}'.format(k))
            if self.print_figs:
                plt.show()
            if only_best:
                break

    def get_combined_cluster_docs(self, cluster='best'):
        """
        Combines all of the documents in a cluster into a single document,
        with the goal of having a single document for each cluster (i.e., call)
        """
        # Join all of the comments in a cluster into a single string and return
        # a list of these strings for each cluster
        if cluster=='best':
            return [' '.join([pt[1].body for pt in center]) \
                    for center in self.scored_clusters[0][1].values()]
        else:
            return [' '.join([pt[1].body for pt in center]) \
                    for center in self.scored_clusters[cluster][1].values()]

    def fit(self, tf_vectors, comments, labels):
        """
        """
        #self.documents = documents
        #self._add_tf_vectors()
        self.game_vector = list(zip(tf_vectors, comments, labels))

        if self._loop_k_means():
            return True
        if self.verbose or self.print_figs:
            self.print_clusters()

    def refit(self):
        """
        Using the LDA generated from the rough fit, perform a better fit with
        more clear classifications.
        """
        pass



if __name__ == '__main__':
    pickle_path = '../ref_analysis/full.pkl'
    with open('../ref_analysis/data/manual_vocab.csv') as f:
        vocab = [word.strip() for word in f]
    with open('../ref_analysis/data/common-english-words.csv') as f:
        stop_words = [word.strip() for word in f]
    #print(stop_words)
    documents = load_data(pickle_path=pickle_path, n_games=None, overwrite=True)
    print(len(documents))
    clusterer = CommentClusterer(vocab=vocab, stop_words=stop_words, time_scale_factor=0.1, print_figs=True, ngram_range=(1,3))
    clusterer.fit(documents)

    grouped_docs = clusterer.get_cluster_docs()
    model = do_LDA(grouped_docs, n_features=5000, n_components=20, stop_words=stop_words, ngram_range=(1,5))

    pickle.dump(model, open('lda_model.pkl', 'wb'))
