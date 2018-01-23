import sqlite3, os, pickle, random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.spatial.distance import euclidean, cosine
from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import dendrogram

from matplotlib import pyplot as plt

from LemmaTokenizer import LemmaTokenizer
from library import *


class CommentClusterer(object):
    """
    """

    def __init__(self, class_tags, vocab=None, distance=cosine, \
                max_iter=100, time_scale_factor=1.05, threshold=0.1, \
                verbose=True, print_figs=False, max_k=3, min_comments=100,
                cluster_method='hierarchical'):

        if min_comments < max_k:
            raise ValueError('Minimum comments must be greater than maximum k!')

        self.class_tags = class_tags

        self.print_figs = print_figs
        self.verbose = verbose

        self.cluster_method = cluster_method

        self.time_scale_factor = time_scale_factor
        self.distance = distance
        self.threshold = threshold

        self.max_iter = max_iter
        self.max_k = max_k
        self.min_comments = min_comments

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

    def _convert_cluster_to_dict(self, comments, cluster_number):
        """
        Takes lists of clusters and their labels and converts them
        to a dictionary of cluster_number: [points].
        """
        out = defaultdict(list)
        for comment, cluster_number in zip(comments, cluster_number):
            out[cluster_number].append(comment)
        return out


    def _get_silhouette_score(self, clusters, distances=None):
        """
        Determines sil_score for clustering
        """
        # Careful with names here: points is the points from the clusters and data is the combined points
        # Label the points in each cluster and add all of the points to a single list for distance calculation
        if self.cluster_method == 'kmeans':
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
                        abs(p1[1].created_utc - p2[1].created_utc) ** \
                        self.time_scale_factor for p2 in combined_points]
                        for p1 in combined_points]

        elif self.cluster_method == 'hierarchical':
            labels = clusters.labels_
        else:
            raise ValueError('Unknown clustering method!')

        return silhouette_score(distances, labels, metric="precomputed")

    def _compute_label_distance(self, pt1, pt2):
        """
        """

        # These similarity scores could potentially be determined from the
        # data by finding the class that are frequently in clusters together.

        time_distance = abs(pt1[1].created_utc - pt2[1].created_utc)
        tag_distance = 0
        tags1 = set(self.class_tags[np.where(pt1[2].astype(int)==1)])
        tags2 = set(self.class_tags[np.where(pt2[2].astype(int)==1)])

        for tag1 in tags1:
            if tag1 == 'M':
                if 'M' in tags2:
                    tags2.remove('M')
                else:
                    tag_distance += 5
            elif tag1 == '9': # False start
                if '9' in tags2:
                    tags2.remove('9')
                if '10' in tags2: # Offsides
                    tag_distance += 5
                    if '10' not in tags1: # Doesn't match backwards
                        tag_distance += 5
                        tags2.remove('10')
                if '11' in tags2:
                    tag_distance += 5
                    if '11' not in tags1: # Doesn't match backwards
                        tag_distance += 5
                        tags2.remove('11')

        # Anything left over
        for left in tags1.symmetric_difference(tags2):
            if left not in '1,2,3,4':
                tag_distance += 30



        return (time_distance ** self.time_scale_factor) + tag_distance


    def _hierachical_clustering(self, k, distance_matrix):

        agger = AgglomerativeClustering(n_clusters=k, affinity='precomputed',\
                    memory='_cache/', compute_full_tree='auto', linkage='average') # memory should speed this up?
        agger.fit(distance_matrix)
        return agger


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
                            + abs(comment.created_utc - center[1]) ** \
                            self.time_scale_factor) for center in centers]
                else:
                    # Always euclidean for time. TF distance defined in object
                    distances = [(self.distance(tfs.todense(), center[0].todense())\
                            + abs(comment.created_utc - center[1]) ** \
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
                        center[0].todense()) + abs(new_center[1] - \
                        center[1]) ** self.time_scale_factor
            centers = new_centers
            if True: #self.verbose:
                print('Iteration:', i)
                print('Centroid movement:', dist)
            if dist < self.threshold:
                if False: #self.verbose:
                    print('Converged!')
                break

        return clusters

    def _loop_clustering(self):
        """
        """
        # Can't fit if there are fewer clusters than data points
        # Probably don't want to put much stock into games with fewer than
        if len(self.game_vector) <= self.min_comments:
            print('Very few commments. Skipping.')
            return True

        if self.cluster_method == 'hierarchical':
            distances = [[self._compute_label_distance(com_i, com_j) for com_i in self.game_vector] for com_j in self.game_vector]
        #print(distances[-1])
        for i in range(2, self.max_k): # Is there potential for multithreading here?
            if self.cluster_method == 'kmeans':
                clusters = self._k_means(i)
                sil_score = self._get_silhouette_score(clusters)
                self.scored_clusters.append((sil_score, clusters, i))
            elif self.cluster_method == 'hierarchical':
                clusters = self._hierachical_clustering(i, distances)
                sil_score = self._get_silhouette_score(clusters, distances)
                if self.print_figs:
                    self.plot_dendrogram(clusters)
                self.scored_clusters.append((sil_score, self._convert_cluster_to_dict(self.game_vector, clusters.labels_), i))
            # Add the silhouette score and clustering to a list
        #print(len(self.scored_clusters))
        # Find the k with the lowest silhouette_score and add the clusters
        # to a list. len(clusters) will give k, so its not stored)
        self.scored_clusters.sort(reverse=True)
        if self.verbose:
            print('Best Silhouette Score: {:.3f}'.format(self.scored_clusters[-1][0]))
            print('K: {}'.format(len(self.scored_clusters[-1][1])))

    def print_silhouette_plot(self):
        """
        """
        pass

    def plot_dendrogram(self, clusters):
        """
        Adapted from scipy documentation.
        """
        if not self.cluster_method == 'hierarchical':
            print('Cannot generate tree for k-means clustering!')
            return

        # Children of hierarchical clustering
        children = clusters.children_
        # Distances between each pair of children
        # Since we don't have this information, we can use a uniform one for plotting
        distance = np.arange(children.shape[0])
        # The number of observations contained in each cluster level
        no_of_observations = np.arange(2, children.shape[0]+2)
        # Create linkage matrix and then plot the dendrogram
        linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, labels=clusters.labels_)
        plt.show()

    def print_clusters(self, only_best=True):
        """
        """
        if self.cluster_method == 'hierarchical':
            return
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

        if self._loop_clustering():
            return True
        if self.verbose or self.print_figs:
            self.print_clusters()



if __name__ == '__main__':
    pickle_path = '../ref_analysis/full.pkl'
    with open('../ref_analysis/data/manual_vocab.csv') as f:
        vocab = [word.strip() for word in f]
    with open('../ref_analysis/data/common-english-words.csv') as f:
        stop_words = [word.strip() for word in f]
    #print(stop_words)
    documents = load_data(pickle_path=pickle_path, n_games=None, overwrite=True)
    #print(len(documents))
    clusterer = CommentClusterer(vocab=vocab, stop_words=stop_words, time_scale_factor=1.0, print_figs=True, ngram_range=(1,3))
    clusterer.fit(documents)

    grouped_docs = clusterer.get_cluster_docs()
    model = do_LDA(grouped_docs, n_features=5000, n_components=20, stop_words=stop_words, ngram_range=(1,5))

    pickle.dump(model, open('lda_model.pkl', 'wb'))
