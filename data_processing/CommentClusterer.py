import sqlite3, os, pickle, random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import silhouette_score

from scipy.spatial.distance import euclidean, cosine
from scipy.sparse import csr_matrix



from library import load_data, LemmaTokenizer
from lda import do_LDA, do_NMF

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class CommentClusterer(object):
    """
    """

    def __init__(self, vocab=None, vectorizer='count', distance=euclidean, \
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
        self.game_vectors = []
        self.documents = None
        self.flairs = set()

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

    def _get_flair_list(self):
        """
        Generates a set of the unique flairs found in the comment threads.
        """
        for game in self.documents:
            for comment in game:
                if comment.author_flair_text:
                    fs = comment.author_flair_text.split('/')
                else:
                    # No flair
                    continue
                [self.flairs.add(f.strip().lower()) for f in fs]

    def _add_tf_vectors(self):
        """
        Adds term frequency vector to game documents.

        Optionally will filter only officiating related comments if a vocabulary is provided.
        """
        i = 0
        for game in self.documents:
            if self.vectorizer.lower() == 'count':
                tf_vectorizer = CountVectorizer(stop_words=self.stop_words, tokenizer=self.tokenizer)
            else:
                tf_vectorizer = TfidfVectorizer(stop_words=self.stop_words, tokenizer=self.tokenizer)

            ref_related = []
            if any(self.vocab):
                for comment in game:
                    for word in comment.body.lower().split():
                        if word in self.vocab:
                            ref_related.append(comment)
                            break
            else:
                ref_related.append(comment)

            # Fit the vectorizer using all of the words in a game_documents
            #print([comment.body for comment in ref_related])
            try:
                tf_vectorizer.fit([comment.body for comment in ref_related])
            except ValueError:
                # Game didn't contain any ref related events
                i += 1
                continue
            tfs_and_comments = []
            for comment in ref_related:
                # Transform wants a list
                tfs_and_comments.append((tf_vectorizer.transform([comment.body]),
                                    comment))

            # Save the comment objects, tf vectors and vectorizer by game
            self.game_vectors.append((tfs_and_comments, tf_vectorizer))

        print('{} games did not have any events!'.format(i))

    def _get_silhouette_score(self, clusters):
        """
        Determines sil_score for clustering
        """
        # Careful with names here: points is the points from the clusters and data is the combined points
        # Label the points in each cluster and add all of the points to a single list for distance calculation
        labels = []
        combined_points = []
        tot_points = 0
        for label, cluster_points in enumerate(clusters.values()):
            i = len(cluster_points)
            tot_points += i
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


    def _k_means(self, game_vector, k):
        """Performs custom k means

        Args:
        - game - feature matrix of (tf_vector, praw comment object)
        - k - number of clusters
        - max_iter - maximum iterations

        Returns:
        - clusters - dict mapping cluster centers to observations
        """

        try:
            # Initialize centers and get time from comment
            centers = [(pt[0], pt[1].created_utc)  for pt in random.sample(game_vector, k)]
        except ValueError:
            return


        for i in range(self.max_iter):
            clusters = defaultdict(list)
            # Calculate the distance of each point to each center and assignment
            # the point to the cluster with that center.
            for tfs, comment in game_vector:
                distances = [(self.distance(tfs.todense(), center[0].todense())\
                            + self.distance(comment.created_utc, center[1]) * \
                            self.time_scale_factor) for center in centers]
                # Determine which center is closest
                center = centers[np.argmin(distances)]
                # Collect the data for the center to use as the key
                center_key = self._make_hashable(center[0]), center[1]
                # Add point to the center that is closest
                clusters[center_key].append((tfs, comment))
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
            if self.verbose:
                print('Iteration:', i)
                print('Centroid movement:', dist)
            if dist < self.threshold:
                if self.verbose:
                    print('Converged!')
                break

        return clusters



    def _loop_k_means(self, max_k=20):
        """
        """
        loops = 0
        for game_vector, tf_vectorizer in self.game_vectors:
            loops+=1
            # zip iterator gets used up, but we reuse this list many times
            game_vector = list(game_vector)
            sil_scores = []
            if len(game_vector) <= max_k:
                print('Very few commments. Skipping.')
                continue
            for i in range(2, max_k):
                clusters = self._k_means(game_vector, k=i)
                sil_score = self._get_silhouette_score(clusters)
                # Add the silhouette score and clustering to a list
                sil_scores.append((sil_score, clusters))
            # Find the k with the lowest silhouette_score and add the clusters
            # to a list. len(clusters) will give k, so its not stored)
            sil_scores.sort()
            try:
                self.game_clusters.append((sil_scores[-1][1], tf_vectorizer))
            except IndexError:
                print("\n\nwhy aren't there clusters?")
                print(len(sil_scores))
                continue # Avoid trying to print next line
            print('\n\nSilhouette Score: {:.3f}\n\n'.format(sil_scores[-1][0]))
            if loops > 5:
                break

    def print_clusters(self):
        """
        """
        game_num = 0
        for clusters, tf_vectorizer in self.game_clusters:
            for num, center in enumerate(clusters):
                times = [pt[1].created_utc for pt in clusters[center]]
                if self.verbose:
                    c_vec = self._to_csr(center[0]).todense().argsort()
                    features = tf_vectorizer.get_feature_names()
                    f_list = np.array(c_vec[:,-1:-11:-1])[0].tolist()
                    top_words = [features[i] for i in f_list]
                    print('Game: {}\tCentroid:{}'.format(game_num, num))
                    print('Time:', center[1])
                    print('Top Words:', top_words)
                if self.print_figs:
                    plt.scatter(center[1], 0)
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
            for center in clusters.values():
                for pt in center:
                    print(pt)
                    break
                docs.append(' '.join([pt[1].body for pt in center]))
        return docs

    def fit(self, documents):
        """
        """
        self.documents = documents
        self._get_flair_list()
        self._add_tf_vectors()
        self._loop_k_means()
        if self.verbose or self.print_figs:
            self.print_clusters()

    def refit(self):
        """
        Using the LDA generated from the rough fit, perform a better fit with
        more clear classifications.
        """
        pass



if __name__ == '__main__':
    pickle_path = '../ref_analysis/big_100.pkl'
    with open('../ref_analysis/data/manual_vocab.csv') as f:
        vocab = [word.strip() for word in f]
    with open('../ref_analysis/data/common-english-words.csv') as f:
        stop_words = [word.strip() for word in f]
    #print(stop_words)
    documents = load_data(pickle_path=pickle_path, n_games=100)
    clusterer = CommentClusterer(vocab=vocab, stop_words=stop_words, time_scale_factor=0.05, print_figs=True, ngram_range=(1,3))
    clusterer.fit(documents)

    grouped_docs = clusterer.get_cluster_docs()
    do_LDA(grouped_docs, n_features=1000, n_components=30, stop_words=stop_words, ngram_range=(1,5))
