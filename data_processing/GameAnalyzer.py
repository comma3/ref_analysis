import sqlite3, pickle, os
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import praw

from library import *
from CommentClusterer import CommentClusterer
from lda import do_LDA

# Want to return something of the format:
# team, rule, incorrect, strength, id
# UNC, offsides, yes, 10, off001

class GameAnalyzer(object):
    """

    """

    def __init__(self, model, vectorizer, game_id, game_thread, home, away, winner):

        self.model = model
        self.vectorizer = vectorizer

        self.game_id = game_id
        self.game_thread = game_thread
        self.home = home.lower()
        self.away = away.lower()
        self.winner = winner.lower()

        self.home_fans = defaultdict(list)
        self.away_fans = defaultdict(list)
        self.unaffiliated_fans = defaultdict(list)

        self.against_home = 0
        self.against_away = 0

        self.clusterer = None # Actual clusters are in clusterer.scored_clusters

        self.comments = None
        self._load_data() # loads comments
        #self._get_user_scores_by_affiliation()
        self._get_flair_set() # Maybe combine these two.

    def _get_flair_set(self):
        """
        Updates pickle containing a set of all flairs found for this sport.
        """

        pickle_path = 'cfb_flairs.pkl'
        if os.path.isfile(pickle_path):
            flairs = pickle.load(open(pickle_path, 'rb'))
        else:
            flairs = set()

        for comment in self.comments:
            if comment.author_flair_text:
                fs = comment.author_flair_text.split('/')
            else: # No flair
                continue
            [flairs.add(f.strip().lower()) for f in fs]
        # This will only store flair list if it doesn't exist.
        # Change the behavior of this whole thing in the future
        pickle.dump(flairs, open(pickle_path, 'wb'))


    def _get_user_scores_by_affiliation(self):
        """
        """
        for comment in self.comments:
            if comment.author_flair_text:
                has_team_in_game = False
                flairs = comment.author_flair_text.lower().split('/')
                if len(flairs) > 2:
                    print(flairs)
                for f in flairs:
                    if self.home in f.lower():
                        self.home_fans[comment.author].append(comment.score)
                        has_team_in_game = True
                        break
                    elif self.away in f.lower():
                        self.away_fans[comment.author].append(comment.score)
                        has_team_in_game = True
                        break
                if not has_team_in_game:
                        self.unaffiliated_fans[comment.author].append(comment.score)
            else:
                self.unaffiliated_fans[comment.author].append(comment.score)

    def _load_data(self):
        """
        Either downloads the comments or stores them as a pickle

        Will change to also use a db when data format is finalized
        """
        self.comments = load_data(self.game_thread)

    def _clean_text(self, comment_text):
        """
        """
        if isinstance(comment_text,str):
            return self.vectorizer.transform([comment_text])
        else:
            return self.vectorizer.transform(comment_text)

    def find_clusters(self, **clusterer_params):
        """
        """
        self.clusterer = CommentClusterer(**clusterer_params)
        return self.clusterer.fit(self.comments)


    def make_silhouette_plot(self):
        """
        """
        pass


    def classify_comments(self):
        """
        """
        ref_related = []
        for comment in self.comments:
            print(self.model.predict(self._clean_text(comment.body)))



if __name__ == '__main__':

    with open('../ref_analysis/data/manual_vocab.csv') as f:
        vocab = [word.strip() for word in f]
    with open('../ref_analysis/data/common-english-words.csv') as f:
        stop_words = [word.strip() for word in f]
    #print(stop_words)

    game_list = collect_game_threads()
    num_games= len(game_list)
    print('Number of games in DB: {}'.format(num_games))
    model, vectorizer = get_MutliTargetModel(overwrite=True, alpha=0, fit_prior=True)
    print('Recall:')
    model.calc_recall()
    print('Precision:')
    model.calc_preciscion()
    print('Accuracy:')
    model.calc_accuracy()
    grouped_docs = []
    for n, game in enumerate(game_list):
        print('{:.1f}% Complete'.format(n/num_games))
        game_id, game_thread, home, away, winner = game
        analyzer = GameAnalyzer(model, vectorizer, game_id, game_thread, home, away, winner)
        analyzer.classify_comments()
        break

        # if analyzer.find_clusters(vocab=vocab, stop_words=stop_words, \
        #             time_scale_factor=0.1, print_figs=False, ngram_range=(1,3)):
        #     # Game didn't have enough comments
        #     continue
        #
        # #analyzer.clusterer.print_clusters()
        # grouped_docs.append(analyzer.clusterer.get_combined_cluster_docs())

        # k = len(analyzer.clusterer.scored_clusters[0][1])
        # model = do_LDA(grouped_docs, tf_features=500, lda_components=k, stop_words=stop_words, ngram_range=(1,5
    # grouped_path = 'grouped_clusters.pkl'
    # if not os.path.isfile(grouped_path):
    #     pickle.dump(grouped_docs, open(grouped_path, 'wb'))

    #pickle.dump(model, open('lda_model.pkl', 'wb'))

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
