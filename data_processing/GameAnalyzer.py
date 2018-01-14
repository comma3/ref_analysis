import sqlite3
import pickle, os
from time import time
import numpy as np
import pandas as pd
import scipy.stats as stats

import praw

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from library import load_data, collect_game_threads
from CommentClusterer import CommentClusterer

# Want to return something of the format:
# team, rule, incorrect, strength, id
# UNC, offsides, yes, 10, off001

class GameAnalyzer(object):
    """
    Takes a cluster of comments and determines
    1. the suggested rule,
    2. the consesus on the ruling (True/False)
    3. the strenght of the feelings about the call (sentiment)
    4. the team affected
    5. counts of team affiliations and whether they were on the winning or
        losing side.
    """

    def __init__(self, game_id, game_thread, home, away, winner):
        self.game_id = game_id
        self.game_thread = game_thread
        self.home = home
        self.away = away
        self.winner = winner

        self.home_fans = defaultdict(list)
        self.away_fans = defaultdict(list)
        self.unaffiliated_fans = defaultdict(list)

        self.against_home = 0
        self.against_away = 0

        self.clusterer = None # Actual clusters are in clusterer.scored_clusters

        self.lda = None

        self._load_data()
        self._get_user_scores_by_affiliation()
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

        for comment in self.documents:
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
                for f in comment.author_flair_text.split('/'):

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
        """

        self.documents = load_data(self.game_thread)

    def find_clusters(self):
        """
        """
        self.clusterer = CommentClusterer(vocab=vocab, stop_words=stop_words, \
                        time_scale_factor=0.1, print_figs=True, ngram_range=(1,3))
        self.clusterer.fit(self.documents)

    def make_silhouette_plot(self):
        """
        """
        pass



if __name__ == '__main__':

    with open('../ref_analysis/data/manual_vocab.csv') as f:
        vocab = [word.strip() for word in f]
    with open('../ref_analysis/data/common-english-words.csv') as f:
        stop_words = [word.strip() for word in f]
    #print(stop_words)

    game_list = collect_game_threads()

    print('Number of games in DB: {}'.format(len(game_list)))

    for n, game in enumerate(game_list):
        print(game)
        game_id, game_thread, home, away, winner = game
        analyzer = GameAnalyzer(game_id, game_thread, home, away, winner)
        analyzer.find_clusters()

    grouped_docs = clusterer.get_cluster_docs()
    model = do_LDA(grouped_docs, n_features=5000, n_components=20, stop_words=stop_words, ngram_range=(1,5))

    pickle.dump(model, open('lda_model.pkl', 'wb'))

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
