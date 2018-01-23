import sqlite3, pickle, os
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd

import praw

from library import *
from CommentClusterer import CommentClusterer
from ClusterAnalyzer import ClusterAnalyzer


class GameAnalyzer(object):
    """

    """

    def __init__(self, model, game_id, game_thread, home, away, winner,
                    team_nickname_dict_path='team_list.csv', min_comments=500, \
                    squelch_errors=True, error_log_path='missing_games.txt', \
                    print_figs=False):

        self.min_comments = min_comments
        self.game_id = game_id
        self.game_thread = game_thread
        self.print_figs = print_figs
        self.squelch_errors = squelch_errors
        self.error_log_path = error_log_path
        self.no_errors = False

        self.comments = None
        self._load_data() # loads comments

        # High likelihood team's not in dict. Can't realluy trust data either
        # Don't have time for this so we just skip everything.
        if len(self.comments) > self.min_comments:
            self.team_nickname_dict = \
                                make_team_nickname_dict(team_nickname_dict_path)

            self.model = model # MultiTargetModel

            self.home = home
            self.away = away
            self.winner = winner
            self.no_errors = self._standardize_team_names()

            self.ref_mask = []

            self.home_fans = defaultdict(list)
            self.away_fans = defaultdict(list)
            self.unaffiliated_fans = defaultdict(list)

            self.call_list = []

            # Actual clusters are in clusterer.scored_clusters
            self.clusterer = None

            #self._get_user_scores_by_affiliation()
            self._get_flair_set() # Maybe combine these two.
        else:
            print('Too few comments for accurate analysis: {}'.format(
                                                            len(self.comments)))

    def _standardize_team_names(self):
        """
        Set team names to unique names from dictionary.
        Raise exception if name is not unqiue.
        """
        try:
            standard_home = self.team_nickname_dict[self.home.strip().lower()]
        except KeyError:
            if self.squelch_errors:
                with open(self.error_log_path, 'a') as elog:
                    elog.write("Home team not in dictionary {}. Opponent {}.\n".format(self.home, self.away))
                return False
            else:
                raise KeyError("Home team not in dictionary {}. Opponent {}.".format(self.home, self.away))

        try:
            standard_away = self.team_nickname_dict[self.away.strip().lower()]
        except KeyError:
            if self.squelch_errors:
                with open(self.error_log_path, 'a') as elog:
                    elog.write("Away team not in dictionary {}. Opponent {}.\n".format(self.away, self.home))
                return False
            else:
                raise KeyError("Away team not in dictionary {}. Opponent {}.".format(self.away, self.home))

        if len(standard_home) > 1:
            if self.squelch_errors:
                with open(self.error_log_path, 'a') as elog:
                    elog.write(("Home team name not unique. Received {} and" \
                    " gave {} with the away team {}.\n".format(self.home, \
                     standard_home, self.away)))
                return False
            else:
                raise ValueError("Home team name not unique. Received {} and gave {} with the away team {}.".format(self.home, standard_home, self.away))

        if len(standard_away) > 1:
            if self.squelch_errors:
                with open(self.error_log_path, 'a') as elog:
                    elog.write("Away team name not unique. Received {} and gave {} with the home team {}.\n".format(self.away, standard_away, self.home))
                return False
            else:
                raise ValueError("Away team name not unique. Received {} and gave {} with the home team {}.".format(self.away, standard_away, self.home))

        self.home = [x for x in standard_home][0] # Should only be one thing
        self.away = [x for x in standard_away][0]

        if self.winner.lower() != 'none' and self.winner: # Can be none if we didn't find a postgame thread.
            try:
                standard_winner = self.team_nickname_dict[self.winner.strip().lower()]
            except KeyError:
                if self.squelch_errors:
                    with open(self.error_log_path, 'a') as elog:
                        elog.write("Winner not in dictionary {}.\n".format(self.winner))
                    return False
                else:
                    raise KeyError("Winner not in dictionary {}.".format(self.winner))

            if len(standard_winner) > 1:
                if self.squelch_errors:
                    with open(self.error_log_path, 'a') as elog:
                        elog.write("Winner team name not unique. Received {} for winner given {}.\n".format(standard_winner, self.winner))
                    return False
                else:
                    raise ValueError("Winner team name not unique. Received {} for winner given {}.".format(standard_winner, self.winner))
            self.winner = [x for x in standard_winner][0]

        return True


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
            (flairs.add(f.strip().lower()) for f in fs)
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
                    raise ValueError("How'd you get extra flairs?")
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

    def _process_text(self, comment_text=None):
        """
        Clean and vectorize
        """
        # Clean a single text string
        if isinstance(comment_text, str):
            return self.model.vectorizer.transform([sub_home_away(comment_text, self.home, self.away, self.team_nickname_dict)])
        else: # Clean whole list of comments
            comments = np.array([sub_home_away(comment.body, self.home, self.away, self.team_nickname_dict) for comment in self.comments])
            return self.model.vectorizer.transform(comments)

    def classify_comments(self):
        """
        """
        if len(self.comments) > self.min_comments and self.no_errors:
            self.class_labels = self.model.classifier.predict(self._process_text())
            self.filter_comments()

    def filter_comments(self):
        """
        """
        self.ref_mask = np.where([1 if any(row[1:]==1) else 0 for row in self.class_labels])
        self.ref_labels = self.class_labels[self.ref_mask]
        self.ref_tfvectors = self._process_text()[self.ref_mask]
        self.ref_times = np.array([comment for comment in self.comments])[self.ref_mask]
        #print(self.ref_labels.shape)
        #print(self.ref_tfvectors.shape)
        #print(self.ref_times.shape)

    def find_isolated(self):
        """
        Find comments that are far away and unlikely to get their own cluster.

        Should be useful for finding whiners. Might improve clustering to remove them.
        """
        pass

    def find_clusters(self, **clusterer_params):
        """
        """
        if len(self.comments) > self.min_comments and self.no_errors:
            # min comments here should be different as this is already filtred
            # comments
            self.clusterer = CommentClusterer(self.model.target_classes, \
                                                            **clusterer_params)
            return self.clusterer.fit(self.ref_tfvectors, self.ref_times, \
                                                self.ref_labels)

    def analyze_clusters(self):
        """
        """
        if len(self.comments) > self.min_comments and self.no_errors:
            #print(self.clusterer.scored_clusters[0][1])
            sil_score, clusters, k = self.clusterer.scored_clusters[-1] # take best

            for cluster in clusters.values(): # dict of center: [assoc. pts]
                call = ClusterAnalyzer(cluster, self.home, self.away, self.model.target_classes, self.team_nickname_dict)
                self.call_list.append(call.predict())

    def add_to_db(self):
        """
        """
        pass


if __name__ == '__main__':

    # Vocab list deprecated.
    # with open('../ref_analysis/data/manual_vocab.csv') as f:
    #     vocab = [word.strip() for word in f]
    with open('../ref_analysis/data/common-english-words.csv') as f:
        stop_words = [word.strip() for word in f]
    #print(stop_words)

    game_list = collect_game_threads()
    num_games= len(game_list)
    print('Number of games in DB: {}'.format(num_games))
    model = get_MultiTargetModel(overwrite=False, alpha=0, fit_prior=True)
    # print('Recall:')
    # model.calc_recall()
    # print('Precision:')
    # model.calc_preciscion()
    # print('Accuracy:')
    # model.calc_accuracy()
    for n, game in enumerate(game_list):
        print('{:.1f}% Complete'.format(n/num_games))
        game_id, game_thread, home, away, winner = game
        analyzer = GameAnalyzer(model, game_id, game_thread, home, away, winner)
        analyzer.classify_comments()
        if analyzer.find_clusters(time_scale_factor=1.5, print_figs=True, min_comments=25, max_k=10):
            #print("Game didn't have enough comments")
            # method returns True if it fails to cluster and prints it's own
            # message. It's mostly fine to skip such games.
            continue
        analyzer.analyze_clusters()
        print() # just adding some spacing to console


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
