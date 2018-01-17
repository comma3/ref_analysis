import sqlite3, pickle, os
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd

import scipy.stats as stats

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import praw

# Want to return something of the format:
# team, rule, incorrect, strength, id
# UNC, offsides, yes, 10, off001

class ClusterAnalyzer(object):
    """
    Takes a cluster of comments and determines
    1. the suggested rule,
    2. the consesus on the ruling (True/False)
    3. the strenght of the feelings about the call (sentiment)
    4. the team affected
    5. counts of team affiliations and whether they were on the winning or
        losing side.
    """

    def __init__(self, cluster, home, away, class_labels):

        self.cluster = cluster
        self.home = home
        self.away = away
        self.class_labels = class_labels

        self.nicknames_dict = self._load_nicknames()

        self.user_dist = defaultdict(int) # d[team] : # unique poseters
        self.home_scores = []
        self.away_scores = []
        self.unaffiliated_scores = []

        self.team_affected = None
        self.rule = None


        self._collect_votes()

        # for comment in self.comments:
        #     if comment.author_flair_text:
        #         for f in comment.author_flair_text.split('/'):
        #             self.user_dist[f] += 1

    def _load_nicknames(self):
        """
        """
        return defaultdict(list)


    def _collect_votes(self):
        """
        """
        #print(self.cluster)
        for tf, comment, labels in self.cluster:
            if comment.author_flair_text:
                if self.home.lower() in comment.author_flair_text.lower():
                    self.home_scores.append((labels, comment.score,\
                                            self._mention_team(comment)))
                elif self.away.lower() in comment.author_flair_text.lower():
                    self.away_scores.append((labels, comment.score,\
                                            self._mention_team(comment)))
                else:
                    self.unaffiliated_scores.append((labels, comment.score,\
                                            self._mention_team(comment)))
            else:
                self.unaffiliated_scores.append((labels, comment.score,\
                                            self._mention_team(comment)))

    def predict(self):
        """
        """
        bad_call = {
                    'home' : 0,
                    'away' : 0
                    }

        bad_call_scores =   {
                            'home' : 0,
                            'away' : 0
                            }

        print(len(self.home_scores))
        print(len(self.away_scores))
        print(len(self.unaffiliated_scores))
        if self.home_scores:
            class_totals = np.zeros(self.home_scores[0][0].shape)
        elif self.away_scores:
            class_totals = np.zeros(self.away_scores[0][0].shape)
        else:
            class_totals = np.zeros(self.unaffiliated_scores[0][0].shape)

        print(class_totals)

        # for key, score_list in [('home', self.home_scores), ('away', self.away_scores), ('', self.unaffiliated_scores)]
        if self.home_scores:
            for comment_classes, score, mentioned in self.home_scores:
                class_totals += comment_classes
                if comment_classes[np.where(self.class_labels == 'D')]:
                    # We are going to skip these for now. Will apply their effects
                    # when we have a better idea of which team got the penalty
                    continue
                if comment_classes[np.where(self.class_labels == 'S')]:
                    bad_call['away'] += 1
                    bad_call_scores['away'] += score
                elif comment_classes[np.where(self.class_labels == 'E')]:
                    bad_call['away'] += 1
                    bad_call_scores['away'] -= score
                else:
                    bad_call['home'] += 1
                    bad_call_scores['home'] += score

        print(self.class_labels[np.argsort(class_totals)])


    def _mention_team(self, comment):
        """
        """
        teams = set()
        we = set()
        for word in comment.body.lower():
            if word in self.home:
                teams.add(self.home)
            if word in self.away:
                teams.add(self.away)
            elif word == 'we' or word == 'us':
                if self.home in comment.author_flair_text.lower():
                    we.add(self.home)
                elif self.away in comment.author_flair_text.lower():
                    we.add(self.away)
            elif word in self.nicknames_dict[self.home]:
                teams.add(self.home)
            elif word in self.nicknames_dict[self.away]:
                teams.add(self.away)
        return None

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
