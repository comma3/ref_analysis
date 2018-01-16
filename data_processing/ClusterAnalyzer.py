import sqlite3
import pickle, os
from time import time
import numpy as np
import pandas as pd
import scipy.stats as stats

import praw

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


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

    def __init__(self, comments, model, home, away):

        self.comments = comments
        self.model = model
        self.home = home
        self.away = away

        self.user_dist = defaultdict(int) # d[team] : # unique poseters
        self.home_scores = [] # votes[home] :
        self.away_scores = []
        self.unaffiliated_scores = defaultdict(list)

        self.team_affected = None
        self.rule = None
        self.bad_call = None
        self.egregiousness = 0


        for comment in self.comments:
            if comment.author_flair_text:
                [self.user_dist[f] += 1 for f in comment.author_flair_text.split('/')]



    def _collect_votes(self):
        """
        """
        for comment in self.comments:
            if comment.author_flair_text:
                if self.home.lower() in comment.author_flair_text.lower():
                    self.home_scores.append((self.model.predict(comment.body), \
                                comment.score, self._mention_team(comment.body)))
                elif self.away.lower() in comment.author_flair_text.lower():
                    self.away_scores.append(comment.score)
                else:
                    self.unaffiliated_scores.append((self.model.predict(comment.body), \
                                comment.score, self._mention_team(comment.body)))
            else:
                self.unaffiliated_scores.append((self.model.predict(comment.body), \
                            comment.score, self._mention_team(comment.body)))

    def predict(self):
        """
        """

        # Assign comments as suggesting bad call
        # then find the votes for each of these calls
        home_bad_call = 0
        home_disagree = 0
        home_total = 0
        away_positive = 0
        away_negative = 0
        away_total = 0
        for category, score, mentioned in self.home_scores:
            if category == 'D':
                home_disagree += score
            if category == 'E':
            # Should add something like len(categories) > 1 (discussion of multiple fouls is likely an excuse)
            # Sort of implicit admission
                home_positive += 1
                # Probably want to use these votes for something
            if category == 'S':
                away_bad_call += 1
            else:
                home_negative += 1
            home_total += score



    def _mention_team(self, comment):
        """
        """

        for word in comment.body.lower():
            if word in self.home:
                return self.home
            elif word in self.away:
                return self.away
            elif word == 'we' or word == 'us':
                if self.home in comment.author_flair_text.lower():
                    return self.home
                elif self.away in comment.author_flair_text.lower()
                    return self.away
            elif word in nicknames_dict[self.home]:
                return self.home
            elif word in nicknames_dict[self.away]:
                return self.away
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
