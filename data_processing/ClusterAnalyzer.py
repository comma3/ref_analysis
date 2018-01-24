import sqlite3, pickle, os
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd

import scipy.stats as stats

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

import praw

from library import *

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

    def __init__(self, cluster, home, away, tags, team_nickname_dict):

        self.nicknames_dict = team_nickname_dict

        self.cluster = cluster
        self.home = home
        self.away = away

        self.user_dist = defaultdict(int) # d[team] : # unique poseters
        self.home_scores = []
        self.away_scores = []
        self.unaffiliated_scores = []

        self.argumentative = {
                        'home' : 0,
                        'away' : 0
                        }
        self.positive = {
                    'home' : 0,
                    'away' : 0
                    }

        self.whiny = {
                    'home' : 0,
                    'away' : 0
                    }
        self._collect_votes()

        self.class_tags = tags
        self.tag_counts = self._get_class_array()

        self.team_affected = None
        self.rule = None

    def _get_class_array(self):
        """
        Finds the size of possible tag array.
        """
        if self.home_scores:
            return np.zeros(self.home_scores[0][0].shape)
        elif self.away_scores:
            return np.zeros(self.away_scores[0][0].shape)
        elif self.unaffiliated_scores:
            return np.zeros(self.unaffiliated_scores[0][0].shape)
        else:
            raise Exception('wtf')

    def _collect_votes(self):
        """
        """


        for tf, comment, labels in self.cluster:
            if comment.author_flair_text:
                # We might miss some bandwagon fans that have specific flairs
                if self.home in comment.author_flair_text.lower():
                    if comment.score < 1:
                        self.argumentative['home'] += 1
                    else:
                        self.positive['home'] += 1
                    self.home_scores.append((labels, comment.score,\
                                            self._mention_team(comment)))
                elif self.away in comment.author_flair_text.lower():
                    if comment.score < 1:
                        self.argumentative['away'] += 1
                    else:
                        self.positive['away'] += 1
                    self.away_scores.append((labels, comment.score,\
                                            self._mention_team(comment)))
                else:
                    self.unaffiliated_scores.append((labels, comment.score,\
                                            self._mention_team(comment)))
            else:
                self.unaffiliated_scores.append((labels, comment.score,\
                                            self._mention_team(comment)))

    def _check_flair(self, comment):
        """
        Checks a variety of nicknames for match. For example, lsu is given
        by flair, but home team is given by Lousiana State.
        """
        if self.home in comment.author_flair_text.lower():
            return True

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

        # Make a loop so we can easily keep the algorithm the same for both
        for score_list, same, opposite in [(self.home_scores, 'home', 'away'), \
                                            (self.away_scores, 'away', 'home')]:
            if score_list:
                for comment_tags, score, mentioned in score_list:
                    self.tag_counts += comment_tags
                    if comment_tags[np.where(self.class_tags == 'D')]:
                        # We are going to skip these for now. Will apply their effects
                        # when we have a better idea of which team got the penalty
                        continue
                    elif comment_tags[np.where(self.class_tags == 'S')]:
                        bad_call[opposite] += 1
                        bad_call_scores[opposite] += score
                    elif comment_tags[np.where(self.class_tags == 'E')]:
                        bad_call[opposite] += 1
                        bad_call_scores[opposite] -= score
                    else: # Just assume they are complaining
                        bad_call[same] += 1
                        bad_call_scores[same] += score

        if self.unaffiliated_scores:
            for comment_tags, score, mentioned in self.unaffiliated_scores:
                self.tag_counts += comment_tags
                if mentioned:
                    if comment_tags[np.where(self.class_tags == 'S')]:
                        if ('home' not in mentioned and 'away' in mentioned) \
                            or \
                            ('home' in mentioned and 'away' not in mentioned):
                            bad_call[mentioned[0]] += 1
                            bad_call_scores[mentioned[0]] += score
                    if comment_tags[np.where(self.class_tags == 'G')]:
                        if 'home' in mentioned and 'away' in mentioned:
                            continue
                        else:
                            if 'home' in mentioned:
                                bad_call['away'] += 1
                                bad_call_scores['away'] += score
                            else:
                                bad_call['home'] += 1
                                bad_call_scores['home'] += score

                elif comment_tags[np.where(self.class_tags == 'SA')] and \
                    not comment_tags[np.where(self.class_tags == 'SH')]:
                    bad_call['away'] += 1
                    bad_call_scores['away'] += score
                elif comment_tags[np.where(self.class_tags == 'GH')] and \
                    not comment_tags[np.where(self.class_tags == 'SH')]:
                    bad_call['away'] += 1
                    bad_call_scores['away'] += score

                elif comment_tags[np.where(self.class_tags == 'SH')] and \
                    not comment_tags[np.where(self.class_tags == 'SA')]:
                    bad_call['home'] += 1
                    bad_call_scores['home'] += score
                elif comment_tags[np.where(self.class_tags == 'GA')] and \
                    not comment_tags[np.where(self.class_tags == 'SA')]:
                    bad_call['home'] += 1
                    bad_call_scores['home'] += score

        self._set_rule()
        print('Rule: {}'.format(self.rule))
        print('Bad call?')
        print(bad_call)
        print(bad_call_scores)

        if bad_call['home'] == 0:
            if bad_call_scores['away'] < 3:
                self.whiny['away'] += 1
        if bad_call['away'] == 0:
            if bad_call_scores['home'] < 3:
                self.whiny['home'] += 1

        if bad_call_scores['home'] == bad_call_scores['away']:
            pass # Not sure yet
        if bad_call_scores['home'] < 0 and bad_call_scores['away'] < 0:
            pass
        elif bad_call_scores['home'] > bad_call_scores['away']:
            self.team_affected = 'home'
        elif bad_call_scores['home'] < bad_call_scores['away']:
            self.team_affected = 'away'

        if self.team_affected == 'home':
            return (self.home, self.rule, bad_call_scores['home'] - bad_call_scores['away'])
        if self.team_affected == 'away':
            return (self.away, self.rule, bad_call_scores['away'] - bad_call_scores['home'])

    def _set_rule(self):
        """
        After the class labels are counted, we can predict what rule the cluster
        is talking about.
        """
        found = False
        missed = False
        #print(self.class_tags[np.argsort(self.tag_counts)][::-1])
        for code in self.class_tags[np.argsort(self.tag_counts)][::-1]:
            if found: # Check if missed is the next most frequent ruel
                if code == 'M':
                    missed = True
                break
            if code in '0,1,2,3,4,SHSAGHGAEDCMRCRR': # 0-4 are separated by commas to avoid matching, eg, 12
                continue
            elif code == 'M':
                missed = True
            else:
                rule = code
                found = True
        if missed:
            self.rule = 'Missed ' + rule
        else:
            self.rule = rule

    def _mention_team(self, comment):
        """
        """
        teams = []
        for nick, full_teams in self.nicknames_dict.items():
            if nick in comment.body:
                if self.home in full_teams:
                    teams.append('home')
                if self.away in full_teams:
                    teams.append('away')
        return teams

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
