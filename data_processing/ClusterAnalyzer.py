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

    def __init__(self, comments, model):

        self.comments = comments
        self.model = model

        self.user_dist = defaultdict(int) # d[team] : # unique poseters
        self.votes = defaultdict(list) # votes[home] :

        self.team_affected = None
        self.rule = None
        self.bad_call = None
        self.egregiousness = 0


        for comment in self.comments:
            if comment.author_flair_text:
                [self.user_dist[f] += 1 for f in comment.author_flair_text.split('/')]



    def _count_votes(self):
        """
        """
        pass

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
