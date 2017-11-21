import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


class CommentAnalyzer():
    """
    Uses a thread id and a predetermined model to determine the
    relevant outputs TODO: calls against, calls for
    """

    def __init__(self, thread_id, model):
        self.thread_id = thread_id
        self.model = model


    def label_comments(self):
        """
        Tag top level comments as call related or not.
        """
        pass

    def determine_support(self):
        """
        Take a top level comment and determine if children are agreeing or
        disagreeing with top post.

        """
        pass

    def score_comment(self):
        """
        Use votes and supporting comments to determine a score for positively
        labeled comments.
        """
        pass

    def update_team(self):
        """
        Query the DB and update the values for the teams in the thread.
        Should also destroy the object.
        """
