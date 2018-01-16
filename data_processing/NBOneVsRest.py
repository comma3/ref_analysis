import string

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB


def fit_classifier(model, X, y, **kwargs):
    """
    """
    clf = OneVsRestClassifier(model(**kwargs), n_jobs=1)
    clf.fit(X, y)

def make_targets(target_column):
    mlb = MultiLabelBinarizer()

    clean = map(string.replace(' ', ''), target_column)
    dummies = map(split(','), clean)
    targets = mlb.fit_transform(dummies)
