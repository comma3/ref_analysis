import string, sqlite3

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from library import LemmaTokenizer

class MultiTargetModel():
    """
    """

    def __init__(self, model, n_jobs=1):
        self.model = model
        self.classifier = None

        self.X = None
        self.targets = None

        self.precision = None
        self.recall = None
        self.accuracy = None

    def fit_classifier(self, X, y, **kwargs):
        """
        """
        self.X = X
        self.classifier = OneVsRestClassifier(self.model(**kwargs))
        self._make_targets(y)
        self.classifier.fit(X, self.targets)

        #for row in zip(y,preds):
            # Proabably need to create my own recall metric here
           #print(row)
        return self.classifier



    def _make_targets(self, y):
        """
        Convert column of stringlists (e.g., '1,12,E') to MultiLabelBinarizer
        """
        mlb = MultiLabelBinarizer()
        # Type casting messed up and need to get value out of tuple
        strings = y.astype(str)
        clean = [x.replace(' ', '') for x in strings]
        dummies = [x.split(',') for x in clean]
        self.targets = mlb.fit_transform(dummies)

    def make_predictions(self, X):
        """
        """
        self.predictions = self.classifier.predict(X)
        return self.predictions

    def calc_accuracy(self):
        """
        """
        self.accuracy = self.classifier.score(X, self.targets)
        print(self.accuracy)

    def calc_recall(self):
        """
        """
        labels = np.array(labels)
        preds = np.array(preds)
        labels[labels == 0] = 2
        preds[preds == 0] = 3
        correct = labels == preds
        labels[labels == 2] = 0
        self.recall = correct.sum(axis = 0) / labels.sum(axis=0)
        for score in self.recall:
            print(score)

    def calc_preciscion(self):
        """
        """
        self.targets = np.array(labels)
        preds = np.array(preds)
        labels[labels == 0] = 2
        preds[preds == 0] = 3
        correct = labels == preds
        preds[preds == 3] = 0
        self.precision = correct.sum(axis = 0) / preds.sum(axis=0)
        for score in self.precision:
            print(score)


if __name__ == '__main__':
    db='/data/cfb_game_db.sqlite3'
    query = """SELECT
                body, category
                FROM
                training_data
                """
    conn = sqlite3.connect(db)
    curr = conn.cursor()
    curr.execute(query)
    data = np.array(curr.fetchall())
    conn.close()

    text = data[:,0]
    labels = data[:,1]

    y = make_targets(labels)

    with open('../ref_analysis/data/common-english-words.csv') as f:
        stop_words = [word.strip() for word in f]

    vectizer = CountVectorizer(stop_words=stop_words, tokenizer=LemmaTokenizer())
    X = vectizer.fit_transform(text)

    multilabler = MultiTargetModel(MultinomialNB)
    multilabler = fit_classifier(X, y)
