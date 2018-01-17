import string, sqlite3

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from library import *
from LemmaTokenizer import LemmaTokenizer

class MultiTargetModel():
    """
    """

    def __init__(self, model, n_jobs=1, vectorizer=CountVectorizer, stop_words='english', tokenizer=LemmaTokenizer):
        self.model = model
        self.n_jobs = n_jobs
        self.vectorizer = vectorizer(stop_words=stop_words, tokenizer=tokenizer())
        self.tokenizer = tokenizer
        self.stop_words = stop_words

        self.X = None
        self.targets = None
        self.classifier = None
        self.class_labels = None

        self.precision = None
        self.recall = None
        self.accuracy = None


    def fit_classifier(self, X, y, **kwargs):
        """
        """
        self.X = self.vectorizer.fit_transform(X)
        self.classifier = OneVsRestClassifier(self.model(**kwargs))
        self._make_targets(y) # Modifies self.targets
        self.classifier.fit(self.X, self.targets)


    def _make_targets(self, y):
        """
        Convert column of stringlists (e.g., '1,12,E') to MultiLabelBinarizer
        """
        mlb = MultiLabelBinarizer(n_jobs=self.n_jobs)
        # Type casting messed up and need to get value out of tuple
        strings = y.astype(str)
        clean = [x.replace(' ', '') for x in strings]
        dummies = [x.split(',') for x in clean]
        self.targets = mlb.fit_transform(dummies)
        self.class_labels = mlb.classes_
        #print(self.class_labels)

    def make_predictions(self, X):
        """
        """
        self.predictions = self.classifier.predict(X)
        return self.predictions

    def calc_accuracy(self, X=None):
        """
        """
        if not X:
            X = self.X
        self.accuracy = self.classifier.score(X, self.targets)
        print(self.accuracy)

    def calc_recall(self, X=None):
        """
        """
        if not X:
            X = self.X
        labels = np.array(self.targets)
        preds = np.array(self.make_predictions(X))
        labels[labels == 0] = 2
        preds[preds == 0] = 3
        correct = labels == preds
        labels[labels == 2] = 0
        self.recall = correct.sum(axis = 0) / labels.sum(axis=0)
        avg = 0
        for score in self.recall:
            print(score)
            avg += score
        print('Average Recall: {}'.format(avg/len(self.recall)))

    def calc_preciscion(self, X=None):
        """
        """
        if not X:
            X = self.X
        labels = np.array(self.targets)
        preds = np.array(self.make_predictions(X))
        labels[labels == 0] = 2
        preds[preds == 0] = 3
        correct = labels == preds
        preds[preds == 3] = 0
        self.precision = correct.sum(axis = 0) / preds.sum(axis=0)
        avg = 0
        for score in self.precision:
            print(score)
            avg += score
        print('Average Precision: {}'.format(avg/len(self.precision)))

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

    with open('../ref_analysis/data/common-english-words.csv') as f:
        stop_words = [word.strip() for word in f]

    vectizer = CountVectorizer(stop_words=stop_words, tokenizer=LemmaTokenizer())
    X = vectizer.fit_transform(text)

    multilabler = MultiTargetModel(MultinomialNB)
    multilabler.fit_classifier(X, labels)
