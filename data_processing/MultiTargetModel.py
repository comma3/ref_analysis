import string, sqlite3

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from library import *
from LemmaTokenizer import LemmaTokenizer

class MultiTargetModel():
    """
    """
    def __init__(self, model, n_jobs=1, vectorizer=CountVectorizer, stop_words='english', tokenizer=LemmaTokenizer, **vectorizer_params):
        self.model = model
        self.n_jobs = n_jobs
        self.mlb = None # MultiLabelBinarizer
        self.vectorizer = vectorizer(stop_words=stop_words, tokenizer=tokenizer(), **vectorizer_params)
        self.tokenizer = tokenizer
        self.stop_words = stop_words

        self.X = None
        self.targets = None
        self.classifier = None
        self.target_classes = None

        self.precision = None
        self.recall = None
        self.accuracy = None


    def fit_classifier(self, X, y, **kwargs):
        """
        """
        self.X = self.vectorizer.fit_transform(X)
        self.classifier = OneVsRestClassifier(self.model(**kwargs))
        # Normally we pass targets as a list of strings
        # but we need to process it first to do test train split
        if y.ndim == 1:
            self._make_targets(y) # Modifies self.targets
        else:
            self.targets = y
        self.classifier.fit(self.X, self.targets)


    def _make_targets(self, y):
        """
        Convert column of stringlists (e.g., '1,12,E') to MultiLabelBinarizer
        """
        self.mlb = MultiLabelBinarizer()
        # Type casting messed up and need to get value out of tuple
        strings = y.astype(str)
        clean = [x.replace(' ', '') for x in strings]
        dummies = [x.split(',') for x in clean]
        self.targets = self.mlb.fit_transform(dummies)
        self.target_classes = self.mlb.classes_
        #print(self.target_classes)

    def _transform_targets(self, y):
        strings = y.astype(str)
        clean = [x.replace(' ', '') for x in strings]
        dummies = [x.split(',') for x in clean]
        return self.mlb.transform(dummies)

    def make_predictions(self, X):
        """
        """
        self.predictions = self.classifier.predict(X)
        return self.predictions

    def calc_accuracy(self, X=None, y=None):
        """
        """
        if X is None:
            X = self.X
        else:
            X = self.vectorizer.transform(X)
        if y is None:
            y = self.targets

        self.accuracy = self.classifier.score(X, y)
        print(self.accuracy)

    def calc_recall(self, X=None, y=None):
        """
        """
        if X is None:
            X = self.X
        else:
            X = self.vectorizer.transform(X)
        if y is None:
            y = self.targets

        labels = np.array(y)
        preds = np.array(self.make_predictions(X))
        labels[labels == 0] = 2
        preds[preds == 0] = 3
        #correct = labels[(labels==1) & (preds==1)]
        correct = labels == preds
        preds[preds == 3] = 0
        labels[labels == 2] = 0
        self.recall = correct.sum(axis = 0) / labels.sum(axis=0) # Labels = True Positives + False Negatives
        avg = 0
        for score in self.recall:
            print(score)
            avg += score
        print('Average Recall: {}'.format(avg/len(self.recall)))
        weighted = 0
        weights = labels.sum(axis=0) / labels.sum() # columnwise totals divided by total of array
        for score, weight in zip(self.recall, weights):
            weighted += score*weight
        print('Average Weighted Recall: {}'.format(weighted))

    def calc_preciscion(self, X=None, y=None):
        """
        """
        if X is None:
            X = self.X
        else:
            X = self.vectorizer.transform(X)
        if y is None:
            y = self.targets

        labels = np.array(y)
        preds = np.array(self.make_predictions(X))
        labels[labels == 0] = 2 # Hacky solution. Couldn't find a way to only evaluate
        preds[preds == 0] = 3
        correct = labels == preds # True positives
        preds[preds == 3] = 0
        labels[labels ==2] = 0
        self.precision = (correct.sum(axis = 0)+1) / (preds.sum(axis=0)+1) # Predictions = True Positives + False Positives

        avg = 0
        for score in self.precision:
            print(score)
            avg += score
        print('Average Precision: {}'.format(avg/len(self.precision)))
        weighted = 0
        weights = labels.sum(axis=0) / labels.sum() # columnwise totals divided by total of array
        for score, weight in zip(self.precision, weights):
            weighted += score*weight
        print('Average Weighted Precision: {}'.format(weighted))



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

    print('Number of documents in training db: {}'.format(len(text)))

    with open('../ref_analysis/data/common-english-words.csv') as f:
        stop_words = [word.strip() for word in f]

    # Cleans and prepares data for test train split
    strings = labels.astype(str)
    clean = [x.replace(' ', '') for x in strings]
    dummies = [x.split(',') for x in clean]

    # Create a label array
    mlb = MultiLabelBinarizer()
    ys = mlb.fit_transform(dummies)
    sums = ys.sum(axis=0)
    print('Counts of classes:\n', sums)
    # Drop classes with fewer than 100 samples (needed to get stratified
    # test train split data)
    drop = []
    for col in range(ys.shape[1]):
        if sums[col] < 100:
            drop.append(col)
    # Prints shape before and after dropping so we can make sure they are
    # actually dropping.
    print(ys.shape)
    ys = np.delete(ys, np.array(drop), axis=1)
    print(ys.shape)

    X_train, X_test, y_train, y_test = train_test_split(text, ys, stratify=ys)
    model = MultiTargetModel(MultinomialNB, vectorizer=CountVectorizer, stop_words=stop_words, tokenizer=LemmaTokenizer, ngram_range=(1, 5), binary=False)
    model.fit_classifier(X_train, y_train, alpha=0, fit_prior=False)

    print('Recall:')
    model.calc_recall(X_test, y_test)
    print('Precision:')
    model.calc_preciscion(X_test, y_test)
    print('Accuracy:')
    model.calc_accuracy(X_test, y_test)
