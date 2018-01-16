import sqlite3, os, pickle, random, time

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

import praw

from MultiTargetModel import MultiTargetModel


class LemmaTokenizer(object):
    """
    Implemenation of a tokenizer/lemmatizer from sklearn's documentation.
    """
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def load_data(thread, overwrite=False, subreddit = 'cfb',\
                bot_params='bot1', verbose=True):
    """
    Loads invidudal game data. The function has multiple uses, many of which
    were only important during development.

    --------------
    INPUT
    --------------
    n_games:    Int - number of games to select from the game database
                    a value of 0 or None will collect the entire DB.
    pickle_path: Str - path to pickle file. If file exists at the location,
                    the pickle will be used. Otherwise, new data will be collected
                    from Reddit using thread ids from the game database
    overwrite:  Bool - If a pickle_path is provided and the file already exists,
                    a setting of True will overwrite the existing file
    subreddit:  Str - subreddit where the game threads are located
    db:         Str - path to the game DB
    bot_params: Str - praw bot parameters located in ~/.config/praw.ini
                (see praw documentation)
    verbose:    Bool - print status updates.
    --------------
    OUTPUT
    --------------
    documents:  List of praw comment objects - Each objects contains all of the
                    top level comments from a game. Should be able to connect to
                    reddit and get comment forrest reply.

    """
    pickle_path = '/data/comment_pickles/{}.pkl'.format(thread)
    if not os.path.isfile(pickle_path) or overwrite:
        if verbose:
            print('Collecting data from reddit')
        reddit = praw.Reddit(bot_params)
        start = time.time()
        if verbose:
            print('working on thread:', thread)
        submission = reddit.submission(id=thread)
        submission.comment_sort = 'new'
        comments = submission.comments
        comments.replace_more()
        game_documents = [top_level_comment \
                                for top_level_comment in comments]
        if verbose:
            print('Time: {:.1f}'.format(time.time()-start))
        if pickle_path:
            pickle.dump(game_documents, open(pickle_path, 'wb'))
    else:
        if verbose:
            print('Loading data from pickle')
        game_documents = pickle.load(open(pickle_path, 'rb'))

    if verbose:
        print('Finished loading data!')
    return game_documents # list (games) of lists of praw comment objects

def collect_game_threads(db='/data/cfb_game_db.sqlite3', n_games=None):
    """
    Makes query to database to collect game thread ids.

    --------------
    INPUT
    --------------
    n_games:    Int - number of games to select from the game database
                    a value of 0 or None will collect the entire DB.
    db:         Str - path to the game DB
    --------------
    OUTPUT
    --------------
    games:      List - list of game ids that praw can directly access.
    """
    if n_games: # if a number of games is specified, select that many
        query = """SELECT
                    game_id, game_thread, home, away, winner
                    FROM
                    games
                    LIMIT
                    {}
                    """.format(n_games)
    else: # select all threads
        query = """SELECT
                    game_id, game_thread, home, away, winner
                    FROM
                    games
                    """
    conn = sqlite3.connect(db)
    curr = conn.cursor()
    curr.execute(query)
    games = curr.fetchall()
    conn.close()
    return games

def replace_many(to_replace, replace_with, string):
    """
    Replace all of the items in the to_replace list with replace_with.
    --------------
    INPUT
    --------------
    to_replace: Itrable -  a string or list containing characters or substrings
                        to replace.
    --------------
    OUTPUT
    --------------
    string:     Str - string with all of the replacements made
    """
    for s in to_replace:
        string.replace(s, replace_with)
    return string

def get_MultiTargetModel(pickle_path='model.pkl', db='/data/cfb_game_db.sqlite3', overwrite=False, verbose=True, **kwargs):
    """

    """

    if not os.path.isfile(pickle_path) or overwrite:
        if verbose:
            print('Getting data from {}'.format(db))
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

        if verbose:
            print('Fitting new model.')

        multilabler = MultiTargetModel(MultinomialNB, vectorizer, stop_words=stop_words, tokenizer=LemmaTokenizer())
        multilabler.fit_classifier(X, labels, **kwargs)
        print('Recall:')
        multilabler.calc_recall()
        print('Precision:')
        multilabler.calc_preciscion()
        print('Accuracy:')
        multilabler.calc_accuracy()
        out = (multilabler, vectorizer)
        if pickle_path:
            print('Saving model as {}'.format(pickle_path))
            pickle.dump(out, open(pickle_path, 'wb'))

    else:
        if verbose:
            print('Loading data from pickle')
        out = pickle.load(open(pickle_path, 'rb'))

    return out


if __name__ == '__main__':
    load_data(n_games=1000, pickle_path='../ref_analysis/big_1000.pkl', overwrite=True)
