import sqlite3, os, pickle, random, time, csv
from collections import OrderedDict

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import praw

from LemmaTokenizer import LemmaTokenizer
#from MultiTargetModel import MultiTargetModel

def make_team_nickname_dict(filename):
    """
    Loads team nickname dict from disk and preprosses the text.

    INPUT:
    ---------------
    filename: str - path to the team_nickname_dict

    OUTPUT:
    ---------------
    team_nickname_dict: ordered dictionary - dictionary of unique nicknames to
            the unique base team name (i.e., the flair tag). the base teams is a
            set that may include multiple teams. Correct team can be found by
            match home/away team. Ordered by nickname length so longer matches
            are found first when looped.

    """
    dictionaryoutput = OrderedDict()
    with open(filename) as file:
        # ensure no empty strings and that the text is lowered and stripped
        entries = [[x.strip().lower() for x in row] \
                        for row in filter(bool, list(csv.reader(file)))]
        entries.sort(key=lambda x: len(x[0]), reverse=True) # sort by the length of the key
        for entry in entries:
            dictionaryoutput[entry[0]] = set(filter(bool, entry[1:])) # Convet list of base_teams to set
        #print(dictionaryoutput)
    return dictionaryoutput


def sub_home_away(doc, home, away, team_nickname_dict):
    """
    Uses a dictionary to replace team names and nicknames with a tag indicating
    whether they are the home or away team.

    These tags are stored in the training data but generated dynamically
    during fitting (original training text can be found from the comment id
    field in the db).
    """

    try:
        standard_home = team_nickname_dict[home]
    except KeyError:
        raise KeyError("Home team not in dictionary {}. Opponent {}".format(home, away))

    if len(standard_home) > 1:
        raise ValueError("Home team name not unique. Received {} and gave {} with the away team {}.".format(home, standard_home, away))

    try:
        standard_away = team_nickname_dict[away]
    except KeyError:
        raise KeyError("Away team not in dictionary {}. Opponent {}".format(away, home))

    if len(standard_away) > 1:
        raise ValueError("Home team name not unique. Received {} and gave {} with the home team {}.".format(away, standard_away, home))

    doc = doc.lower()
    # base_team is the standardized unique team name determined from the flair.
    # team_nickname_dict is an ordered dict that's ordered by reverse length of the key
    # we want to replace the longest string possible first
    for team_nick, base_team_set in team_nickname_dict.items():
        if ' {} '.format(team_nick) in doc or ' {}.'.format(team_nick) in doc or \
        ' {}?'.format(team_nick) in doc or ' {}!'.format(team_nick) in doc or \
        ' {},'.format(team_nick) in doc or ' {}:'.format(team_nick) in doc or \
        ' {};'.format(team_nick) in doc or ' {}-'.format(team_nick) in doc or \
        " {}'".format(team_nick) in doc or ' {}"'.format(team_nick) in doc or \
        " {})".format(team_nick) in doc or '({} '.format(team_nick) in doc or \
        doc.endswith(' {}'.format(team_nick)) or doc.startswith('{} '.format(team_nick)): # Check whether the team is mentioned in the document
            for base_team in base_team_set:
                # If the nickname is appropriate for the home team (i.e., don't replace blindly)
                if base_team in standard_home: #standards are sets
                    doc = doc.replace(team_nick, 'hometeamtrack')
                elif base_team in standard_away:
                    doc = doc.replace(team_nick, 'awayteamtrack')

    return doc


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

def collect_game_threads(db='/data/cfb_game_db.sqlite3', n_games=None, random=False):
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
    if random:
        query = query + "ORDER BY RANDOM()"
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

        multilabler = MultiTargetModel(MultinomialNB, vectorizer=CountVectorizer, stop_words=stop_words, tokenizer=LemmaTokenizer)
        multilabler.fit_classifier(text, labels, **kwargs)
        if pickle_path:
            print('Saving model as {}'.format(pickle_path))
            pickle.dump(multilabler, open(pickle_path, 'wb'))

    else:
        if verbose:
            print('Loading data from pickle')
        multilabler = pickle.load(open(pickle_path, 'rb'))

    return multilabler


if __name__ == '__main__':
    pass
    #make_team_nickname_dict('team_list.csv')
