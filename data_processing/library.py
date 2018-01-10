import sqlite3, os, pickle, random, time
import praw

def load_data(n_games=1, pickle_path='', overwrite=False, subreddit = 'cfb',\
                db='/data/cfb_game_db.sqlite3', bot_params='bot1', \
                verbose=True):
    """
    bot_params are collected from ~/.config/praw.ini
    """
    if not os.path.isfile(pickle_path) or overwrite:
        reddit = praw.Reddit(bot_params)
        threads = collect_game_threads(db, n_games)
        if verbose:
            print(threads)
            print('Number of threads: ', len(threads))
        game_documents = []
        start = time.time()
        for i, thread in enumerate(threads):
            if verbose:
                print('working on thread:', thread)
            submission = reddit.submission(id=thread[0])
            submission.comment_sort = 'new'
            comments = submission.comments
            comments.replace_more()
            game_comments = []
            for top_level_comment in comments:
                game_comments.append([top_level_comment.created_utc,\
                                        top_level_comment.body])
            game_documents.append(game_comments)
            if verbose:
                print('Thread number: {} Average time: {}'.format(i, \
                                                    (time.time()-start)//(i+1)))
        if pickle_path:
            pickle.dump(game_documents, open(pickle_path, 'wb'))
    else:
        if verbose:
            print('Loading from pickle')
        game_documents = pickle.load(open(pickle_path, 'rb'))

    return game_documents # list (games) of lists of comments

def collect_game_threads(db, n_games=1):
    """
    TODO
    """
    if n_games: # if a number of games is specified, select that many
        query = """SELECT
                    game_thread
                    FROM
                    games
                    LIMIT
                    {}
                    """.format(n_games)
    else: # select all threads
        query = """SELECT
                    game_thread
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
    """
    for s in to_replace:
        string.replace(s, replace_with)
    return string
