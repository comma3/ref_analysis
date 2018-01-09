import sqlite3, os, pickle, random, time
import praw

def load_data(pickle_path='', overwrite=False, n_games=1):

    if not os.path.isfile(pickle_path) or overwrite:
        db = '/data/cfb_game_db.sqlite3'
        subreddit = 'cfb'
        bot_params = 'bot1' # These are collected from praw.ini
        reddit = praw.Reddit(bot_params)
        threads = collect_game_threads(db, n_games)
        print(threads)
        print('Number of threads: ', len(threads))
        documents = []
        start = time.time()
        for i, thread in enumerate(threads):
            print('working on thread:', thread)
            submission = reddit.submission(id=thread[0])
            submission.comment_sort = 'new'
            comments = submission.comments
            comments.replace_more()
            game_comments = []
            for top_level_comment in comments:
                game_comments.append([top_level_comment.created_utc,\
                                        top_level_comment.body])
            documents.append(game_comments)
            print('Thread number: {} Average time: {}'.format(i, \
                                                    (time.time()-start)//(i+1))
        if pickle_path:
            pickle.dump(documents, open(pickle_path, 'wb'))
    else:
        print('Loading from pickle')
        documents = pickle.load(open(pickle_path, 'rb'))

    return documents

def collect_game_threads(db, n_games=1):
    """
    TODO
    """
    if n_games:
        query = """SELECT
                    game_thread
                    FROM
                    games
                    LIMIT
                    {}
                    """.format(n_games)
    else:
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

def categorize(string):
    string = string.lower()
    string = replace_many(['pass interference', 'opi', 'dpi', 'interference'],'pi' ,string)
    string = replace_many(['holding'],'hold' ,string)
    string = replace_many(['pass interference', 'opi', 'dpi', 'interference'],'pi' ,string)
