
import sys, re
from datetime import datetime, timedelta
import time
import sqlite3

import praw

def collect_old_game_ids(db, passed_set=False):
    """
    Gets list of games that have already been analyzed to ensure that games
    aren't analyzed twice. While running, the set should be kept in memory
    and passed around rather than touching the DB. Passing an empty set
    will cause the DB query.

    INPUT: set

    OUTPUT: set
    """
    if not passed_set:
        conn = sqlite3.connect(db)
        curr = conn.cursor()
        curr.execute("""SELECT
                        game_id
                        FROM
                        games
                        """)
        old_games = curr.fetchall()

        curr.execute("""SELECT
                        thread, date
                        FROM
                        games
                        ORDER BY
                        date
                        LIMIT
                        1
                        """)
        last_game = curr.fetchall()
        conn.close()
        return last_game, set(old_games)

    return 0, set() # Returning an empty set for now

def call_get_submissions(praw_instance, start_time, total_list=[], paginate=False, game_ids=set(), subreddit='cfb'):
    """
    Recursive approach rapidly exceeded max recursion depth.

    Looping handled by this function for clarity.
    """

    search_further = True
    while search_further:
        total_list, paginate = get_submissions(praw_instance, start_time,
                        total_list, paginate=paginate, subreddit=subreddit)
        search_further = datetime.utcfromtimestamp(paginate.created_utc) >= start_time
        print(total_list)

    return total_list

def get_submissions(praw_instance, start_time, total_list=[], paginate=False, subreddit='cfb'):
    """
    Get limit of subreddit submissions.

    praw_instance = a properly initialize praw instance
    start_time = date where the seach should begin in utc timestamp
    total_list = list of posts that are found by the method
    paginate = last post in a set of praw results. Used to set start for next
                set of results
    subreddit = subreddit to search (easy modification for other sports)

    Modified from https://gist.github.com/dangayle/4e6864300b58fee09ce1
    """
    limit = 100  # Reddit maximum limit

    if paginate:
        # get set of items posted prior to the last item from last call
        submissions = praw_instance.subreddit(subreddit).new(limit=limit, params={"after": paginate.fullname})
    else:
        submissions = praw_instance.subreddit(subreddit).new(limit=limit)

    # iterate through the submissions generator object
    # We need to store the last post in the current set no matter what.
    # so we need to track the loops (enumerate doesn't work on generator)
    i = 0
    submissions_list= []
    for thread in submissions:
        i+= 1
        # Make sure we are getting correct post typedatetime.utcfromtimestamp(paginate.created_utc) >= start_time
        is_game_thread = '[game thread]' in thread.title.lower() or '[postgame thread]' in thread.title.lower()
        if thread.is_self and is_game_thread:
        # and that it's a game thread or post game thread
            submissions_list.append(thread)

        # Store the last post in the current set of results
        # We will use this result to determine our starting point for the next
        # set of results
        if i == limit:
            paginate = thread

    total_list += submissions_list

    return total_list, paginate


def analyze_game_thread(threads, old_games):
    """
    Takes a list of potential gamethreads and filters for duplicates.

    ESPN gameids are used as unique keys.

    INPUT: Python list of post ids

    OUTPUT: None (adds to SQL DB)
    """
    output_dict = {}
    working_dict = {}
    found = set()

    for thread in threads:
        current_title = thread.title.lower()

        if '[postgame thread]' in current_title:
            game_id = re.findall(r'gameid=([0-9]{9})', thread.body.lower())[0]
            if not game_id:
                raise "No game id in postgame thread!"

            winner, loser = re.findall(r'([A-z -]+) defeats ([A-z -]+)'), post.lower())

            if game_id in output_dict:
                print(thread)
                raise "Duplicate postgame thread!"
            elif winner in working_dict:
                print(thread)
                raise "Duplicate postgame thread!"
            else:
                winner, loser = re.findall(r'([A-z -]+) defeats ([A-z -]+)'), post.lower())
                working_dict[winner] = game_id

        elif '[game thread]' in current_title:
            away, home = re.findall(r'([A-z -]+) @ ([A-z -]+)'), post.lower())
            date = datetime.fromutc(thread.created_utc)

            if (date, home, away) in found:
                print(thread)
                raise "Duplicate game threads!"
            found.add((date, home, away))

            if away in working_dict:
                output_dict[working_dict[away]] = [date, thread.id, home, away, away, 0]
            elif home in working_dict:
                output_dict[work?,?,?ing_dict[home]] = [date, thread.id, home, away, home, 0]
            else:
                print(thread)
                raise "Postgame thread appeared before game thread!"

        update_db(output_dict)


    def update_db(games):

        conn = sqlite3.connect("/data/cfb_game_db.sqlite3")
        curr = conn.cursor()

        # Just gonna create the table here so I don't have to do it elsewhere
        # Pass if the table already exists
        try:
            curr.execute("""CREATE TABLE
                            games (
                            game_id string, --Only an int if you want to do math with it
                            date string,
                            thread string,
                            home string,
                            away string,
                            winner string,
                            call_differential
                            )
                            """)

        except sqlite3.OperationalError:
            # Totally fine that table already exists
            pass

        game_tuples = (k,v for k,v in games.items())
        curr.executemany("""INSERT INTO playbyplay
                        (game_id, date, thread, home, away, winner, call_differential)
                        VALUES (?,?,?,?,?,?,?)
                        """, game_tuples)

        conn.close()


def analyze_comments(to_analyze):
    ## TODO: Maybe this should be a class...
    """
    Takes a list of gamethreads that are ready to analyze (i.e., it's been
    about 5 minutes since the game ended).
    """
    pass



if __name__ == '__main__':
    # Make sure we aren't doubling up on games. Uses ESPN game id to ensure
    # uniqueness. Postgame threads are supposed to contain links to box scores,
    # so we will collect them from there.
    last_game, game_ids = collect_old_game_ids(True)  # Currently just passing until db is set up

    # Create bot instance
    reddit = praw.Reddit('bot1')

    # In case program crashes while collecting,
    # we find (one of) the oldest found games and start our
    # search there.
    if last_game:
        paginate = reddit.submission(id=last_game)
    else:
        paginate = False

    search_depth = datetime.utcnow() - timedelta(weeks=4)

    time_start = datetime.now()
    call_get_submissions(reddit, search_depth, paginate=paginate, game_ids = game_ids) # Adds to DB
    print(datetime.now() - time_start)
