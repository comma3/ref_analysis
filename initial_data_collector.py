import sys, re
from datetime import datetime, timedelta
from dateutil import parser

import time
import sqlite3

import praw

def collect_old_game_ids(db, passed_set=False):
    """
    Gets list of games that have already been analyzed to ensure that games
    aren't analyzed twice. While running, the set should be kept in memory
    and passed around rather than touching the DB. Passing an empty set
    will start the DB query.

    INPUT: set

    OUTPUT: thread id of the most recent game, set of game_ids
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

def generate_dates(season_start, season_end, interval=None):
    """
    Generates a complete list of utc timestamps for dates between season_start
    and season_end. Could be modified to skip days or to collect actual dates
    that have games from a DB or the web. I prefer the exhaustive search for
    now as it only needs to be completed once.

    INPUT:
    season_start: String of date in MM/DD/YYYY for the first day of the season.
    season_end: String of date in MM/DD/YYYY for the last day of the season.

    OUTPUT:
    list of utc timestamps corresponding to every 4am EST in that period

    TODO: address DST?
    """

    epoch = datetime.utcfromtimestamp(0)
    # Naive datetime (local machine time) setting "day" start to 4am EST
    next_date = (parser.parse(season_start)-epoch).total_seconds() - 57600
    stop_date = (parser.parse(season_end)-epoch).total_seconds() + 86400
    while next_date < stop_date:
        next_date = next_date + 86400
        # Stack overflow said this was the best method...
        yield next_date

def get_submissions(praw_instance, start_time, stop_time=None, query='', subreddit='cfb'):
    """
    Get limit of subreddit submissions by date. Defaults to search a singal day
    so usage is suggested to put a utc timestamp for midnight

    praw_instance = a properly initialize praw instance
    start_time = int or float of date where the seach should begin in
                utc timestamp
    stop_time = defaults to a timedelta of +1 day
    total_list = list of posts that are found by the method
    subreddit = string respresenting subreddit to search (easy modification for
                other sports).
    query = string for reddit search (reddit search is very unreliable, so we
            check ourselves).
    """

    if not stop_time:
        # Making an assumption that there won't be 1000 posts in a single day
        # Reddit search limits results to 1000 for anything
        stop_time = date = start_time + 86400 # add a day

    game_threads = []
    i = 0 # Can't use enumerate on generator
    for thread in praw_instance.subreddit(subreddit).submissions(start_time, stop_time, query):
        i += 1
        title = thread.title.lower()
        if '[postgame thread]' in title or '[game thread]' in title:
            game_threads.append(thread)
    print(i)
    if i > 950:
        raise "Dangerously close to search limits"
    return game_threads


def analyze_game_thread(threads, old_games):
    """
    Takes a list of potential game_threads and filters for duplicates.

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

            print(re.findall(r'([A-z -]+) defeats ([A-z -]+)', current_title.lower()))
            winner, loser = re.findall(r'([A-z -]+) defeats ([A-z -]+)', current_title.lower())

            if game_id in output_dict:
                print(thread)
                raise "Duplicate postgame thread!"
            elif winner in working_dict:
                print(thread)
                raise "Duplicate postgame thread!"
            else:
                winner, loser = re.findall(r'([A-z -]+) defeats ([A-z -]+)', current_title.lower())
                working_dict[winner] = game_id

        elif '[game thread]' in current_title:
            print(re.findall(r'\] ([A-z -]+) @ ([A-z -]+)', current_title.lower()))
            print(re.findall(r'\] ([A-z -]+) @ ([A-z -]+)', current_title.lower())[0])

            away, home = re.findall(r'\] ([A-z -]+) @ ([A-z -]+)', current_title.lower())[0]
            date = datetime.fromutc(thread.created_utc)

            if (date, home, away) in found:
                print(thread)
                raise "Duplicate game threads!"
            found.add((date, home, away))

            if away in working_dict:
                output_dict[working_dict[away]] = [date, thread.id, home, away, away, 0]
            elif home in working_dict:
                output_dict[working_dict[home]] = [date, thread.id, home, away, home, 0]
            else:
                print(thread)
                # Probably just pass here. Occasionally game threads aren't created
                raise "Postgame thread appeared before game thread!"
    #update_db(output_dict)
    print(output_dict)


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
                        call_differential int,
                        approx_home_fans int, -- maybe detractors?
                        approx_away_fans int,
                        approx_impartial_fans int
                        )
                        """)

    except sqlite3.OperationalError:
        # Totally fine that table already exists
        pass

    game_tuples = [(k,v) for k,v in games.items()]
    curr.executemany("""INSERT INTO playbyplay
                    (game_id,
                    date,
                    thread,
                    home,
                    away,
                    winner,
                    call_differential)
                    VALUES
                    (?,?,?,?,?,?,?,0,0,0,0)
                    """, game_tuples)

    conn.close()


def analyze_comments(to_analyze):
    """
    Takes a list of game_threads that are ready to analyze.
    """
    pass



if __name__ == '__main__':
    # Input variables
    # Could make this a command line script
    db = '~/data/cfb_game_db.sqlite3'
    subreddit = 'cfb'
    season_start ='9/1/2017'
    season_end = '9/7/2017'

    # These are collected from praw.initialize
    # See fake_praw.ini for format with credentials redacted
    bot_params = 'bot1'

    # Make sure we aren't doubling up on games. Uses ESPN game id to ensure
    # uniqueness. Postgame threads are supposed to contain links to box scores,
    # so we will collect them from there.
    last_game, game_ids = collect_old_game_ids(db, True)  # Currently just passing until db is set up

    # Create bot instance
    reddit = praw.Reddit(bot_params)

    game_threads = []

    # Time how long a season takes to collect
    # Mostly limited by Reddit TOS to 1 request every 2 seconds
    # Approx. 4.5 minutes per CFB season
    time_start = datetime.now()
    for date in generate_dates(season_start, season_end):
        #print(date) # UTC
        print(datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')) # Human
        game_threads.extend(get_submissions(reddit, date))
        # Determine size in memory -> ~100 MB / CFB season
        size = sum([sys.getsizeof(x) for x in game_threads])
        print('List size: ', len(game_threads), ' memory ', size)
    # Could do some parallelization here
    analyze_game_thread(game_threads, game_ids) # Adds to DB
    print(datetime.now() - time_start)
