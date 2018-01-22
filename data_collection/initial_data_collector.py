#!/usr/bin/python

import sys, re
from datetime import datetime, timedelta
from dateutil import parser

import time
import sqlite3

import praw

def clean_team_names(title):
    """
    The format of game thread titles is variable. This function attempts to
    handle all of the formats. It may need to be adjusted for other sports.

    INPUTS:
    title: Str title of game thread

    OUTPUT:
    tuple of cleaned team names

    """
    title = title.replace(u'\xe9','e') # replaces e` in san jose
    # vs. indicates neutral site. May alter stats somewhat.
    first, second = re.findall(r'[\]]([\s\S]+) (?:@|[vs.]+|at|defeat[s]*|beat[s]*|upset[s]*|survive[s]*) ([\s\S]+)', title)[0]
    # Some games have titles (e.g., FCS Championship: ) and are split on ':'
    # First we need to get rid of time (e.g., 6:30) - just remove :[0-5]
    for i in range(6):
        first = first.replace(':' + str(i),'')
        second = second.replace(':' + str(i),'')
    # Then, we split on the colon. I like the EAFP approach rather than LBYL
    try:
        first = first.split(':')[1]
    except IndexError:
        pass
    try:
        second = second.split(':')[1]
    except IndexError:
        pass

    first = first.strip() # These are needed so that the correct group is found
    second = second.strip()
    first_clean = re.findall(r'([a-z \-&\.\']+)', first)[0]
    second_clean = re.findall(r'([a-z \-&\.\']+)', second)[0]
    return first_clean.strip(), second_clean.strip()

def collect_old_game_ids(db):
    """
    Gets list of games that have already been analyzed to ensure that games
    aren't analyzed twice. While running, the set should be kept in memory
    and passed around rather than touching the DB. Passing an empty set
    will start the DB query.

    INPUT: set

    OUTPUT: thread id of the most recent game, set of game_ids
    """

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
    return set(old_games)


def generate_dates(season_start, season_end, year, interval=86400):
    """
    Generates a complete list of utc timestamps for dates between season_start
    and season_end. Could be modified to skip days or to collect actual dates
    that have games from a DB or the web. I prefer the exhaustive search for
    now as it only needs to be completed once and is relatively fast.

    INPUT:
    season_start: String of date in MM/DD for the first day of the season.
    season_end: String of date in MM/DD for the last day of the season.
    year: year in question
    interval: Int indicating the width of query window. Defaults to 1 day but if
                there are too many posts per day a smaller interval may be
                necessary. Units = seconds

    OUTPUT:
    list of utc timestamps corresponding to every 4am EST in that period (does
    not account for DST)

    TODO: address DST?
    """

    # Check if we span new years
    season_start += '/' + str(year)
    if season_end.split('/')[0] < season_start.split('/')[0]:
        season_end += '/' + str(year + 1)
    else:
        season_end += '/' + str(year)

    epoch = datetime.utcfromtimestamp(0)
    # Naive datetime (local machine time) setting "day" start to 4am EST
    # Stack overflow said this was the best method to convert to utc...
    next_date = (parser.parse(season_start)-epoch).total_seconds() - 57600
    stop_date = (parser.parse(season_end)-epoch).total_seconds() + 86400
    while next_date < stop_date:
        next_date = next_date + interval
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
        stop_time = start_time + 86400 # add a day

    game_threads = []
    i = 0 # Can't use enumerate on generator
    for thread in praw_instance.subreddit(subreddit).submissions(start_time, stop_time, query):
        i += 1
        title = thread.title.lower()
        is_postgame = '[post game thread]' in title or '[postgame thread]' in title
        is_game = '[game thread]' in title
        if is_postgame or is_game:
            game_threads.append(thread)
    print('Total threads on this date: ', i)
    if i > 999:
        # Want to raise an exception here so we don't miss any data
        raise BaseException("Exceeded search limits!")
    return game_threads

def match_postgame_with_game(home, away, date, working_dict, output_dict, thread, no_post_game, game_date=None):

    if not game_date:
        game_date = date

    try:
        output_dict[working_dict[(away, date)][0]] = [game_date, thread.id, working_dict[(away, date)][1], home, away, away, thread.score]
    except KeyError:
        output_dict[working_dict[(home, date)][0]] = [game_date, thread.id, working_dict[(home, date)][1], home, away, home, thread.score]

    return output_dict, no_post_game

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

    # Want to find the postgame thread first
    # Reddit queries are ordered in time, so it makes sense to search from the
    # beginning to the end and flip here.
    if threads[0].created_utc < threads[1].created_utc:
        threads = threads[::-1]
    # Although we can still use the game threads if we can't identify a
    # post game thread. Will have to handle them slightly differently
    no_post_game = []
    for thread in threads:
        current_title = thread.title.lower()
        date = datetime.fromtimestamp(thread.created).strftime('%Y-%m-%d')
        if '[post game thread]' in current_title or '[postgame thread]' in current_title:
            # Sometimes "Game threads" and "Postgame threads" will be made for
            # events that aren't games, resulting in some of the regex failing
            try:
                game_id = re.findall(r'id=([0-9]{9})', thread.selftext.lower())[0]
            except IndexError:
                print("No game id in postgame thread!\nTitle: {}\nThread: {}".format(current_title, thread))
                continue
            try:
                winner, loser = clean_team_names(current_title)
            except IndexError:
                # Print these threads for review
                # Most of the time they are fine.
                print('Incorrect postgame thread format')
                print(current_title)
                print(thread)
                continue

            # Check if there's already a postgame thread
            if working_dict.get((winner, date), [None])[0] == game_id:
                # Hasn't happened yet, but I'm going to raise an exception
                # so I can investigate if it happens.
                print("Duplicate postgame thread!\nThread id: {}\nGame_id: {}\nOther thread:{}".format(thread, game_id, working_dict[winner,date][1]))
            elif not winner:
                # This issue should be resolved
                print('+++++++++++\nNo winner returned!\n+++++++++')
                print(loser)
                print(current_title)
                print(thread)
            else:
                working_dict[(winner, date)] = (game_id,thread.id)

        elif '[game thread]' in current_title:
            try:
                away, home = clean_team_names(current_title)
            except IndexError:
                print('Incorrect game thread format')
                print(current_title)
                print(thread)
                continue


            # Attempt to correlate postgame thread with game thread
            # Sometimes there are no postgame thread or they lacked game_id
            # This is especially true for older threads (before ca. 2015)
            # where threads were manually created
            try:
                output_dict, no_post_game = match_postgame_with_game(home, away, date, working_dict, output_dict, thread, no_post_game)
            except KeyError:
                # games can span midnight so we make sure the game thread
                # wasn't from the day before the postgame thread
                new_day = parser.parse(date)
                day_plus_one = new_day + timedelta(days=1)
                next_day = day_plus_one.strftime('%Y-%m-%d')
                try:
                    output_dict, no_post_game = match_postgame_with_game(home, away, next_day, working_dict, output_dict, thread, no_post_game, game_date=date)
                except KeyError:
                    #print('No post game thread found!')
                    no_post_game.append([date, thread.id, 'None', home, away, 'None', thread.score])

    print('Number of no post game:',len(no_post_game)) # Should be zero
    print("Number of games found in this interval: ", len(output_dict))
    return output_dict, no_post_game

def update_db(games, no_post_games, db):
    """
    Takes dictionary of games and adds to the db
    """
    # need to check for existence of folder and make it
    conn = sqlite3.connect(db)
    curr = conn.cursor()
    # Just gonna create the table here so I don't have to do it elsewhere
    # Pass if the table already exists
    try:
        curr.execute("""CREATE TABLE
                        games (
                        game_id string, --Only an int if you want to do math with it
                        date string,
                        game_thread string,
                        postgame_thread string,
                        home string,
                        away string,
                        winner string,
                        approx_viewers int,
                        call_differential int,
                        approx_home_fans int, -- maybe detractors?
                        approx_away_fans int,
                        approx_impartial_fans int
                        );
                        """)
        conn.commit()
    except sqlite3.OperationalError:
        # Totally fine that table already exists
        pass

    # Flatten dictionary results into a single list
    game_list =[]
    for k,v in games.items():
        temp = [k]
        for x in v:
            temp.append(x)
        game_list.append(temp)

    for game in no_post_games:
        # replace game_id with 0 - other info from above:
        # [date, thread.id, 'None', home, away, 'None', thread.score]
        temp = ['0']
        temp.extend(game)
        game_list.append(temp)

    curr.executemany("""INSERT INTO games
                    (game_id,
                    date,
                    game_thread,
                    postgame_thread,
                    home,
                    away,
                    winner,
                    approx_viewers,
                    call_differential,
                    approx_home_fans,
                    approx_away_fans,
                    approx_impartial_fans)
                    VALUES
                    (?,?,?,?,?,?,?,?,0,0,0,0);
                    """, game_list)
    conn.commit()
    conn.close()

if __name__ == '__main__':
    # Input variables
    # Could make this a command line script
    db = '/data/cfb_game_db.sqlite3'
    subreddit = 'cfb'
    season_start ='8/15' # 'Month/Day' requires /
    season_end = '1/20'
    # Postgame threads didn't really start appearing until 2014
    # In the future, I'd like to make search more flexible.
    first_year = 2014 # Parser should accept both 20XX and XX
    last_year = 2017
    bot_params = 'bot1' # These are collected from praw.ini
    reddit = praw.Reddit(bot_params) # Create bot instance

    # Make sure we aren't doubling up on games. Uses ESPN game id to ensure
    # uniqueness. Postgame threads are supposed to contain links to box scores,
    # so we will collect them from there.
    #game_ids = collect_old_game_ids(db, True)  # Currently just passing until db is set up
    game_ids = set()
    # Time how long a season takes to collect limited by Reddit TOS to 1
    # request every 2 seconds. Approx. 4.5 minutes per CFB season
    time_start = datetime.now()

    for year in range(first_year, last_year+1):
        print('==========START {}============='.format(year))
        game_threads = []
        for date in generate_dates(season_start, season_end, year):
            #print(date) # UTC
            print(datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')) # Human
            game_threads.extend(get_submissions(reddit, date))
            # Determine size in memory -> ~100 MB / CFB season
            size = sum([sys.getsizeof(x) for x in game_threads])
            print('List size: ', len(game_threads), ' Memory ', size)
        print('==========THREADS FOR {}============='.format(year))
        # Could do some parallelization here
        valid_threads, no_post_game = analyze_game_thread(game_threads, game_ids)
        update_db(valid_threads, no_post_game, db)
    print(datetime.now() - time_start)
