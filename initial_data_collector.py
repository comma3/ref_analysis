
import sys, re
from datetime import datetime, timedelta
import time
import sqlite3

import praw

def collect_old_game_ids(passed_set):
    """
    Gets list of games that have already been analyzed to ensure that games
    aren't analyzed twice. While running, the set should be kept in memory
    and passed around rather than touching the DB. Passing an empty set
    will cause the DB query.

    INPUT: set

    OUTPUT: set
    """
    if not passed_set:
        conn = sqlite3.connect("D:\\reddit\\reddit_user_data.sqlite3")
        curr = conn.cursor()
        curr.execute("""SELECT username
                        FROM users
                        """)
        old_games = curr.fetchall()
        conn.close()
        return set(old_games)

    return set() # Returning an empty set for now

def call_get_submissions(praw_instance, start_time, total_list=[], paginate=False, subreddit='cfb'):
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


def analyze_game_thread(threads):
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
                output_dict[working_dict[home]] = [date, thread.id, home, away, home, 0]
            else:
                print(thread)
                raise "Postgame thread appeared before game thread!"

    conn = sqlite3.connect("/data/cfb_game_db.sqlite3")
    curr = conn.cursor()
    curr.execute("""SELECT username
                    FROM users
                    """)
    old_games = curr.fetchall()
    conn.close()





def analyze_comments(to_analyze):
    ## TODO: Maybe this should be a class...
    """
    Takes a list of gamethreads that are ready to analyze (i.e., it's been
    about 5 minutes since the game ended).
    """
    pass



# Make sure we aren't doubling up on games. Uses ESPN game id to ensure
# uniqueness. Postgame threads are supposed to contain links to box scores,
# so we will collect them from there.
game_ids = collect_old_game_ids(True) # Currently just passing until db is set up

# Create bot instance
reddit = praw.Reddit('bot1')

# Store game threads in case there are multiple threads for a game.
# Clear this at some interval (maybe daily at like 4 am to limit maintanenceS).
potential_valid_game_threads = {}

search_depth = datetime.utcnow() - timedelta(weeks=4)
time_start = datetime.now()
results = call_get_submissions(reddit, search_depth)
print(datetime.now() - time_start)



print(results)





# # Find new posts in /r/CFB
# submissions = reddit.subreddit('CFB').new(limit=5)
# for post in submissions:
#     # First we need to find the correct game thread in case multiple threads
#     # are created. I believe duplicates are usually deleted quite quickly,
#     # so 5 minutes should be plenty. We'll count posts in case there are
#     # multiple game threads to differentiate.
#
#     is_old_enough = post.created_utc < time.time() - 300:
#     # This old enough metric may not work if posts are created too quickly
#     # That is, they might not be returned by subreddit.new() after 5 minutes
#
#     # Checking age first and continuing prevents excess nesting and improves
#     # readability.
#     if not is_old_enough:
#         continue
#
#     if '[game thread]' in post.lower():
#         if not post.is_self:
#             raise "Game thread is not a selfpost. Post id: {}".format(post.id)
#
#         away, home = re.findall(r'([A-z -]+) @ ([A-z -]+)'), post.lower())
#         try:
#             potential_valid_game_threads[(away, home)].append(post.id)
#         except KeyError:
#             potential_valid_game_threads[(away, home)] = [post.id]
#
#
#     elif '[postgame thread]' in post.lower():
#         if not post.is_self:
#             raise "Postgame thread is not a selfpost. Post id: {}".format(post.id)
#
#         game_id = re.findall(r'gameId=([0-9]+)', post.selftext)[0]
#         if game_id in game_ids:
#             raise "Game already analyzed. Post id: {}".format(post.id)
#
#         winner, loser = re.findall(r'([A-z -]+) defeats ([A-z -]+)'), post.lower())
#
#         # Determining the winner and getting game thread list at the same time
#         # May be worth checking if both exist (i.e., someone made an
#         # incorrect game thread or post game thread. Five minue wait should
#         # prevent those threads from being found)
#         try:
#             game_threads = potential_valid_game_threads[(winner, loser)]
#             home_winner = False
#         except:
#             game_threads = potential_valid_game_threads[(loser, winner)]
#             home_winner = True
#
#         if not game_threads:
#             # Do a search back a few hours and try to find the game thread.
#             # Alternatively, may simply raise an exception
#             pass
#
#         # Generate list of games to analyze
#         to_analyze.append(determine_game_thread(game_threads))
#
#         # Do some clean up to keep the potential_valid_game_threads
#         # list manageable. Problems may occur if a duplicate postgame thread
#         # appears long after the game ended and a postgame thread was already
#         # found. Unique checks using game_id should prevent problems.
#         for thread in game_threads:
#             if home_winner:
#                 del potential_valid_game_threads[(loser, winner)]
#             else:
#                 del potential_valid_game_threads[(winner, loser)]
#
#         # Game_id, Game_thread_id, Post_gamethread_id, Winner, Loser, home_winner
#
#         # Game_thread_id, post_id, post_body
#
#
# # We may want some parallelization here so we don't miss new threads
# analyze_comments(to_analyze)
