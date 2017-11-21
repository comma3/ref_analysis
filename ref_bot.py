import praw
import time
import sqlite3
import re


def determine_game_thread(threads):
    """
    Takes a list of potential gamethreads and returns the post that is
    identified as the most likely candidate. In most cases, this should
    return a single post immediately. If there are conflicting posts,
    the post with the most comments is selected.

    INPUT: Python list of post ids

    OUTPUT: A single string for the correct post id
    """
    if len(threads) != 1:
        # determine which post has more comments
        # This really shouldn't ever be more than 2 or 3
        current_leader = ('', 0)
        for thread in threads:
            num_comments = len(reddit.submission(id=thread[0]).comments)
            if num_comments > current_leader[1]:
                current_leader = (post, num_comments)
        if current_leader[0]:
            return current_leader[0]
        else:
            raise 'No valid game thread found!'
    return threads[0]

def analyze_comments(to_analyze):
    ## TODO: Maybe this should be a class...
    """
    Takes a list of gamethreads that are ready to analyze (i.e., it's been
    about 5 minutes since the game ended).
    """
    pass

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


# Make sure we aren't doubling up on games. Uses ESPN game id to ensure
# uniqueness. Postgame threads are supposed to contain links to box scores,
# so we will collect them from there.
game_ids = collect_old_game_ids(True)

# Create bot instance
reddit = praw.Reddit('bot1')

# Store game threads in case there are multiple threads for a game.
# Clear this at some interval (maybe daily at like 4 am to limit maintanence).
potential_valid_game_threads = {}

# Find new posts in /r/CFB
submissions = reddit.subreddit('CFB').new(limit=5)
for post in submissions:
    # First we need to find the correct game thread in case multiple threads
    # are created. I believe duplicates are usually deleted quite quickly,
    # so 5 minutes should be plenty. We'll count posts in case there are
    # multiple game threads to differentiate.

    is_old_enough = post.created_utc < time.time() - 300:
    # This old enough metric may not work if posts are created too quickly
    # That is, they might not be returned by subreddit.new() after 5 minutes

    # Checking age first and continuing prevents excess nesting and improves
    # readability.
    if not is_old_enough:
        continue

    if '[game thread]' in post.lower():
        if not post.is_self:
            raise "Game thread is not a selfpost. Post id: {}".format(post.id)

        away, home = re.findall(r'([A-z -]+) @ ([A-z -]+)'), post.lower())
        try:
            potential_valid_game_threads[(away, home)].append(post.id)
        except KeyError:
            potential_valid_game_threads[(away, home)] = [post.id]


    elif '[postgame thread]' in post.lower():
        if not post.is_self:
            raise "Postgame thread is not a selfpost. Post id: {}".format(post.id)

        game_id = re.findall(r'gameId=([0-9]+)', post.selftext)[0]
        if game_id in game_ids:
            raise "Game already analyzed. Post id: {}".format(post.id)

        winner, loser = re.findall(r'([A-z -]+) defeats ([A-z -]+)'), post.lower())

        # Determining the winner and getting game thread list at the same time
        # May be worth checking if both exist (i.e., someone made an
        # incorrect game thread or post game thread. Five minue wait should
        # prevent those threads from being found)
        try:
            game_threads = potential_valid_game_threads[(winner, loser)]
            home_winner = False
        except:
            game_threads = potential_valid_game_threads[(loser, winner)]
            home_winner = True

        if not game_threads:
            # Do a search back a few hours and try to find the game thread.
            # Alternatively, may simply raise an exception
            pass

        # Generate list of games to analyze
        to_analyze.append(determine_game_thread(game_threads))

        # Do some clean up to keep the potential_valid_game_threads
        # list manageable. Problems may occur if a duplicate postgame thread
        # appears long after the game ended and a postgame thread was already
        # found. Unique checks using game_id should prevent problems.
        for thread in game_threads:
            if home_winner:
                del potential_valid_game_threads[(loser, winner)]
            else:
                del potential_valid_game_threads[(winner, loser)]

        # Game_id, Game_thread_id, Post_gamethread_id, Winner, Loser, home_winner

        # Game_thread_id, post_id, post_body


# We may want some parallelization here so we don't miss new threads
analyze_comments(to_analyze)
