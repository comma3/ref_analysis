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


# Make sure we aren't doubling up on games
# Should use ESPN game id to ensure uniqueness
# postgame threads are supposed to contain links to box scores,
# so we will collect them from there.
game_ids = set()
# conn = sqlite3.connect("D:\\reddit\\reddit_user_data.sqlite3")
# curr = conn.cursor()
# curr.execute("""SELECT username
#                 FROM users
#                 """)
# temp_games = curr.fetchall()
# conn.close()c
#
# for game in temp_games:
#     game_id.add(game[0])

# Create bot instance
reddit = praw.Reddit('bot1')

# Store game threads in case there are multiple threads for a game.
# Clear this at some interval (maybe daily at like 4 am to limit maintanence).
potential_valid_game_threads = []

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


    is_game_thread = '[game thread]' in post.lower()
    is_postgame_thread ='[postgame thread]' in post.lower()

    if is_game_thread:
        if not post.is_self:
            raise "Game thread was not a selfpost. Post id: {}".format(post.id)

        away, home = re.findall(r'([A-z -]+) @ ([A-z -]+)'), post.lower())
        potential_valid_game_threads[post.id] = (away, home)

    # Check for postgame threads in new posts
    # We are only going to act on postgame threads that are older than 5 minutes
    # so that we ensure that we have the correct game
    elif is_postgame_thread:
        # We can start analyzing the game thread because commenting is effectly
        # finished
        if not post.is_self:
            raise "Postgame thread is not a selfpost. Post id: {}".format(post.id)

        try:
            game_id = re.findall(r'gameId=([0-9]+)', post.selftext)[0]
        except: # TODO: determine the error
            raise "No ESPN link in postgame thread id {}".format(post.id)

        winner, loser = re.findall(r'([A-z -]+) defeats ([A-z -]+)'), post.lower())

        game_threads = [(thread, winner, loser) for thread, teams in potential_valid_game_threads.items() if away in teams and home in teams]

        # Keep track of whether the winning team was home or away
        home_winner = winner == gamethreads[0][1]

        if not game_threads:
            # Do a search back a few hours and try to find the game thread.
            pass

        # Generate list of games to analyze
        to_analyze.append(determine_game_thread(game_threads))

        # Do some clean up.
        # This should keep the potential_valid_game_threads list manageable.
        # Problems may occur if a duplicate postgame thread appears long
        # after the game ended and a  postgame thread was already found.
        for thread in game_threads:
            del potential_valid_game_threads[thread]

        # Game_id, Game_thread_id, Post_gamethread_id, Winner, Loser, Winner_H


# We may want some parallelization here so we don't miss new threads
analyze_comments(to_analyze)






# # Just gonna create the table here so I don't have to do it elsewhere
# # Pass if the table already exists
# # This is just me being lazy and it's nice to see the table structure here
# try:
#     curr.execute("""CREATE TABLE
#                     users (
#                     username string,
#                     gender string,
#                     age int,
#                     subreddits string
#                     )
#                     """)
#     curr.execute("""CREATE TABLE
#                     comments (
#                     user string,
#                     body string,
#                     score int,
#                     date string,
#                     subreddit string
#                     )
#                     """)
#
# except sqlite3.OperationalError:
#     pass



# for key, value in users.items():
#     user = (key, value[1], value[2], ','.join(value[3]))
#     curr.execute("""INSERT INTO users
#                 (username, gender, age, subreddits)
#                 VALUES (?,?,?,?)
#                 """, user)
#     for comment in value[0]:
#         curr.execute("""INSERT INTO comments
#                     (user, body, score, date, subreddit)
#                     VALUES (?,?,?,?,?)
# #                     """, (key, comment[0], comment[1], comment[2], comment[3]))
# conn.commit()
# conn.close()
# print('Added to DB!')