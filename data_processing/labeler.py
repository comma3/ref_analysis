import sqlite3

from library import *



def get_label(thread, comments, home, away, already_analyzed):
    """
    """
    if thread in already_analyzed:
        return already_analyzed, ''

    training_data = []
    num_comments = len(comments)
    for i, comment in enumerate(comments):
        # Replace the names and nicknames of the team with home/away tag
        comment = sub_home_away(comment, home, away)
        category = ''
        print('==========================\n{} of {} Comments - Home: {} Away: {}\nFlair: {}\n\nBody:\n{}'.format(i, num_comments, home, away, comment.author_flair_text, comment.body))
        while not is_cat_valid(category):
            category = input('==========================\nEnter class: ')
            if category.lower() == 'quit':
                # Add here so that we can stop some games if they are long.
                # Don't want team specific things getting picked up (teams and
                # nicknames should be on stop_words list)
                already_analyzed.add(thread)
                return already_analyzed, 'quit'
            elif category.lower() == 'redo':
                category = ''
                while not is_cat_valid(category):
                    category = input('==========================\nEnter class of previous comment: ')
                if not len(training_data):
                    print('Sorry already committed to DB')
                else:
                    temp_comment, temp_thread, temp_flair, temp_body, temp_time, _ = training_data.pop()
                    training_data.append((temp_comment, temp_thread, temp_flair, temp_body, temp_time, category))
                category = ''
                print('Now enter category for the following:\n{} of {} Comments\nFlair: {}\n\nBody:\n{}'.format(i, num_comments, comment.author_flair_text, comment.body))
            elif category.lower() == 'modify':

                to_replace = input('Replace which word? ')
                homeaway = input('with home or away? ')
                if homeaway == 'home':
                    comment.replace(to_replace, 'hometeamtrack')
                elif homeawy == 'away':
                    comment.replace(to_replace, 'awayteamtrack')
                category = input('==========================\nEnter class: ')

        training_data.append((str(comment), thread, comment.author_flair_text, comment.body, comment.created_utc, category))
        if i % 25 == 0:
            # Add periodically
            add_to_db(training_data)
            training_data = []
    # Also add when the loop is over in case it's not divisible by 25.
    add_to_db(training_data)
    already_analyzed.add(thread)

    return already_analyzed, ''

def is_cat_valid(string):
    """
    """
    if not string or string == '00':
        return False
    for cat in string.split(','):
        cat = cat.strip()
        if cat in 'MEDS':
            continue
        try:
            num = int(cat)
        except ValueError:
            print('Bad category. Try again or type "quit" to exit.')
            return False
        if num < 0 or num > 30:
            print('Bad category. Try again or type "quit" to exit.')
            return False
        for c in cat:
            if c not in '0123456789MEDS':
                print('Bad category. Try again or type "quit" to exit.')
                return False
    return True

def add_to_db(training_data, db = '/data/cfb_game_db.sqlite3'):
    """
    """
    conn = sqlite3.connect(db)
    curr = conn.cursor()
    # Just gonna create the table here so I don't have to do it elsewhere
    # Pass if the table already exists
    # (str(comment), thread, comment.author_flair_text, comment.body, comment.created_utc, category)
    try:
        curr.execute("""CREATE TABLE
                        training_data (
                        comment_id string,
                        game_thread string,
                        author_flair string,
                        body string,
                        time int,
                        category string
                        );
                        """)
        conn.commit()
    except sqlite3.OperationalError:
        # Totally fine that table already exists
        pass

    curr.executemany("""INSERT INTO training_data
                    (
                    comment_id,
                    game_thread,
                    author_flair,
                    body,
                    time,
                    category
                    )
                    VALUES
                    (?,?,?,?,?,?);
                    """, training_data)
    conn.commit()
    conn.close()

    print('Sucessfully added to DB!')


"""
Categories
S - for sorry. indicates commenter pulls for other team.
SH - sorry to the home team - explicitly labeled (e.g., sorry cougs, you got screwed)
SA - sorry to the away team - explicitly labeled
G - for got away with one
GH - home got lucky
GA - away got lucky
E - for excuse, make up call, etc.
D - for dumb, admits its a correct call
C - admits its a correct call
M - for miss
R - for reviewed - may help remove complaints about bad calls that were overturned
RC - original was correct
RR - call reversed
----------------------------------------------
0 - Not ref related
1 - General complaint - try to find complaints not associated with a specific call
2 - Nonspecific bad call - different from 1 in that it refers to a recent event not just "refs are bad"
3 - Imagined flag - shoes, gloves, phantom, etc.
4 - Play call - need to differentiate bad call from bad play call

5 - Pass Interference
6 - Facemask
7 - Pick 6/Interception - should generic interception be labeled or only to try and differentiate pick route from pick(interception)?
8 - Pick gambling (pick em)
9 - Pick/Rub Route

10 - False Start
11 - Offsides
12 - Holding
13 - Encroachment

13 - Bad spot
14 - Incomplete pass
15 - Complete pass
16 - Grounding
17 - Illegal forward pass

17 - Late hit
18 - Targetting





20 - Illegal formation

24 - Fumble

8 - Unnecessary Roughness/Unsportsmanlike conduct

13 - Chop Block

16 - Hands to the face
18 - Clipping
19 - Illegal motion/shift
20 - Illegal substitution/too many men

25 - Not fumble
26 - Roughing the passer
27 - Roughing/running into the kicker
28 - Roughing the snapper
29 - Tripping
30 - Sideline infraction
31 - Horse Collar
"""

if __name__ == '__main__':

    analyzed_list_path = 'already_analyzed.pkl'

    if os.path.isfile(analyzed_list_path):
        already_analyzed = pickle.load(open(analyzed_list_path, 'rb'))
    else:
        already_analyzed = set()

    game_list = collect_game_threads()
    print(already_analyzed)
    for thread in game_list:
        home, away = thread[2], thread[3]
        comments = load_data(thread[1], verbose=False)
        already_analyzed, quit = get_label(thread[1], comments, home, away, already_analyzed)
        if quit == 'quit':
            pickle.dump(already_analyzed, open(analyzed_list_path, 'wb'))
            break
        print('++++++++++++++++++++++++++\n\t\t\t\tNEW GAME\n++++++++++++++++++++++++++')
