from library import *


def get_label(thread, comments, already_analyzed):
    """
    """
    if thread in already_analyzed:
        return already_analyzed, ''

    training_data = []
    num_comments = len(comments)
    for i, comment in enumerate(comments):
        category = ''
        print('==========================\n{} of {} Comments\nFlair: {}\n\nBody:\n{}'.format(i, num_comments, comment.author_flair_text, comment.body))
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
S (for sorry) indicates commenter pulls for other team.
E for excuse
D for disagree/dumb
M for miss
----------------------------------------------
0 - Not ref related*
1 - General complaint*
2 - Nonspecific bad call*
3 - Play call* - need to differentiate bad call from bad play call
4 - Pass Interference*
5 - Imagined flag (shoes, gloves, etc.)*
6 - Facemask*
7 - Pick 6*
14 - Late hit*
17 - Pick/rub*
8 - Sarcastic correct call
9 - Nonspecific correct call
10 - Holding*
24 - Fumble*
12 - Bad spot
13 - Overturned
14 - False Start
15 - Incomplete pass
16 - Complete pass
18 - Targetting
19 - Grounding
20 - Illegal formation
21 - pick gambling (pick em)

8 - Unnecessary Roughness/Unsportsmanlike conduct
9 - Offsides
10 - False Start
11 - Encroachment


13 - Chop Block

15 - Grounding
16 - Hands to the face

18 - Clipping
19 - Illegal motion/shift
20 - Illegal substitution/too many men
21 - Illegal forward pass

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
        comments = load_data(thread[1], verbose=False)
        already_analyzed, quit = get_label(thread[1], comments, already_analyzed)
        if quit == 'quit':
            pickle.dump(already_analyzed, open(analyzed_list_path, 'wb'))
            break
        print('++++++++++++++++++++++++++\n\t\t\t\tNEW GAME\n++++++++++++++++++++++++++')
