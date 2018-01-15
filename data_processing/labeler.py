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
                return already_analyzed, 'quit'
        training_data.append((thread, str(comment), comment.author_flair_text, comment.body, comment.created_utc, category))

    add_to_db(training_data)

    already_analyzed.add(thread)

    return already_analyzed, ''

def is_cat_valid(string):
    """
    """
    if not string:
        return False
    if 'S' in string:
        try:
            num = int(string[1:])
        except ValueError:
            print('Bad category. Try again or type "quit" to exit.')
            return False
    else:
        try:
            num = int(string)
        except ValueError:
            print('Bad category. Try again or type "quit" to exit.')
            return False
    if num < -30 or num > 30:
        print('Bad category. Try again or type "quit" to exit.')
        return False
    for c in string:
        if c not in '0123456789S-':
            print('Bad category. Try again or type "quit" to exit.')
            return False
    return True

def add_to_db(training_data):
    """
    """
    print(training_data)


"""
Categories
Negative with the same label means missed call
S (for sorry) indicates commenter pulls for other team.
----------------------------------------------
0 - Not ref related
1 - General complaint
2 - Nonspecific call
3 - Holding
4 - Pass Interference
5 - Block in the back
6 - Facemask
7 - Targeting
8 - Unnecessary Roughness/Unsportsmanlike conduct
9 - Offsides
10 - False Start
11 - Horse Collar
12 - Bad spot
13 - Chop Block
14 - Encroachment
15 - Grounding
16 - Hands to the face
17 - Pick/rub
18 - Clipping
19 - Illegal motion/shift
20 - Illegal substitution/too many men
21 - Illegal forward pass
22 - Incomplete pass
23 - Complete pass
24 - Fumble
25 - Not fumble
26 - Roughing the passer
27 - Roughing/running into the kicker
28 - Roughing the snapper
29 - Tripping
30 - Sideline infraction
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
