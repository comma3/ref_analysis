from collections import defaultdict
import sqlite3
import csv, pickle

def write_list(outputfilename, list):
    """
    Writes a list to a csv.

    PARAMS: filename, list
    """

    with open(outputfilename, 'w', newline='', encoding='utf-8') as outfile:
        for row in list:
            outfile.write(row+'\n')

def get_all_team_names(db='/data/cfb_game_db.sqlite3'):
    """
    """
    query = """SELECT DISTINCT
                home, away
                FROM
                games
                """
    conn = sqlite3.connect(db)
    curr = conn.cursor()
    curr.execute(query)
    teams = curr.fetchall()
    conn.close()
    all_teams = set()
    for home, away in teams:
        all_teams.add(home.strip())
        all_teams.add(away.strip())
    #print(all_teams)
    team_dict = defaultdict(set)
    last_t = ''
    for t in sorted(list(all_teams)):  # needs to be sorted so that shorter names come first
        print(t)
        if last_t in t:
            team_dict[t.replace(last_t, '').strip()].add(last_t) # Get mascot
            if len(t.split()):
                for t in t.split():
                    team_dict[t.strip()].add(last_t)
        last_t = t.replace('- ', '').strip()

    #write_list('team_list.csv', sorted(list(all_teams)))


def get_flairs():
    flairs = pickle.load(open('cfb_flairs.pkl', 'rb'))
    write_list('flair_list.csv', flairs)

if __name__ == '__main__':
    #get_all_team_names()
    get_flairs()
