from collections import defaultdict
import sqlite3

def get_all_team_names(db='/data/cfb_game_db.sqlite3'):
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
        all_teams.add(home)
        all_teams.add(away)
    #print(all_teams)
    team_dict = defaultdict(list)
    last_t = ''
    for t in sorted(list(all_teams)):
        if last_t in t:
            team_dict[t.replace(last_t, '').strip()].append(last_t)
            if len(t.split()):
                for t in t.split():
                    team_dict[t.strip()].append(last_t)
        last_t = t.replace('- ', '').strip()

    print(team_dict)

if __name__ == '__main__':
    get_all_team_names()
