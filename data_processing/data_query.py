import sqlite3


def get_differentials(db='/data/cfb_game_db.sqlite3'):

    conn = sqlite3.connect(db)
    curr = conn.cursor()
    curr.execute("""
                SELECT
                game_thread, team_affected, count(team_affected)
                FROM
                calls
                GROUP BY
                game_thread, team_affected
                ORDER BY
                game_thread;
                """)

    differentials = curr.fetchall()
    conn.close()
    return differentials

def get_argumentative(db='/data/cfb_game_db.sqlite3'):

    conn = sqlite3.connect(db)
    curr = conn.cursor()
    curr.execute("""
                SELECT
                team, CAST(argumentative AS float) / CAST(positive AS float)
                FROM
                fanbase;
                """)

    argumentative = curr.fetchall()
    conn.close()
    return argumentative

def get_whininess(db='/data/cfb_game_db.sqlite3'):

    conn = sqlite3.connect(db)
    curr = conn.cursor()
    curr.execute("""
                SELECT
                team, whiny
                FROM
                fanbase;
                """)

    whiny = curr.fetchall()
    conn.close()
    return whiny
