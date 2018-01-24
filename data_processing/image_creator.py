import sqlite3, glob
from collections import defaultdict

import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox

from library import get_games_dict

def plotImage(x, y, im, x_scale, y_scale):
    """
    Plots an image at each x and y location.
    """
    bb = Bbox.from_bounds(x, y, x_scale, y_scale) # Change figure aspect ratio
    bb2 = TransformedBbox(bb,ax.transData)
    bbox_image = BboxImage(bb2,
                        norm = None,
                        origin=None,
                        clip_on=False)

    bbox_image.set_data(im)
    ax.add_artist(bbox_image)

def get_figure_points():
    """
    """

    conn = sqlite3.connect('/data/cfb_game_db.sqlite3')
    df = pd.read_sql_query("""
                    SELECT
                    game_thread, team_affected, count(team_affected) as count
                    FROM
                    calls
                    GROUP BY
                    game_thread, team_affected
                    ORDER BY
                    game_thread;
                    """, conn)

    aggressiveness = pd.read_sql_query("""
                    SELECT
                    team, CAST(argumentative AS float) / CAST(positive AS float) as agg
                    FROM
                    fanbase;
                    """, conn).fillna(0)

    conn.close
    game_dict = get_games_dict()

    calls_for = defaultdict(int)
    calls_against = defaultdict(int)
    games = defaultdict(int)
    groups = df.groupby('game_thread')
    for group in groups:
        teams = game_dict[group[0]]
        for team in teams:
            team = [x for x in team][0]
            games[team] += 1
            for i, row in group[1].iterrows():
                if row['team_affected'] in team:
                    calls_against[team] += row['count']
                else:
                    calls_for[team] += row['count']

    keys = set()
    for k in calls_for.keys():
        keys.add(k)
    for k in calls_against.keys():
        keys.add(k)

    differentials = []
    scaled = []
    output = []
    for k in keys:
        differential = calls_for[k] - calls_against[k]
        differentials.append((differential, k))
        scaled.append((k, round(differential/games[k], 3)))
        output.append((k, round(differential/games[k], 3), aggressiveness[aggressiveness['team'] == k]['agg'].values[0]))

    return output


if __name__ == '__main__':

    markers = {}
    for image in glob.glob('logos/*.png'):
        clean_name = image.split('/')[1].replace('logo', '').replace('.png', '').replace('-', ' ').strip()
        markers[clean_name] = plt.imread(image)

    points = get_figure_points()

    # Create figure
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', which='major', labelsize=20)
    for team, x, y in points:
        try:
            plotImage(x, y, markers[team],0.3,0.04)
        except KeyError:
            pass

    # Set the x and y limits
    ax.set_ylim(-0.03,0.4)
    ax.set_xlim(-2.2,6)

    plt.xlabel('Average Call Differential').set_fontsize(26)
    plt.ylabel('Argumentativeness').set_fontsize(26)

    plt.show()
