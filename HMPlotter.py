import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import re
import matplotlib.colors as mcolors


def use_regex(input_text):
    pattern = re.compile(
        r"([0-9]+(-[0-9]+)+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)\.json", re.IGNORECASE)
    match = pattern.match(input_text)
    particpants = match.group(3)
    run = match.group(4)
    puzzle_id = match.group(5)
    puzzle_id2 = match.group(6)
    return int(particpants), int(run), int(puzzle_id), int(puzzle_id2)


def movement_of_players(df, participant_id, run):

    events = df["events"]
    df_events = pd.DataFrame(events)
    df_events["code"] = df['events'].apply(lambda x: x.get('code'))
    df_events["timestamp"] = df['events'].apply(lambda x: x.get('timestamp'))
    df_events["x"] = df['events'].apply(lambda x: x.get('x'))
    df_events["y"] = df['events'].apply(lambda x: x.get('y'))
    df_events = df_events.drop('events', axis=1)

    # index of attach and release events so that we can plot the movement of players in different colors
    attachIndex = df_events.index[df_events['code'] == 3].tolist()
    releaseIndex = df_events.index[df_events['code'] == 4].tolist()

    interactions = np.array([])
    for i in range(len(attachIndex)):
        interactions = np.append(
            interactions, (range(attachIndex[i], releaseIndex[i])))
        interactions = interactions.flatten()
    interactions = interactions.astype(int)

    x = np.array([])
    y = np.array([])
    for index, row in df_events.iterrows():
        if index in interactions:
            x = np.append(x, row['x'])
            # x = x.flatten()
            # x = x.astype(int)
            y = np.append(y, row['y'])
            # y = y.flatten()
            # y = y.astype(int)
    return x, y


def main():
    # folder = input("Enter the folder path: ")
    folder = '/home/erfan/Downloads/pnp'
    # desiredpuzzel = int(input("Enter the puzzel number: "))
    desiredpuzzel = 1
    x = np.array([])
    y = np.array([])

    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            participant_id, run, puzzel, puzzel2 = use_regex(filename)
            if puzzel == desiredpuzzel:
                # Load the JSON file
                with open(os.path.join(folder, filename)) as json_file:
                    data = json.load(json_file)
                    df = pd.DataFrame(data)

                # get the movement of players
                xi, yi = movement_of_players(df, participant_id, run)
                x = np.append(x, xi)
                y = np.append(y, yi)
    # print(y.shape)
    # fig, ax = plt.subplots()
    # img = plt.imread('/home/erfan/Downloads/puzzleexamplescreenshots/puzzle2.png')
    # print(img.shape)
    #max of x and y 
    m = max(x.max(), y.max())
    # ax.imshow(img, extent=[-2,2,-2,2])

# fig, ax = plt.subplots()
    # img = plt.imread('/home/erfan/Downloads/puzzleexamplescreenshots/puzzle2.png')
    # print(img.shape)
    # colors = [(0,0,1,c) for c in np.linspace(0,1,100)]
    # cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)
    plt.hist2d(x, y, bins=(60, 60))
    plt.show()


if __name__ == '__main__':
    main()
