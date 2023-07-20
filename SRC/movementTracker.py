import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

# json file as a pandas dataframe


def df_from_json(file):
    try:
        df = pd.DataFrame(file)
    except:
        df = pd.DataFrame(file, index=[0])
    return df


def interaction(df, participant_id, run, type, sparce=False, direction=False, pos=False, solved=False):

    # valid interactions types: box1, obj1, obj2, obj3, obj4
    if type not in ['box1', 'box2', 'obj1', 'obj2', 'obj3', 'obj4', 'total', 'free', 'Glue', 'Unglue']:
        raise ValueError('Invalid interaction type')

    if solved:
        # return the value of the solved key as string
        return str(df['solved'].values[0])
    else:

        try:

            events = df["events"]
            df_events = pd.DataFrame(events)
            df_events["code"] = df['events'].apply(lambda x: x.get('code'))
            df_events["timestamp"] = df['events'].apply(
                lambda x: x.get('timestamp'))
            # the time stamp (unix time)
            df_events['timestamp'] = df_events['timestamp'].str.split(
                '-').str[0]
            df_events['timestamp'] = df_events['timestamp'].astype(np.int64)
            df_events['timestamp'] = pd.to_datetime(
                df_events['timestamp'], unit='us')
            df_events["x"] = df['events'].apply(lambda x: x.get('x'))
            df_events["y"] = df['events'].apply(lambda x: x.get('y'))
            df_events["description"] = df['events'].apply(
                lambda x: x.get('description'))

            df_events = df_events.drop('events', axis=1)

            if type == 'total' or type == 'free':
                attachIndex = (
                    df_events.index[df_events['description'].str.contains("Attach")]+1).tolist()
                releaseIndex = (
                    df_events.index[df_events['description'].str.contains("Release")]-1).tolist()

            else:
                attachIndex = (
                    df_events.index[df_events['description'] == "Attach "+type]+1).tolist()
                releaseIndex = (
                    df_events.index[df_events['description'] == "Release "+type]-1).tolist()

            if direction:
                # here is an array since any type may be interacted multiple times
                s = np.array([])
                if type == 'Glue' or type == 'Unglue':
                    GlueAction = (
                        df_events.index[df_events['description'].str.contains(type)]).tolist()
                    for i in range(len(GlueAction)):
                        x_start = df_events.loc[GlueAction[i], 'x']
                        startTime = df_events.loc[GlueAction[i], 'timestamp']
                        y_start = df_events.loc[GlueAction[i], 'y']
                        # x_end = df_events.loc[GlueAction[i], 'x']
                        # endTime = df_events.loc[GlueAction[i], 'timestamp']
                        # y_end = df_events.loc[GlueAction[i], 'y']
                        glue_description = df_events.loc[GlueAction[i],
                                                         'description']
                        # d=endTime-startTime
                        # d=d.total_seconds()
                        # d=round(d,2)

                        # x=np.array([x_start,x_end])
                        # y=np.array([y_start,y_end])
                        # geo=nesw(x,y)

                        s = np.append(s, str(startTime)+"_"+glue_description)

                if type == 'free':

                    x_start = df_events.loc[0, 'x']
                    startTime = df_events.loc[0, 'timestamp']
                    y_start = df_events.loc[0, 'y']

                    # x_end = df_events.loc[len(df_events)-1, 'x']
                    # endTime = df_events.loc[len(df_events)-1, 'timestamp']
                    # y_end = df_events.loc[len(df_events)-1, 'y']

                    for i in range(len(attachIndex)):

                        x_end = df_events.loc[attachIndex[i]-1, 'x']
                        endTime = df_events.loc[attachIndex[i]-1, 'timestamp']
                        y_end = df_events.loc[attachIndex[i]-1, 'y']

                        d = endTime-startTime
                        d = d.total_seconds()
                        d = round(d, 2)

                        x = np.array([x_start, x_end])
                        y = np.array([y_start, y_end])
                        if pos:
                            xx, yy = PosChange(x, y)
                            geo = str(xx)+" " + str(yy)
                        else:
                            geo = nesw(x, y)

                        s = np.append(s, str(startTime)+"_"+type +
                                      " " + geo + " " + str(d)+"s")

                        x_start = df_events.loc[releaseIndex[i]+1, 'x']
                        startTime = df_events.loc[releaseIndex[i] +
                                                  1, 'timestamp']
                        y_start = df_events.loc[releaseIndex[i]+1, 'y']

                    return s

                else:
                    for i in range(len(attachIndex)):
                        x_start = df_events.loc[attachIndex[i], 'x']
                        startTime = df_events.loc[attachIndex[i], 'timestamp']
                        y_start = df_events.loc[attachIndex[i], 'y']
                        x_end = df_events.loc[releaseIndex[i], 'x']
                        endTime = df_events.loc[releaseIndex[i], 'timestamp']
                        y_end = df_events.loc[releaseIndex[i], 'y']

                        d = endTime-startTime
                        d = d.total_seconds()
                        d = round(d, 2)

                        x = np.array([x_start, x_end])
                        y = np.array([y_start, y_end])
                        if pos:
                            xx, yy = PosChange(x, y)
                            geo = str(xx)+" " + str(yy)

                        else:
                            geo = nesw(x, y)

                        s = np.append(s, str(startTime)+"_"+type +
                                      " " + geo + " " + str(d)+"s")

                    return s

            else:

                interactions = np.array([])
                for i in range(len(attachIndex)):
                    interactions = np.append(
                        interactions, (range(attachIndex[i], releaseIndex[i])))
                    interactions = interactions.flatten()
                interactions = interactions.astype(int)

                x = np.array([])
                y = np.array([])
                for index, row in df_events.iterrows():
                    if type == 'free':
                        if index not in interactions:
                            x = np.append(x, row['x'])
                            y = np.append(y, row['y'])
                    elif index in interactions:
                        x = np.append(x, row['x'])
                        y = np.append(y, row['y'])

                if sparce:
                    return x[::2], y[::2]

                else:
                    return x, y

        except:
            return np.array([]), np.array([])


# get the list of unique descriptions in the json file
def get_descriptions(df):
    events = df["events"]
    df_events = pd.DataFrame(events)
    df_events["description"] = df['events'].apply(
        lambda x: x.get('description'))
    df_events = df_events.drop('events', axis=1)
    return df_events['description'].unique()


def nesw(x, y):
    x_diff = x[-1]-x[0]
    y_diff = y[-1]-y[0]
    direction_step = np.sqrt(x_diff**2+y_diff**2)
    if direction_step != 0:
        x_diff = x_diff/direction_step
        y_diff = y_diff/direction_step

    # transform the [x_diff, y_diff] to angle in degrees
    angle = np.arctan2(y_diff, x_diff) * 180 / np.pi
    # print(angle)
    angle = angle % 360
    angle = round(angle, 0)

    if angle in range(0, 30) or angle in range(330, 360):
        direction = "E"
    elif angle in range(30, 60):
        direction = "NE"
    elif angle in range(60, 120):
        direction = "N"
    elif angle in range(120, 150):
        direction = "NW"
    elif angle in range(150, 210):
        direction = "W"
    elif angle in range(210, 240):
        direction = "SW"
    elif angle in range(240, 300):
        direction = "S"
    elif angle in range(300, 330):
        direction = "SE"
    return direction


def PosChange(x, y):
    x_diff = x[-1]-x[0]
    y_diff = y[-1]-y[0]
    direction_step = np.sqrt(x_diff**2+y_diff**2)
    if direction_step != 0:
        x_diff = x_diff/direction_step
        y_diff = y_diff/direction_step
        x_diff = round(x_diff, 2)
        y_diff = round(y_diff, 2)
    return x_diff, y_diff
 