import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

#json file as a pandas dataframe
def df_from_json(file):
    df = pd.DataFrame(file)
    return df

def interaction(df, participant_id, run,type):
    #valid interactions types: Attach box1, Attach obj1, Attach obj2, Attach obj3

    if type not in ['Attach box1', 'Attach box2', 'Attach obj1', 'Attach obj2', 'Attach obj3', 'Attach obj4']: raise ValueError('Invalid interaction type')

    events = df["events"]
    df_events = pd.DataFrame(events)
    df_events["code"] = df['events'].apply(lambda x: x.get('code'))
    df_events["timestamp"] = df['events'].apply(lambda x: x.get('timestamp'))
    df_events["x"] = df['events'].apply(lambda x: x.get('x'))
    df_events["y"] = df['events'].apply(lambda x: x.get('y'))
    df_events["description"] = df['events'].apply(lambda x: x.get('description'))
    df_events = df_events.drop('events', axis=1)

    # index of attach and release events so that we can plot the movement of players in different colors
    
    attachIndex = (df_events.index[df_events["description"]== type]+1).tolist()
    releaseIndex = (df_events.index[df_events['code'] == 4]-1).tolist() 

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
