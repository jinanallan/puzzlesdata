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

    #valid interactions types: box1, obj1, obj2, obj3, obj4
    if type not in ['box1', 'box2', 'obj1', 'obj2', 'obj3', 'obj4', 'total', 'free']: raise ValueError('Invalid interaction type')

    events = df["events"]
    df_events = pd.DataFrame(events)
    df_events["code"] = df['events'].apply(lambda x: x.get('code'))
    df_events["timestamp"] = df['events'].apply(lambda x: x.get('timestamp'))
    df_events["x"] = df['events'].apply(lambda x: x.get('x'))
    df_events["y"] = df['events'].apply(lambda x: x.get('y'))
    df_events["description"] = df['events'].apply(lambda x: x.get('description'))
    df_events = df_events.drop('events', axis=1)


    # index of attach and release events so that we can plot the movement of players in different colors
    if type == 'total' or type == 'free':
        attachIndex = (df_events.index[df_events['description'].str.contains("Attach")]+1).tolist() 
        releaseIndex = (df_events.index[df_events['description'].str.contains("Release")]-1).tolist()
    else:
        attachIndex = (df_events.index[df_events['description'] == "Attach "+type]+1).tolist() 
        releaseIndex = (df_events.index[df_events['description'] == "Release "+type]-1).tolist() 

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
    return x, y

#get the list of unique descriptions in the json file
def get_descriptions(df):
    events = df["events"]
    df_events = pd.DataFrame(events)
    df_events["description"] = df['events'].apply(lambda x: x.get('description'))
    df_events = df_events.drop('events', axis=1)
    return df_events['description'].unique()
