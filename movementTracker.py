import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

#json file as a pandas dataframe
def df_from_json(file):
    try:
        df = pd.DataFrame(file)
    except :
        df = pd.DataFrame(file, index=[0])
    return df

def interaction(df, participant_id, run,type, sparce=False, direction=False):

    #valid interactions types: box1, obj1, obj2, obj3, obj4
    if type not in ['box1', 'box2', 'obj1', 'obj2', 'obj3', 'obj4', 'total', 'free']: raise ValueError('Invalid interaction type')
    
    try:
        events = df["events"]
        df_events = pd.DataFrame(events)
        df_events["code"] = df['events'].apply(lambda x: x.get('code'))
        df_events["timestamp"] = df['events'].apply(lambda x: x.get('timestamp'))
        #the time stamp (unix time)  
        df_events['timestamp'] = df_events['timestamp'].str.split('-').str[0] 
        df_events['timestamp'] = df_events['timestamp'].astype(int)
        df_events["x"] = df['events'].apply(lambda x: x.get('x'))
        df_events["y"] = df['events'].apply(lambda x: x.get('y'))
        df_events["description"] = df['events'].apply(lambda x: x.get('description'))
        df_events = df_events.drop('events', axis=1)

        if type == 'total' or type == 'free':
            attachIndex = (df_events.index[df_events['description'].str.contains("Attach")]+1).tolist() 
            releaseIndex = (df_events.index[df_events['description'].str.contains("Release")]-1).tolist()
        else:
            attachIndex = (df_events.index[df_events['description'] == "Attach "+type]+1).tolist() 
            releaseIndex = (df_events.index[df_events['description'] == "Release "+type]-1).tolist() 
        

        if direction:
            s=np.array([])
            startTime = df_events.loc[attachIndex, 'timestamp']
            startTime = startTime.values[0]

            for i in range(len(attachIndex)):

                intercation =range(attachIndex[i], releaseIndex[i])
                interactions = interactions.flatten()
                interactions = interactions.astype(int)

                x = np.array([])
                y = np.array([])
                for index, row in df_events.iterrows():
                    if index in interactions: 
                        x = np.append(x, row['x'])
                        y = np.append(y, row['y'])
                geo=nesw(x,y)
                s=np.append(s,type+" "+geo +" "+ str(startTime[i]))
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
    
        
#get the list of unique descriptions in the json file
def get_descriptions(df):  
    events = df["events"]
    df_events = pd.DataFrame(events)
    df_events["description"] = df['events'].apply(lambda x: x.get('description'))
    df_events = df_events.drop('events', axis=1)
    return df_events['description'].unique()

def nesw(x,y):
    #the first implementation is based on looking at the first and last points
    #x and y are numpy arrays
    #returns the direction of movement
    #0: stationary, 1: up, 2: right, 3: down, 4: left
    x_diff = x[-1]-x[0]
    y_diff = y[-1]-y[0]
    direction_step= np.sqrt(x_diff**2+y_diff**2)
    x_diff = x_diff/direction_step
    y_diff = y_diff/direction_step
    #set the direction as north, east, south, west and  their combinations
    if x_diff>0 :
        if y_diff>0:
            direction = "NE"
        else:
            direction = "SE"
    else:
        if y_diff>0:
            direction = "NW"
        else:
            direction = "SW"
    return direction
