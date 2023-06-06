import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#json file as a pandas dataframe
def df_from_json(file):
    try:
        df = pd.DataFrame(file)
    except :
        df = pd.DataFrame(file, index=[0])
    return df

def interaction(df, participant_id, run, listed=False):

    
    try:
    
        events = df["events"]
        df_events = pd.DataFrame(events)
        df_events["code"] = df['events'].apply(lambda x: x.get('code'))
        df_events["timestamp"] = df['events'].apply(lambda x: x.get('timestamp'))
        #the time stamp (unix time)  
        df_events['timestamp'] = df_events['timestamp'].str.split('-').str[0] 
        df_events['timestamp'] = df_events['timestamp'].astype(int)
        df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit='us')
        df_events["x"] = df['events'].apply(lambda x: x.get('x'))
        df_events["y"] = df['events'].apply(lambda x: x.get('y'))
        df_events["description"] = df['events'].apply(lambda x: x.get('description'))

        df_events = df_events.drop('events', axis=1)

        for index, row in df_events.iterrows():
            if row['description'] == "Moving started ":
                df_events.loc[index, 'description'] = 'free'


        attachIndex = (df_events.index[df_events['description'].str.contains("Attach")]).tolist()
        releaseIndex = (df_events.index[df_events['description'].str.contains("Release")]).tolist()

        for i in range(len(attachIndex)):
            df_events.loc[attachIndex[i]:releaseIndex[i],'description']= df_events.loc[attachIndex[i],'description'].split(" ")[1]
  
            
            

        time_stamp=df_events['timestamp'].values
        x=df_events['x'].values
        y=df_events['y'].values
        description=df_events['description'].values

        interaction_list=[]
        if listed==True:
            #compute the duration of each interaction
            state=description[0]
            start=time_stamp[0]
            for i in range(1,len(description)):
                if description[i]!=state:
                    end=time_stamp[i-1]
                    duration=end-start
                    duration=duration.astype('timedelta64[ns]').astype(int)/1000000000
                    if duration !=0: interaction_list.append([state,duration])
                    state=description[i]
                    start=time_stamp[i]
            return interaction_list



        

        return x,y,description
    


            

    except:
        return None