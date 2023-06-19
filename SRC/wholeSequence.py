import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


#json file as a pandas dataframe
def df_from_json(file):
    try:
        df = pd.DataFrame(file)
    except :
        df = pd.DataFrame(file, index=[0])
    return df

def label_encoder(labels, set_of_all_possible_interactions):

    # print(labels)
    onehot_encoder = OneHotEncoder(categories=[set_of_all_possible_interactions],sparse_output=False)
    labels = np.array(labels).reshape(-1,1) 
    # print(labels)
    onehot_encoded=onehot_encoder.fit_transform(labels)
    transformed_description=onehot_encoded
    transformed_description=np.array(transformed_description,dtype=np.double)
    transformed_description=transformed_description.T
    return transformed_description

def interaction(df, participant_id, run, listed=False, transformed=False):

    
    # try:
    
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
            elif row['description'] == "Left click":
                df_events.loc[index, 'description'] = 'free'
            elif row['description'].startswith("Glue"):
                df_events.loc[index, 'description'] = 'Glue'
            elif row['description'].startswith("Unglue"):
                df_events.loc[index, 'description'] = 'Unglue'

        # df_events = df_events[df_events['description'] != 'Moving started ']

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
        
        elif transformed==True:
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
            
            interaction_list=np.array(interaction_list)
            labels=interaction_list[:,0]
            labels=list(labels)
            # interaction_list=np.array(interaction_list)
            transformed_labels=label_encoder(labels)
            #vstack the duration of each interaction with the corresponding label
            transformed_interaction_list=np.vstack((interaction_list[:,1],transformed_labels))
            transformed_interaction_list=np.array(transformed_interaction_list,dtype=np.double)
            
            return transformed_interaction_list

        else:
            return x,y,description

    # except:
    #     print("Error in interaction function")
    #     return None