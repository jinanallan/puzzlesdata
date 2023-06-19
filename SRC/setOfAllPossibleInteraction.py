import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import movementTracker
import HMPlotter


def possibleInteraction(puzzleNumber):


        folder = '/home/erfan/Downloads/pnp'

        # pnp puzzle number: 1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26
    

        set_of_all_possible_interactions = set()
        for filename in sorted(os.listdir(folder)):
            if filename.endswith('.json'):

                #the participant id, run, puzzle, attempt from the file name:
                participant_id, run, puzzle, attempt = HMPlotter.use_regex(filename)

                if puzzleNumber == puzzle :
                    

                    with open(os.path.join(folder, filename)) as json_file:

                        data = json.load(json_file)
                        df=movementTracker.df_from_json(data)

                        #get the list of all possible interactions:
                        try: 
                            events = df["events"] 
                            
                            df_events = pd.DataFrame(events)
                            df_events["description"] = df['events'].apply(lambda x: x.get('description'))

                            attachIndex = (df_events.index[df_events['description'].str.contains("Attach")]).tolist()
                            releaseIndex = (df_events.index[df_events['description'].str.contains("Release")]).tolist()

                                                
                            for i in range(len(attachIndex)):
                                df_events.loc[attachIndex[i]:releaseIndex[i],'description']= df_events.loc[attachIndex[i],'description'].split(" ")[1]
                            
                    
                            for index, row in df_events.iterrows():
                                if row["description"] == "Moving started ":
                                    df_events.at[index, "description"] = "free"
                                elif row["description"] == "Left click":
                                    df_events.at[index, "description"] = "free"
                                elif row["description"].startswith('Glue'):
                                    df_events.at[index, "description"] = "Glue"
                                elif row["description"].startswith('Unglue'):
                                    df_events.at[index, "description"] = "Unglue"



                                    

                            possible_interactions = df_events["description"].unique()
                            #add the list of all possible interactions to a set:
                            for i in range(len(possible_interactions)):
                                set_of_all_possible_interactions.add(possible_interactions[i])
                            
                        except:
                            pass

        set_of_all_possible_interactions=list(set_of_all_possible_interactions)
        return set_of_all_possible_interactions
