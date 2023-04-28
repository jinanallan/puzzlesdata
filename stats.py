import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import movementTracker
import HMPlotter
def feuters(df):
    df=df.drop('events', axis=1)
    df=df.drop(df.index[1:])
    return df
puzzle_f=["0-1-1-pnp.g", "0-2-1-pnp.g", "0-3-1-pnp.g", "0-4-1-pnp.g", "0-5-1-pnp.g", "0-6-1-pnp.g", "0-7-1-pnp.g",
            "1-1-1-push.g", "1-2-1-push.g", "1-3-1-push.g", "1-4-1-push.g", "1-6-1-push.g",
            "1-8-1-push.g", "1-9-1-push.g","1-5-1-push.g", "1-7-1-push.g",
            "2-1-1-dyn.g", "2-2-1-dyn.g", "2-3-1-dyn.g", "2-4-1-dyn.g", "2-5-1-dyn.g",
            "3-1-1-glue.g", "3-3-1-glue.g", "3-4-1-glue.g", "3-5-1-glue.g","3-6-1-glue.g", "3-7-1-glue.g", "3-2-1-glue.g" ]

puzzle_f=["../../scenes-30a/"+p for p in puzzle_f]
puzzle_id= np.arange(1, len(puzzle_f)+1)

#Calculating the number of parti solved all the puzzles(even with multiple attempts)
#For those who did not solve all the puzzles, we will calculate the number of puzzles they solved and which puzzles they didn't solved  

#the idea is to list all the puzzels and thier final states in  each attemp for each participant

#looking at the data in two diffrent modes of puzzle, pnp  and push 
pnp_folder = "/home/erfan/Downloads/pnp"
push_folder = "/home/erfan/Downloads/push"

df_main = pd.DataFrame()
df_solves = pd.DataFrame()
for folder in [pnp_folder, push_folder]:
        for filename in os.listdir(folder):
            if filename.endswith(".json"): 
                p,r,i,a=HMPlotter.use_regex(filename)
                if i != 5:
                    with open(os.path.join(folder, filename)) as json_file:
                        data = json.load(json_file)
                        df=movementTracker.df_from_json(data)
                        df=feuters(df)
                        #add this dataframe as a row to the main dataframe
                        df_main=pd.concat([df_main,df], ignore_index=True)
                        #group the data by participant

#mapping the puzzle file to the puzzle id
df_main['puzzle-file']=df_main['puzzle-file'].map(dict(zip(puzzle_f, puzzle_id)))

df_main.drop("start-time", axis=1, inplace=True)
df_main.drop("end-time", axis=1, inplace=True)

df_main=df_main.groupby(['participantID', 'puzzle-file', 'attempt'])
print(df_main.first())





        

