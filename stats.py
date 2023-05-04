import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import movementTracker
import HMPlotter
def dropEvents(df):
    try:
        df=df.drop('events', axis=1)
        df=df.drop(df.index[1:])
    except:
        pass
    return df

puzzle_f=[  "0-1-1-pnp.g", "0-2-1-pnp.g", "0-3-1-pnp.g", "0-4-1-pnp.g", "0-5-1-pnp.g", "0-6-1-pnp.g", "0-7-1-pnp.g",
            "1-1-1-push.g", "1-2-1-push.g", "1-3-1-push.g", "1-4-1-push.g", "1-6-1-push.g","1-8-1-push.g", "1-9-1-push.g",
            "1-5-1-push.g", "1-7-1-push.g","2-1-1-dyn.g", "2-2-1-dyn.g", "2-3-1-dyn.g", "2-4-1-dyn.g", "2-5-1-dyn.g",
            "3-1-1-glue.g", "3-3-1-glue.g", "3-4-1-glue.g", "3-5-1-glue.g","3-6-1-glue.g", "3-7-1-glue.g", "3-2-1-glue.g" ]

puzzle_f=["../../scenes-30a/"+p for p in puzzle_f]
puzzle_id= range(len(puzzle_f))

#Calculating the number of parti solved all the puzzles(even with multiple attempts)
#For those who did not solve all the puzzles, we will calculate the number of puzzles they solved and which puzzles they didn't solved  

#the idea is to list all the puzzels and thier final states in  each attemp for each participant

#looking at the data in two diffrent modes of puzzle, pnp  and push 
pnp_folder = input("Enter the path to the pnp folder: ")
# "/home/erfan/Downloads/pnp"

push_folder = input("Enter the path to the push folder: ")
# "/home/erfan/Downloads/push"

df_main = pd.DataFrame()

for folder in [pnp_folder, push_folder]:
        for filename in os.listdir(folder):
            if filename.endswith(".json"): 
                p,r,i,a=HMPlotter.use_regex(filename)
                if True:
                    with open(os.path.join(folder, filename)) as json_file:
                        data = json.load(json_file)
                        df=movementTracker.df_from_json(data)
                        df=dropEvents(df)
                        df_main=pd.concat([df_main,df], ignore_index=True)

df_main.drop("start-time", axis=1, inplace=True)
df_main.drop("end-time", axis=1, inplace=True)

df_main['puzzle-file']=df_main['puzzle-file'].map(dict(zip(puzzle_f, puzzle_id)))
df_main['participantID']=df_main['participantID'].astype(int)

puzzles=df_main['puzzle-file'].unique()
participants=df_main['participantID'].unique()
particpant=participants.sort()
df_main['participantID']=df_main['participantID'].map(dict(zip(participants, range(len(participants)))))

df_grouped=df_main.groupby(['participantID', 'puzzle-file', 'run','attempt'])

solved=np.zeros((len(participants), len(puzzle_id)))

for name , group in df_grouped:
     for row_index, row in group.iterrows():
          if row['solved'] == True:  
             solved[row['participantID'], row['puzzle-file']]=1




solved_all=np.where(np.sum(solved, axis=1)==len(puzzles))
solved_all=solved_all[0]
solved_all=participants[solved_all]

notall=np.where(np.sum(solved, axis=1)!=len(puzzles))
notall=notall[0]

print("The number of participants who solved all the puzzles:", len(solved_all), "out of",len(participants))
print("The participants who eventually solved all the puzzles: ", solved_all)
print ("The participants who did not solve all the puzzles: ", np.setdiff1d(participants, solved_all))

for i in notall:
     print("The participant", participants[i], "solved", np.sum(solved[i]).astype(int), "puzzles out of", len(puzzles))
     print("The puzzles that the participant", participants[i], "did not solve are:", np.setdiff1d(puzzles, np.where(solved[i]==1)))
                 
          
      









        

