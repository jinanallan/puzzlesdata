import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt

def df_from_json(file):
    file = json.load(open(file))
    try:
        df = pd.DataFrame(file)
    except:
        df = pd.DataFrame(file, index=[0])
    return df
df= pd.read_csv("./Data/df.csv")


unique_participants = df["participant_id"].unique().tolist()
unique_participants = np.array(unique_participants)
unique_puzzles = df["puzzle_id"].unique().tolist()
unique_puzzles = np.array(unique_puzzles)

# go through every row in the dataframe

for 


        

