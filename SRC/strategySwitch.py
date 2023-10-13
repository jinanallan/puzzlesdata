import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import ast

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

strategy_run1 = np.empty((len(unique_participants), len(unique_puzzles)))


for solutions in df.iterrows():
    participant_id = solutions[1]["participant_id"]
    puzzle_id = solutions[1]["puzzle_id"]
    run= solutions[1]["run"]
    attempt = solutions[1]["attempt"]

    id = str(participant_id) + "_"  + str(run) + "_"+ str(puzzle_id) + "_" + str(attempt)

    if puzzle_id in [1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26]:
        cluster_file = f"./Plots_Text/clustering/puzzle{puzzle_id}_POSVEC/cluster_ids_puzzle{puzzle_id}_POSVEC.json"

        with open(cluster_file) as json_file:
            cluster_ids = json.load(json_file)

        for idx, cluster in cluster_ids.items():
            if id in cluster:
                print( id, idx)
                # strategy_switch[unique_participants == participant_id, unique_puzzles == puzzle_id] = idx



       


        

