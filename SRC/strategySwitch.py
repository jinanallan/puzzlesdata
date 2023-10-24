import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 16
LAEGER_SIZE = 18

plt.rc('axes', titlesize=LAEGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE) 
plt.rc('legend', fontsize=SMALL_SIZE) 


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

pnp_puzzle = np.array([1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26])
switch = np.zeros((len(unique_participants), len(pnp_puzzle)))

for solutions in df.iterrows():
    participant_id = solutions[1]["participant_id"]
    puzzle_id = solutions[1]["puzzle_id"]
    run= solutions[1]["run"]
    attempt = solutions[1]["attempt"]

    id = str(participant_id) + "_"  + str(run) + "_"+ str(puzzle_id) + "_" + str(attempt)

    if puzzle_id in pnp_puzzle :
        cluster_file = f"./Plots_Text/clustering/puzzle{puzzle_id}_POSVEC/cluster_ids_puzzle{puzzle_id}_POSVEC.json"

        with open(cluster_file) as json_file:
            cluster_ids = json.load(json_file)

        for idx, cluster in cluster_ids.items():
            if id in cluster:
                df.at[solutions[0], "cluster_id"] = idx

# print(df)


df_grouped = df.groupby(["participant_id", "puzzle_id"])

for group in df_grouped:
    if group[0][1] in pnp_puzzle:
        #print cluster ids
        # print(group[0])
        if len(group[1]["cluster_id"].unique().tolist()) != 1:  
            print(f'participant {group[0][0]} puzzle {group[0][1]} switch strategy')
            switch[np.where(unique_participants == group[0][0]), np.where(pnp_puzzle == group[0][1])] = 1

columnsun1 = np.sum(switch, axis=0).astype(float)
columnsun1 /= len(unique_participants)

plt.figure(figsize=(20,10))
plt.suptitle('Strategy switch within all attempts and runs (pick and place puzzles)', fontsize=20)

plt.subplot(1, 2, 1)
plt.imshow(switch, cmap='binary')
plt.colorbar()
plt.title("Strategy switch" , fontsize=15)
plt.xlabel("Puzzle ID", labelpad=20)
plt.ylabel("Participant ID", labelpad=20)
plt.xticks(np.arange(len(pnp_puzzle)), pnp_puzzle,rotation=90)
plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
plt.yticks(np.arange(len(unique_participants)), unique_participants)

plt.subplot(1, 2, 2)
plt.bar(height=columnsun1, x=np.arange(len(pnp_puzzle)), color="lightslategray")
plt.title("Switching rate" , fontsize=15)
plt.xlabel("Puzzle ID", labelpad=20)
plt.ylabel("Switching rate", labelpad=20)
plt.xticks(np.arange(len(pnp_puzzle)), pnp_puzzle,rotation=90)

plt.savefig("./Plots_Text/clustering/strategy_switch.png")



                


       


        

