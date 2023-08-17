import pandas as pd
import numpy as np
import re
import os
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
plt.rcParams.update({'figure.autolayout': True})

def use_regex_frames(input_text):
    pattern = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)", re.IGNORECASE)

    match = pattern.match(input_text)
    
    particpants = match.group(3)
    run = match.group(4)
    puzzle_id = match.group(5)
    attempt = match.group(6)
    return int(particpants), int(run), int(puzzle_id), int(attempt)

def df_from_json(file):
    file = json.load(open(file))
    try:
        df = pd.DataFrame(file)
    except:
        df = pd.DataFrame(file, index=[0])
    return df

df= pd.read_csv("./Data/df.csv")

unique_participants = df["participant_id"].unique().tolist()
unique_participants= np.array(unique_participants)
unique_puzzles = df["puzzle_id"].unique().tolist()
unique_puzzles = np.array(unique_puzzles)
sol_matrix1 = np.zeros((len(unique_participants), len(unique_puzzles)))
sol_matrix2 = np.zeros((len(unique_participants), len(unique_puzzles)))


for pilot in [3,4]:

    folder = "./Data/Pilot{}/Ego-based/".format(pilot)

    for file in os.listdir(folder):
        if file.endswith(".json"):

            particpants, run, puzzle_id, attempt = use_regex_frames(file)
            df = df_from_json(folder+file)
            df = df.iloc[0]

            total_time= df["total-time"]
            solved= df['solved'] 

            particpants = np.where(unique_participants == particpants)[0][0]
            puzzle_id = np.where(unique_puzzles == puzzle_id)[0][0]

            if solved:
                if run == 1:
                    sol_matrix1[particpants, puzzle_id] = total_time
                else:
                    sol_matrix2[particpants, puzzle_id] = total_time
            else:
                if run == 1:
                    sol_matrix1[particpants, puzzle_id] = -1
                else:
                    sol_matrix2[particpants, puzzle_id] = -1

plt.figure(figsize=(20,15))
plt.suptitle('Time Solved [s]', fontsize=20)
plt.subplot(1, 2, 1)
vmax = np.max(sol_matrix1)
plt.imshow(sol_matrix1, cmap="hot")

for i in range(len(unique_participants)):
    for j in range(len(unique_puzzles)):

        if sol_matrix1[i, j] == 0:
            plt.text(j, i,"*", ha="center", va="center", color="w", fontsize=8, fontweight="bold")

        if sol_matrix1[i, j] == -1:
            plt.text(j, i,"N", ha="center", va="center", color="w", fontsize=8, fontweight="bold")

plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
plt.yticks(np.arange(len(unique_participants)), unique_participants)
plt.xlabel("Puzzle ID" , labelpad=20)
plt.ylabel("Participant ID ", labelpad=20) 
plt.title("Run 1" , pad=20)
plt.text(0, 2+len(unique_participants), "N = not solved", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
plt.colorbar( orientation='vertical', pad=0.2, shrink=0.5, label="Time [s]")

plt.subplot(1, 2, 2)
plt.imshow(sol_matrix2, cmap="hot", vmax=vmax)
for i in range(len(unique_participants)):
    for j in range(len(unique_puzzles)):
        if sol_matrix2[i, j] == 0:
            plt.text(j, i,"*", ha="center", va="center", color="w", fontsize=8, fontweight="bold")

        if sol_matrix2[i, j] == -1:
            plt.text(j, i,"N", ha="center", va="center", color="w", fontsize=8, fontweight="bold")
plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
plt.yticks(np.arange(len(unique_participants)), unique_participants)
plt.xlabel("Puzzle ID", labelpad=20)
plt.ylabel("Participant ID", labelpad=20 ) 
plt.title("Run 2", pad=20)
#"N" = not solved
#"*" = missing data
plt.text(0, 2+len(unique_participants), "* = missing data", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
#share colorbar
plt.colorbar( orientation='vertical', pad=0.2, shrink=0.5, label="Time [s]")

plt.savefig("./Data/timeDistribution.png", dpi=300)
                