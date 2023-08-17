import re
import os
import json
import pandas as pd
import numpy as np
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
    pattern = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_frames", re.IGNORECASE)

    match = pattern.match(input_text)
    
    particpants = match.group(3)
    run = match.group(4)
    puzzle_id = match.group(5)
    attempt = match.group(6)
    return int(particpants), int(run), int(puzzle_id), int(attempt)

def use_regex(input_text):
    pattern = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)", re.IGNORECASE)

    match = pattern.match(input_text)
    
    particpants = match.group(3)
    run = match.group(4)
    puzzle_id = match.group(5)
    attempt = match.group(6)
    return int(particpants), int(run), int(puzzle_id), int(attempt)

df= pd.DataFrame(columns=["participant_id", "run", "puzzle_id", "attempt", "file", "frame file"])

for pilot in [3,4]:
    frame_folder= "./Data/pilot{}/Frames/".format(pilot)
    pnp_folder= "./Data/pilot{}/Ego-based/".format(pilot)

    frame_files = os.listdir(frame_folder)
    pnp_files = os.listdir(pnp_folder)


    for file in pnp_files:
        if file.endswith(".json"):

            particpants, run, puzzle_id, attempt = use_regex(file)
        
            new_row = {"participant_id": particpants, "run": run, "puzzle_id": puzzle_id, "attempt": attempt, "file": file, "frame file": None}

            df.loc[len(df)] = new_row

    for file in frame_files:
        if file.endswith(".json"):

            particpants, run, puzzle_id, attempt = use_regex_frames(file)

            df.loc[(df["participant_id"] == particpants) & (df["run"] == run) &
                    (df["puzzle_id"] == puzzle_id) & (df["attempt"] == attempt), "frame file"] =file

df = df.sort_values(by=["participant_id", "run", "puzzle_id", "attempt"])

df.to_csv("./Data/df.csv", index=False)
# df= pd.read_csv("./Data/df.csv")

unique_participants = df["participant_id"].unique().tolist()
unique_puzzles = df["puzzle_id"].unique().tolist()
sol_matrix1 = np.zeros((len(unique_participants), len(unique_puzzles)))
sol_matrix2 = np.zeros((len(unique_participants), len(unique_puzzles)))

for index, row in df.iterrows():
    # print(row["participant_id"], row["puzzle_id"])

    i=unique_participants.index(row["participant_id"])
    j=unique_puzzles.index(row["puzzle_id"])
    # print(i, j)
    if row["run"] == 1:
        sol_matrix1[i][j] += 1
    else:
        sol_matrix2[i][j] += 1

sol_matrix1 = sol_matrix1.astype(int)
columnsun1 = np.sum(sol_matrix1, axis=0).astype(float)*-1
columnsun1 /= len(unique_participants)
sol_matrix2 = sol_matrix2.astype(int) 
columnsun2 = np.sum(sol_matrix2, axis=0).astype(float)*-1
columnsun2 /= len(unique_participants)

plt.figure(figsize=(20,10))
plt.suptitle('Number of attempts at each run', fontsize=20)
plt.subplot(1, 2, 1)
plt.imshow(sol_matrix1, cmap="turbo", vmin=0)
plt.bar(height=columnsun1, x=np.arange(len(unique_puzzles)), bottom=-0.5, color="lightslategray")
# plt.axhline(y=min(columnsun1)-0.5, color='k', linestyle=':', linewidth=1, )

for i in range(len(unique_participants)):
    for j in range(len(unique_puzzles)):
        plt.text(j, i, sol_matrix1[i, j], ha="center", va="center", color="w", fontsize=8, fontweight="bold")

plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
plt.yticks(np.arange(len(unique_participants)), unique_participants)
plt.xlabel("Puzzle ID" , labelpad=20)
plt.ylabel("Participant ID - avg. number of attempts", labelpad=20) 
plt.title("Run 1" , pad=20)
# plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(sol_matrix2, cmap="turbo", vmin=0)
plt.bar(height=columnsun2, x=np.arange(len(unique_puzzles)), bottom=-0.5, color="lightslategray" )

for i in range(len(unique_participants)):
    for j in range(len(unique_puzzles)):
        plt.text(j, i, sol_matrix2[i, j], ha="center", va="center", color="w", fontsize=8, fontweight="bold")

plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
plt.yticks(np.arange(len(unique_participants)), unique_participants)
plt.xlabel("Puzzle ID", labelpad=20)
plt.ylabel("Participant ID - avg. number of attempts", labelpad=20 ) 
# plt.colorbar()
plt.title("Run 2", pad=20)
plt.savefig("./Data/Distribution.png", dpi=300)
