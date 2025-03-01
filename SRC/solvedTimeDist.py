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

sol_matrix1_best = np.empty((len(unique_participants), len(unique_puzzles)))
sol_matrix2_best = np.empty((len(unique_participants), len(unique_puzzles)))
sol_matrix1_best[:] = np.inf
sol_matrix2_best[:] = np.inf

sol_att_matrix1 = np.genfromtxt("./Data/sol_matrix1.csv", delimiter=',')
sol_att_matrix2 = np.genfromtxt("./Data/sol_matrix2.csv", delimiter=',')

attCol1 = np.mean(sol_att_matrix1, axis=0)
attCol2 = np.mean(sol_att_matrix2, axis=0)
ascore = attCol1+attCol2
ascore = ascore/2
ascore = (ascore - np.min(ascore))/(np.max(ascore) - np.min(ascore))

for pilot in [3,4]:

    folder = "./Data/Pilot{}/Ego-based/".format(pilot)
    for file in os.listdir(folder):

        if file.endswith(".json"):
            
            particpants, run, puzzle_id, attempt = use_regex_frames(file)

            df = df_from_json(folder+file)
            df = df.iloc[0]

            total_time= df["total-time"]
            total_time = float("{:.2f}".format(total_time))
            solved= df['solved'] 
          
            particpants_index = np.where(unique_participants == particpants)[0][0]
            puzzle_id_index = np.where(unique_puzzles == puzzle_id)[0][0]

            
            if run == 1:
                n_attempts = sol_att_matrix1[particpants_index, puzzle_id_index]
                n_attempts = n_attempts.astype(int)
            else:
                n_attempts = sol_att_matrix2[particpants_index, puzzle_id_index]
                n_attempts = n_attempts.astype(int)

            

            # if solved and n_attempts>attempt+1:
            #     print("participant {} puzzle {} run {} **attempt {} total time {} solved {} while n_attempts {}".
            #           format(particpants, puzzle_id, run, attempt, total_time, solved, n_attempts))
            #     #conclusion: it is possible to solve a puzzle and do more attempts

            #finding the fastest time

            if solved:
                if run == 1:
                    sol_matrix1[particpants_index, puzzle_id_index] = total_time
                    if total_time < sol_matrix1_best[particpants_index, puzzle_id_index]:
                        sol_matrix1_best[particpants_index, puzzle_id_index] = total_time
                else:
                    sol_matrix2[particpants_index, puzzle_id_index] = total_time
                    if total_time < sol_matrix2_best[particpants_index, puzzle_id_index]:
                        sol_matrix2_best[particpants_index, puzzle_id_index] = total_time
            else:
                if run == 1:
                    sol_matrix1[particpants_index, puzzle_id_index] = -1
                else:
                    sol_matrix2[particpants_index, puzzle_id_index] = -1

#manually adding missing data
for particpants in [32]:
    particpants_index = np.where(unique_participants == particpants)[0][0]
    for puzzle_id in np.arange(0, 27):
        puzzle_id_index = np.where(unique_puzzles == puzzle_id)[0][0]
        sol_matrix2_best[particpants_index, puzzle_id_index] = np.nan

for particpants in [38]:
    particpants_index = np.where(unique_participants == particpants)[0][0]
    for puzzle_id in np.arange(0, 14):
        puzzle_id_index = np.where(unique_puzzles == puzzle_id)[0][0]
        sol_matrix2_best[particpants_index, puzzle_id_index] = np.nan

for particpants in [40]:
    particpants_index = np.where(unique_participants == particpants)[0][0]
    for puzzle_id in np.arange(0, 10):
        puzzle_id_index = np.where(unique_puzzles == puzzle_id)[0][0]
        sol_matrix2_best[particpants_index, puzzle_id_index] = np.nan

columnsun1 = np.mean(np.ma.masked_invalid(sol_matrix1), axis=0)
columnsun2 = np.mean(np.ma.masked_invalid(sol_matrix2), axis=0)

#best time score is normalized (between 0 and 1) value of  columnsun1+columnsun2
bScore = columnsun1+columnsun2
bScore = bScore/2
bScore = (bScore - np.min(bScore))/(np.max(bScore) - np.min(bScore))


# plt.figure(figsize=(20,15))
# plt.suptitle('Time Solved [s]', fontsize=20)
# plt.subplot(1, 2, 1)
# vmax = np.max(sol_matrix1)
# plt.imshow(sol_matrix1, cmap="hot")

# for i in range(len(unique_participants)):
#     for j in range(len(unique_puzzles)):

#         if sol_matrix1[i, j] == 0:
#             plt.text(j, i,"*", ha="center", va="center", color="w", fontsize=8, fontweight="bold")

#         if sol_matrix1[i, j] == -1:
#             plt.text(j, i,"N", ha="center", va="center", color="w", fontsize=8, fontweight="bold")

# plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
# plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
# plt.yticks(np.arange(len(unique_participants)), unique_participants)
# plt.xlabel("Puzzle ID" , labelpad=20)
# plt.ylabel("Participant ID ", labelpad=20) 
# plt.title("Run 1" , pad=20)
# plt.text(0, 2+len(unique_participants), "N = not solved", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
# plt.colorbar( orientation='vertical', pad=0.2, shrink=0.5, label="Time [s]")

# plt.subplot(1, 2, 2)
# plt.imshow(sol_matrix2, cmap="hot", vmax=vmax)
# for i in range(len(unique_participants)):
#     for j in range(len(unique_puzzles)):
#         if sol_matrix2[i, j] == 0:
#             plt.text(j, i,"*", ha="center", va="center", color="w", fontsize=8, fontweight="bold")

#         if sol_matrix2[i, j] == -1:
#             plt.text(j, i,"N", ha="center", va="center", color="w", fontsize=8, fontweight="bold")
# plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
# plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
# plt.yticks(np.arange(len(unique_participants)), unique_participants)
# plt.xlabel("Puzzle ID", labelpad=20)
# plt.ylabel("Participant ID", labelpad=20 ) 
# plt.title("Run 2", pad=20)
# #"N" = not solved
# #"*" = missing data
# plt.text(0, 2+len(unique_participants), "* = missing data", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
# #share colorbar
# plt.colorbar( orientation='vertical', pad=0.2, shrink=0.5, label="Time [s]")

# plt.savefig("./Data/timeDistribution.png", dpi=300)

#best time
plt.figure(figsize=(20,11))
plt.suptitle('Best Time Solved [s]', fontsize=20)
plt.subplot(1, 2, 1)
vmax = np.max(sol_matrix1_best[sol_matrix1_best != np.inf])
plt.imshow(sol_matrix1_best, cmap="hot", vmax=vmax, vmin=0)

rawsum1 = np.nanmean(np.where(np.isinf(sol_matrix1_best), -1, sol_matrix1_best), axis=1)/60
plt.barh(y=np.arange(len(unique_participants)), width=-rawsum1, left=-0.5, color="lightslategray")

columnsum1 = np.nanmean(np.where(np.isinf(sol_matrix1_best), -1, sol_matrix1_best), axis=0)/60
plt.bar(x=np.arange(len(unique_puzzles)), height=-columnsum1, bottom=-0.5, color="lightslategray")

for i in range(len(unique_participants)):
    for j in range(len(unique_puzzles)):

        # if sol_matrix1_best[i, j] == 0:
        #     plt.text(j, i,"*", ha="center", va="center", color="w", fontsize=8, fontweight="bold")

        if sol_matrix1_best[i, j] == np.inf:
            plt.text(j, i,"N", ha="center", va="center", color="black", fontsize=8, fontweight="bold")
        elif np.isnan(sol_matrix1_best[i, j]):
            plt.text(j, i,"*", ha="center", va="center", color="black", fontsize=8, fontweight="bold")

plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
plt.yticks(np.arange(len(unique_participants)), unique_participants)
plt.xlabel("Puzzle ID" , labelpad=20)
plt.ylabel("Participant ID ", labelpad=20) 
plt.title("Run 1" , pad=20)
plt.text(0, 1+len(unique_participants), "N = not solved, * = missing data", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
plt.text(0, 2+len(unique_participants), "max time run 1: {} [s]".format(vmax), ha="center", va="center", color="black", fontsize=10, fontweight="bold")
plt.colorbar( orientation='vertical', pad=0.1, shrink=0.5, label="Time [s]")

plt.subplot(1, 2, 2)
plt.imshow(sol_matrix2_best, cmap="hot", vmax=vmax, vmin=0)

rawsum2 = np.nanmean(np.where(np.isinf(sol_matrix2_best), -1, sol_matrix2_best), axis=1)/60
plt.barh(y=np.arange(len(unique_participants)), width=-rawsum2, left=-0.5, color="lightslategray")

columnsum2 = np.nanmean(np.where(np.isinf(sol_matrix2_best), -1, sol_matrix2_best), axis=0)/60
plt.bar(x=np.arange(len(unique_puzzles)), height=-columnsum2, bottom=-0.5, color="lightslategray")

for i in range(len(unique_participants)):
    for j in range(len(unique_puzzles)):

        # if sol_matrix2_best[i, j] == 0:
        #     plt.text(j, i,"*", ha="center", va="center", color="w", fontsize=8, fontweight="bold")

        if sol_matrix2_best[i, j] == np.inf:
            plt.text(j, i,"N", ha="center", va="center", color="black", fontsize=8, fontweight="bold")
        elif np.isnan(sol_matrix2_best[i, j]):
            plt.text(j, i,"*", ha="center", va="center", color="black", fontsize=8, fontweight="bold")

plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
plt.yticks(np.arange(len(unique_participants)), unique_participants)
plt.xlabel("Puzzle ID", labelpad=20)
plt.ylabel("Participant ID", labelpad=20 ) 
plt.title("Run 2", pad=20)
#"N" = not solved
#replace nan with np.inf
sol_matrix2_best[np.isnan(sol_matrix2_best)] = np.inf
vmax2 = np.max(sol_matrix2_best[sol_matrix2_best != np.inf])
plt.text(0, 1+len(unique_participants), "N = not solved, * = missing data", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
plt.text(0, 2+len(unique_participants), "max time run 2: {} [s]".format(vmax2), ha="center", va="center", color="black", fontsize=10, fontweight="bold")

plt.colorbar( orientation='vertical', pad=0.1, shrink=0.5, label="Time [s]")

plt.savefig("./Data/bestTimeDistribution.png", dpi=300)
#scale ascore and bScore between 1 and 10
# ascore = ascore*9+1
# bScore = bScore*9+1
# difScore = bScore*ascore
# difScore = difScore/np.max(difScore)
# difScore = difScore*9+1

# plt.figure(figsize=(20,11))
# plt.suptitle(' attempt score - time score - diff score', fontsize=20)
# plt.subplot(1, 3, 1)
# plt.bar(unique_puzzles, ascore, color="black")
# plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
# plt.yticks(np.arange(0, 11, 1))
# plt.xlabel("Puzzle ID" , labelpad=20)
# plt.ylabel("Score", labelpad=20)
# plt.yticks(np.arange(0, 11, 1))
# plt.subplot(1, 3, 2)
# plt.bar(unique_puzzles, bScore, color="black")
# plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
# plt.xlabel("Puzzle ID" , labelpad=20)
# plt.ylabel("Score", labelpad=20)
# plt.yticks(np.arange(0, 11, 1))
# plt.subplot(1, 3, 3)
# plt.bar(unique_puzzles, difScore, color="black")
# plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
# plt.xlabel("Puzzle ID" , labelpad=20)
# plt.ylabel("Score", labelpad=20)
# plt.yticks(np.arange(0, 11, 1))
# plt.tight_layout()
# plt.savefig("./Data/difScore.png", dpi=300)
# plt.close()