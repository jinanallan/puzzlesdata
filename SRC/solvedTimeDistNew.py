"""
This script processes puzzle-solving data for participants and generates visualizations 
of their performance. It calculates metrics such as the best time to solve puzzles, 
average solving times, and normalized scores. The script also handles missing data 
and generates heatmaps and bar charts to visualize the results.
Modules:
    - pandas: For data manipulation and analysis.
    - numpy: For numerical operations and matrix manipulations.
    - re: For regular expression operations.
    - os: For interacting with the file system.
    - json: For reading JSON files.
    - matplotlib.pyplot: For creating visualizations.
Functions:
    - use_regex_frames(input_text):
        Extracts participant ID, run number, puzzle ID, and attempt number from a 
        filename using regular expressions.
        Args:
            input_text (str): The input filename.
        Returns:
            tuple: A tuple containing participant ID, run number, puzzle ID, and attempt number.
    - df_from_json(file):
        Reads a JSON file and converts it into a pandas DataFrame.
        Args:
            file (str): Path to the JSON file.
        Returns:
            pd.DataFrame: A DataFrame containing the JSON data.
Global Variables:
    - SMALL_SIZE, MEDIUM_SIZE, LAEGER_SIZE: Font sizes for matplotlib visualizations.
    - unique_participants: List of unique participant IDs.
    - unique_puzzles: List of unique puzzle IDs.
    - sol_matrix1, sol_matrix2: Matrices to store solving times for run 1 and run 2.
    - sol_matrix1_best, sol_matrix2_best: Matrices to store the best solving times for run 1 and run 2.
    - sol_att_matrix1, sol_att_matrix2: Matrices to store the number of attempts for run 1 and run 2.
    - ascore, bScore: Normalized scores for attempts and best times.
Workflow:
    1. Load participant and puzzle data from CSV files.
    2. Initialize matrices to store solving times and attempts.
    3. Process JSON files for each pilot run to extract solving times and update matrices.
    4. Handle missing data by assigning NaN values to specific participants and puzzles.
    5. Calculate average solving times and normalize scores.
    6. Generate heatmaps and bar charts to visualize solving times and scores.
    7. Save the visualizations and processed data to files.
Outputs:
    - Heatmaps and bar charts for solving times and scores.
    - CSV files containing average best solving times for participants.
    - PNG files for visualizations.
Note:
    - The script assumes a specific folder structure and file naming convention for input data.
    - Missing data is handled by assigning NaN values, which are visualized with specific markers.
"""
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

sol_matrix1_all = np.empty((len(unique_participants), len(unique_puzzles)))
sol_matrix2_all = np.empty((len(unique_participants), len(unique_puzzles)))
sol_matrix1_all[:] = 0
sol_matrix2_all[:] = 0
ever_solved_matrix1 = np.empty((len(unique_participants), len(unique_puzzles)))
ever_solved_matrix2 = np.empty((len(unique_participants), len(unique_puzzles)))
ever_solved_matrix1[:] = 0
ever_solved_matrix2[:] = 0

#relative improvement between 2 runs per puzzle
sol_matrix_diff = np.empty((len(unique_participants), len(unique_puzzles)))
sol_matrix_diff[:] = 0

solved_matrix1 = np.empty((len(unique_participants), len(unique_puzzles)))
solved_matrix2 = np.empty((len(unique_participants), len(unique_puzzles)))
solved_matrix1[:] = 0
solved_matrix2[:] = 0

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

            # print("Processing file: ", file)
            
            df = df_from_json(folder+file)
            df = df.iloc[0]

            total_time= df["total-time"]
            total_time = float("{:.2f}".format(total_time))
            time_solved = df["time-solved"]
            time_solved = float("{:.2f}".format(time_solved))
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
                    solved_matrix1[particpants_index, puzzle_id_index] = 1
                    sol_matrix1[particpants_index, puzzle_id_index] = total_time 

                    if ever_solved_matrix1[particpants_index, puzzle_id_index] == 0:
                        sol_matrix1_all[particpants_index, puzzle_id_index] += time_solved
                        ever_solved_matrix1[particpants_index, puzzle_id_index] = 1
                        print("Participant {} puzzle {} run {} first solved time {}".format(particpants, puzzle_id, run, time_solved))

                    if total_time < sol_matrix1_best[particpants_index, puzzle_id_index]:
                        sol_matrix1_best[particpants_index, puzzle_id_index] = total_time
                else:
                    solved_matrix2[particpants_index, puzzle_id_index] = 1
                    sol_matrix2[particpants_index, puzzle_id_index] = total_time

                    if ever_solved_matrix2[particpants_index, puzzle_id_index] == 0:
                        sol_matrix2_all[particpants_index, puzzle_id_index] += time_solved
                        ever_solved_matrix2[particpants_index, puzzle_id_index] = 1
                        print("Participant {} puzzle {} run {} first solved time {}".format(particpants, puzzle_id, run, time_solved))

                    if total_time < sol_matrix2_best[particpants_index, puzzle_id_index]:
                        sol_matrix2_best[particpants_index, puzzle_id_index] = total_time
            else:
                if run == 1:
                    sol_matrix1[particpants_index, puzzle_id_index] = -1
                    sol_matrix1_all[particpants_index, puzzle_id_index] += total_time

                else:
                    sol_matrix2[particpants_index, puzzle_id_index] = -1
                    sol_matrix2_all[particpants_index, puzzle_id_index] += total_time


#manually adding missing data
for particpants in [32]:
    particpants_index = np.where(unique_participants == particpants)[0][0]
    for puzzle_id in np.arange(0, 27):
        puzzle_id_index = np.where(unique_puzzles == puzzle_id)[0][0]
        sol_matrix2_best[particpants_index, puzzle_id_index] = np.nan
        sol_matrix2_all[particpants_index, puzzle_id_index] = 0 #missing data for the 2nd run      

for particpants in [38]:
    particpants_index = np.where(unique_participants == particpants)[0][0]
    for puzzle_id in np.arange(0, 14):
        puzzle_id_index = np.where(unique_puzzles == puzzle_id)[0][0]
        sol_matrix2_best[particpants_index, puzzle_id_index] = np.nan
        sol_matrix2_all[particpants_index, puzzle_id_index] = 0 #missing data for the 2nd run

for particpants in [40]:
    particpants_index = np.where(unique_participants == particpants)[0][0]
    for puzzle_id in np.arange(0, 10):
        puzzle_id_index = np.where(unique_puzzles == puzzle_id)[0][0]
        sol_matrix2_best[particpants_index, puzzle_id_index] = np.nan
        sol_matrix2_all[particpants_index, puzzle_id_index] = 0 #missing data for the 2nd run

#save sol_matrix1_all and sol_matrix2_all
np.savetxt("./Data/sol_matrix1_all.csv", sol_matrix1_all, delimiter=",")
np.savetxt("./Data/sol_matrix2_all.csv", sol_matrix2_all, delimiter=",")


# columnsun1 = np.mean(np.ma.masked_invalid(sol_matrix1), axis=0)
# columnsun2 = np.mean(np.ma.masked_invalid(sol_matrix2), axis=0)

# #best time score is normalized (between 0 and 1) value of  columnsun1+columnsun2
# bScore = columnsun1+columnsun2
# bScore = bScore/2
# bScore = (bScore - np.min(bScore))/(np.max(bScore) - np.min(bScore))


# # plt.figure(figsize=(20,15))
# # plt.suptitle('Time Solved [s]', fontsize=20)
# # plt.subplot(1, 2, 1)
# # vmax = np.max(sol_matrix1)
# # plt.imshow(sol_matrix1, cmap="hot")

# # for i in range(len(unique_participants)):
# #     for j in range(len(unique_puzzles)):

# #         if sol_matrix1[i, j] == 0:
# #             plt.text(j, i,"*", ha="center", va="center", color="w", fontsize=8, fontweight="bold")

# #         if sol_matrix1[i, j] == -1:
# #             plt.text(j, i,"N", ha="center", va="center", color="w", fontsize=8, fontweight="bold")

# # plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
# # plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
# # plt.yticks(np.arange(len(unique_participants)), unique_participants)
# # plt.xlabel("Puzzle ID" , labelpad=20)
# # plt.ylabel("Participant ID ", labelpad=20) 
# # plt.title("Run 1" , pad=20)
# # plt.text(0, 2+len(unique_participants), "N = not solved", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
# # plt.colorbar( orientation='vertical', pad=0.2, shrink=0.5, label="Time [s]")

# # plt.subplot(1, 2, 2)
# # plt.imshow(sol_matrix2, cmap="hot", vmax=vmax)
# # for i in range(len(unique_participants)):
# #     for j in range(len(unique_puzzles)):
# #         if sol_matrix2[i, j] == 0:
# #             plt.text(j, i,"*", ha="center", va="center", color="w", fontsize=8, fontweight="bold")

# #         if sol_matrix2[i, j] == -1:
# #             plt.text(j, i,"N", ha="center", va="center", color="w", fontsize=8, fontweight="bold")
# # plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
# # plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
# # plt.yticks(np.arange(len(unique_participants)), unique_participants)
# # plt.xlabel("Puzzle ID", labelpad=20)
# # plt.ylabel("Participant ID", labelpad=20 ) 
# # plt.title("Run 2", pad=20)
# # #"N" = not solved
# # #"*" = missing data
# # plt.text(0, 2+len(unique_participants), "* = missing data", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
# # #share colorbar
# # plt.colorbar( orientation='vertical', pad=0.2, shrink=0.5, label="Time [s]")

# # plt.savefig("./Data/timeDistribution.png", dpi=300)

# #best time
# plt.figure(figsize=(20,11))
# plt.suptitle('Best Time Solved [s]', fontsize=20)
# plt.subplot(1, 2, 1)
# vmax_orig = np.max(sol_matrix1_best[sol_matrix1_best != np.inf])
# vmax = np.unique(sol_matrix1_best[sol_matrix1_best != np.inf])[-3]
# plt.imshow(sol_matrix1_best, cmap="hot", vmax=vmax, vmin=0)

# for i in range(len(unique_participants)):
#     for j in range(len(unique_puzzles)):

#         # if sol_matrix1_best[i, j] == 0:
#         #     plt.text(j, i,"*", ha="center", va="center", color="w", fontsize=8, fontweight="bold")

#         if sol_matrix1_best[i, j] == np.inf:
#             plt.text(j, i,"N", ha="center", va="center", color="black", fontsize=8, fontweight="bold")
#         elif np.isnan(sol_matrix1_best[i, j]):
#             plt.text(j, i,"*", ha="center", va="center", color="black", fontsize=8, fontweight="bold")

# # Replace inf values in each row by the max of each column
# for j in range(sol_matrix1_best.shape[1]):  #per puzzle
#     col_max = np.max(sol_matrix1_best[sol_matrix1_best[:, j] != np.inf, j])
#     col_min = np.inf #np.min(sol_matrix1_all[sol_matrix1_all[:, j] != 0 and solved_matrix1[:, j] != 0, j])

#     for k in range(sol_matrix1_all.shape[0]):
#         if sol_matrix1_all[k, j] < col_min and solved_matrix1[k, j] != 0:
#             col_min = sol_matrix1_all[k, j] 

# #using the min over both runs!
#     for k in range(sol_matrix2_all.shape[0]): #over participants!
#         if sol_matrix2_all[k, j] < col_min and solved_matrix2[k, j] != 0:
#             col_min = sol_matrix2_all[k, j]             

#     print("Puzzle, min total time ", j, col_min)
#     out = ""
#     for k in range(sol_matrix1_all.shape[0]):
#         if solved_matrix1[k, j] == 1:
#             sol_matrix1_all[k, j] = col_min / sol_matrix1_all[k, j]
#         else:
#             sol_matrix1_all[k, j] = 0
#         out += str(sol_matrix1_all[k, j]) + " "

#     for k in range(sol_matrix2_all.shape[0]):
#         if solved_matrix2[k, j] == 1:
#             sol_matrix2_all[k, j] = col_min / sol_matrix2_all[k, j]
#         else:
#             sol_matrix2_all[k, j] = 0

# #run1 solved t1, run2 solved t2 : t1-t2/t1, unsolved, solved t2 -> 1.0, unsolved, unsolved: 0, solved, unsolved : 0
#     for k in range(sol_matrix1_all.shape[0]): #over participants, save their improvement per puzzle
#         if solved_matrix2[k, j] == 1 and solved_matrix1[k, j] == 1:  #the efficiency score in the 2nd run is expected to be higher!
#             sol_matrix_diff[k, j] = sol_matrix2_all[k, j] - sol_matrix1_all[k, j] ## / sol_matrix2_all[k, j]  #delta in the efficiency score
#             if sol_matrix_diff[k, j] < 0:
#                 print("Decrease in participant and puzzle: ", k, j)
#             # time1 / best_time - time2 / best_time , if solved twice
#             # 
#         else:
#             sol_matrix_diff[k, j] = 0 ##sol_matrix2_all[k, j] - sol_matrix1_all[k, j]  ###-sol_matrix1_all[k, j]  #same as above :)


#     #    out += str(sol_matrix2_all[k, j]) + " "

#     print(out)

#    # sol_matrix1_all[sol_matrix1_all[:, j] != 0, j] = col_min / sol_matrix1_all[:, j]
#     sol_matrix1_best[sol_matrix1_best[:, j] == np.inf, j] = col_max




# # Replace inf values in each row by the max of each column
# #for j in range(sol_matrix2_best.shape[1]):
#    ## col_max = np.max(sol_matrix2_best[sol_matrix1_best[:, j] != np.inf, j])
#     # col_min = np.inf #np.min(sol_matrix1_all[sol_matrix1_all[:, j] != 0 and solved_matrix1[:, j] != 0, j])    

#     # for k in range(sol_matrix2_all.shape[0]):
#     #     if sol_matrix2_all[k, j] < col_min and solved_matrix2[k, j] != 0:
#     #         col_min = sol_matrix2_all[k, j] 

#     # print("Puzzle, min total time ", j, col_min)
#     # out = ""
#     # for k in range(sol_matrix2_all.shape[0]):
#     #     if solved_matrix2[k, j] == 1:
#     #         sol_matrix2_all[k, j] = col_min / sol_matrix2_all[k, j]
#     #     else:
#     #         sol_matrix2_all[k, j] = 0
#     #     out += str(sol_matrix2_all[k, j]) + " "
#     # print(out)

#    # sol_matrix1_all[sol_matrix1_all[:, j] != 0, j] = col_min / sol_matrix1_all[:, j]
#     #sol_matrix2_best[sol_matrix2_best[:, j] == np.inf, j] = col_max


# #remove the 2,8,10 th element from the array
# sol_matrix_diff = np.delete(sol_matrix_diff, (1, 7, 9), axis=0)

# #sol_matrix_diff = np.delete(sol_matrix_diff, [1,7,9])  #as we want averages per puzzle over all participants we remove participants with missing data
# #remove the 2,8,10 th element from unique_participants as well
# unique_participants_clean = np.delete(unique_participants, [1,7,9])

# rawsum1 = np.nanmean(sol_matrix1_best, axis=1)/60
# rawsum_all = np.nanmean(sol_matrix1_all, axis=1) #average over all puzzle efficiency scores per participant
# scoreperpuzzle_all = np.nanmean(sol_matrix1_all, axis=0) #average over all participant scores per puzzle: which puzzle has the lowest average score
# #per puzzle is flawed because it just compares to the best found solution, potentially there could be puzzles which were never solved..
# diff_mean = np.nanmean(sol_matrix_diff, axis=0) #average over all changes in puzzle efficiency scores per _puzzle_
# print("Avg scores", scoreperpuzzle_all)

# #plot ar bar chart
# x_pos = np.arange(len(unique_puzzles))   # 0, 1, 2, …
# width = 0.35                  # bar width

# fig, ax = plt.subplots()
# #ax.bar(x_pos - width/2, best_time_1, width, label="Run1")
# ax.bar(x_pos, diff_mean, width, label="Score change")

# ax.set_xticks(x_pos)
# ax.set_xticklabels(unique_puzzles, fontsize=8)
# ax.set_xlabel("Puzzle ID")
# ax.set_ylabel("Score change")
# ax.set_title("Average efficiency score changes between two runs. Per puzzle over all participants")
# ax.legend()
# plt.tight_layout()
# #plt.show()
# plt.savefig('./Data/diff_scores_per_puzzle.png', dpi=300, bbox_inches='tight')

# # plot differences as box plot
# fig, ax = plt.subplots()
# ax.boxplot(sol_matrix_diff)
# ax.set_title('Score changes')

# ax.set_xticks(x_pos)
# ax.set_xticklabels(unique_puzzles, fontsize=8)
# ax.set_xlabel("Puzzle ID")
# ax.set_ylabel("Average score")
# ax.legend()
# plt.tight_layout()
# #plt.show()
# plt.savefig('./Data/diff_scores_per_puzzle_boxplot.png', dpi=300, bbox_inches='tight')




# ###plt.barh(y=np.arange(len(unique_participants)), width=-rawsum1, left=-0.5, color="lightslategray")
# plt.barh(y=np.arange(len(unique_participants)), width=-rawsum_all, left=-0.5, color="lightslategray")

# np.savetxt("./Data/participants_avg_best_time_1.csv", rawsum1, delimiter=",")
# np.savetxt("./Data/participants_avg_total_time_1.csv", rawsum_all, delimiter=",")   #the scores summed up
# np.savetxt("./Data/participants_per_puzzle_score_1.csv", scoreperpuzzle_all, delimiter=",") 

# # plot puzzle scores (averaged over participants) as box plot
# fig, ax = plt.subplots()
# ax.bar(x_pos, scoreperpuzzle_all, width, label="Avg scores")
# ##ax.boxplot(scoreperpuzzle_all)
# ax.set_title('Avg scores per puzzle')

# ax.set_xticks(x_pos)
# ax.set_xticklabels(unique_puzzles, fontsize=8)
# ax.set_xlabel("Puzzle ID")
# ax.set_ylabel("Avg score")
# ax.legend()
# plt.tight_layout()
# #plt.show()
# plt.savefig('./Data/avg_scores_per_puzzle_boxplot.png', dpi=300, bbox_inches='tight')
# #difference between the avg change and avg score: some puzzles might have consitently low score, others actually high


# rawsum_diff = np.nanmean(sol_matrix_diff, axis=1) #sum of improvements over all puzzles per participant
# np.savetxt("./Data/participants_avg_diff_time.csv", rawsum_diff, delimiter=",")
# ##print('Improvements per puzzle per participant', sol_matrix_diff)
# nr = 0
# for n in sol_matrix_diff:
#     print(nr, n)
#     nr += 1

# #improvement of a participant



# columnsum1 = np.nanmean(sol_matrix1_best, axis=0)/60
# # plt.bar(x=np.arange(len(unique_puzzles)), height=-columnsum1, bottom=-0.5, color="lightslategray")

# # plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
# # plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
# # plt.yticks(np.arange(len(unique_participants)), unique_participants)
# # plt.xlabel("Puzzle ID" , labelpad=20)
# # plt.ylabel("Participant ID ", labelpad=20) 
# # plt.title("Run 1" , pad=20)
# # plt.text(0, 1+len(unique_participants), "N = not solved, * = missing data", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
# # plt.text(0, 2+len(unique_participants), "max time run 1: {} [s]".format(vmax_orig), ha="center", va="center", color="black", fontsize=10, fontweight="bold")
# # plt.colorbar( orientation='vertical', pad=0.1, shrink=0.5, label="Time [s]")

# # plt.subplot(1, 2, 2)
# # # vmax = np.max(sol_matrix2_best[sol_matrix2_best != np.inf])
# # plt.imshow(sol_matrix2_best, cmap="hot", vmax=vmax, vmin=0)

# # for i in range(len(unique_participants)):
# #     for j in range(len(unique_puzzles)):

# #         # if sol_matrix2_best[i, j] == 0:
# #         #     plt.text(j, i,"*", ha="center", va="center", color="w", fontsize=8, fontweight="bold")

# #         if sol_matrix2_best[i, j] == np.inf:
# #             plt.text(j, i,"N", ha="center", va="center", color="black", fontsize=8, fontweight="bold")
# #         elif np.isnan(sol_matrix2_best[i, j]):
# #             plt.text(j, i,"*", ha="center", va="center", color="black", fontsize=8, fontweight="bold")

# # # Replace inf values in each column by the max of each column
# # for j in range(sol_matrix2_best.shape[1]):
# #     col_max = np.max(sol_matrix2_best[sol_matrix2_best[:, j] != np.inf, j])
# #     sol_matrix2_best[sol_matrix2_best[:, j] == np.inf, j] = col_max

# # rawsum2 = np.nanmean(sol_matrix2_best, axis=1)/60
# rawsum2_all = np.nanmean(sol_matrix2_all, axis=1) #average over all puzzle efficiency scores
# # plt.barh(y=np.arange(len(unique_participants)), width=-rawsum2_all, left=-0.5, color="lightslategray")

# ##np.savetxt("./Data/participants_avg_best_time_2.csv", rawsum1, delimiter=",") #probably supposed to be rawsum2?
# ###np.savetxt("./Data/participants_avg_best_time_2.csv", rawsum2, delimiter=",")

# np.savetxt("./Data/participants_avg_total_time_2.csv", rawsum2_all, delimiter=",")

# rawsum_only_diff = (rawsum2_all - rawsum_all) / rawsum2_all 
# np.savetxt("./Data/participants_avg_diff_rel.csv", rawsum_only_diff, delimiter=",")
# print(rawsum_all, rawsum2_all, rawsum_only_diff)





# columnsum2 = np.nanmean(sol_matrix2_best, axis=0)/60
# # plt.bar(x=np.arange(len(unique_puzzles)), height=-columnsum2, bottom=-0.5, color="lightslategray")

# # plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
# # plt.tick_params(axis='x', which='both', bottom=True, top=True, labelbottom=True, labeltop=True)
# # plt.yticks(np.arange(len(unique_participants)), unique_participants)
# # plt.xlabel("Puzzle ID", labelpad=20)
# # plt.ylabel("Participant ID", labelpad=20 ) 
# # plt.title("Run 2", pad=20)
# # #"N" = not solved
# # #replace nan with np.inf
# # sol_matrix2_best[np.isnan(sol_matrix2_best)] = np.inf
# # vmax2 = np.max(sol_matrix2_best[sol_matrix2_best != np.inf])
# # plt.text(0, 1+len(unique_participants), "N = not solved, * = missing data", ha="center", va="center", color="black", fontsize=10, fontweight="bold")
# # plt.text(0, 2+len(unique_participants), "max time run 2: {} [s]".format(vmax2), ha="center", va="center", color="black", fontsize=10, fontweight="bold")

# # plt.colorbar( orientation='vertical', pad=0.1, shrink=0.5, label="Time [s]")

# # plt.savefig("./Data/bestTimeDistribution.png", dpi=300)
# #scale ascore and bScore between 1 and 10
# # ascore = ascore*9+1
# # bScore = bScore*9+1
# # difScore = bScore*ascore
# # difScore = difScore/np.max(difScore)
# # difScore = difScore*9+1

# # plt.figure(figsize=(20,11))
# # plt.suptitle(' attempt score - time score - diff score', fontsize=20)
# # plt.subplot(1, 3, 1)
# # plt.bar(unique_puzzles, ascore, color="black")
# # plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
# # plt.yticks(np.arange(0, 11, 1))
# # plt.xlabel("Puzzle ID" , labelpad=20)
# # plt.ylabel("Score", labelpad=20)
# # plt.yticks(np.arange(0, 11, 1))
# # plt.subplot(1, 3, 2)
# # plt.bar(unique_puzzles, bScore, color="black")
# # plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
# # plt.xlabel("Puzzle ID" , labelpad=20)
# # plt.ylabel("Score", labelpad=20)
# # plt.yticks(np.arange(0, 11, 1))
# # plt.subplot(1, 3, 3)
# # plt.bar(unique_puzzles, difScore, color="black")
# # plt.xticks(np.arange(len(unique_puzzles)), unique_puzzles, rotation=90)
# # plt.xlabel("Puzzle ID" , labelpad=20)
# # plt.ylabel("Score", labelpad=20)
# # plt.yticks(np.arange(0, 11, 1))
# # plt.tight_layout()
# # plt.savefig("./Data/difScore.png", dpi=300)
# # plt.close()