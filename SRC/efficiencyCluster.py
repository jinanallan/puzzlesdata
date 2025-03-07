import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the participant distance file
participant_distance_file = './Data/participants_distances_1.csv'
participant_distances = pd.read_csv(participant_distance_file,header=None, index_col=0)
# Directory containing the clustering results
clustering_results_dir = './Plots_Text/clustering/softdtwscore/'

# Loop over each puzzle clustering result
for file_name in os.listdir(clustering_results_dir):
    puzzle_folder_path = os.path.join(clustering_results_dir, file_name)
    if os.path.isdir(puzzle_folder_path):
        cluster_ids_file = None
        for f in os.listdir(puzzle_folder_path):
            if f.startswith('cluster_ids'):
                cluster_ids_file = os.path.join(puzzle_folder_path, f)
                
                #now we have the cluster_ids json file, we can load it and plot the data
                with open(cluster_ids_file) as json_file:

                    cluster_ids = json.load(json_file)
                    #open a figure with subplots of the saze of cluster_ids.keys()
                    fig, axs = plt.subplots(len(cluster_ids.keys()), 1, figsize=(20, 15))
                    fig.suptitle('Participant score for each cluster in ' + file_name, fontsize=20)
                    #get the max number of participants in a cluster
                    max_solutions = max([len(cluster_ids[cluster]) for cluster in cluster_ids.keys()])

                    for cluster in cluster_ids.keys():
                        ids = cluster_ids[cluster]
                        ids = [int(i.split('_')[0]) for i in ids]
                        ids = np.array(ids)
                        ids = ids[~np.isin(ids, [40, 38, 32])]
                        if len(ids)!=0:
                            participant_scores = participant_distances.loc[ids].values.flatten()
                            axs[int(cluster)-1].bar(np.arange(len(ids)), participant_scores)
                            axs[int(cluster)-1].set_ylim(0, 4)
                            axs[int(cluster)-1].set_xlim(-0.5, max_solutions+0.5)
                            axs[int(cluster)-1].set_title('Cluster ' + cluster)
                            axs[int(cluster)-1].set_ylabel('Participant score')
                            #plot the line at the mean
                            axs[int(cluster)-1].axhline(y=np.mean(participant_scores), color='r', linestyle='--')
                            #place a gap between the subplots
                            plt.subplots_adjust(hspace=0.5)


                #save the figure
                plt.savefig(puzzle_folder_path + '/participant_score_across_clustser.png')
                plt.close()
                                