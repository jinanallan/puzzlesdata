import numpy as np
import os
import json
import matplotlib.pyplot as plt
import movementTracker
import HMPlotter
import wholeSequence
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from dtaidistance import dtw, clustering
from dtaidistance import dtw_ndim
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.patches as mpatches
from setOfAllPossibleInteraction import possibleInteraction

#pairwise distance matrix of each solution

def solutions(puzzleNumber, colorbased=False):

    folder = '/home/erfan/Downloads/pnp'
    # folder=input("Enter the path of the folder containing pnp the json files: ")
    # folder = str(folder)

   
    # pnp puzzle number: 1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26
    sequences=[]
    interaction_lists=[]
    ids=[]

  
    set_of_all_possible_interactions = possibleInteraction(puzzleNumber)
    
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.json'):

            #the participant id, run, puzzle, attempt from the file name:
            participant_id, run, puzzle, attempt = HMPlotter.use_regex(filename)

            if puzzleNumber == puzzle :
                

                with open(os.path.join(folder, filename)) as json_file:

                    data = json.load(json_file)
                    df=movementTracker.df_from_json(data)
                    
                    solved_stats=movementTracker.interaction(df, participant_id, run,type="total",solved=True)
                    
                    if solved_stats== "True":

                        ids.append(str(participant_id) + "_" + str(run) + "_" +str(puzzle) + "_" +str(attempt))

                        #solution sequence of the puzzle:
                        x,y,description=wholeSequence.interaction(df, participant_id, run)
                        # print(description)
                        #full list of interaction containing [type of interaction, time of interaction]:
                        interaction_list=wholeSequence.interaction(df, participant_id, run, listed=True)
                        interaction_lists.append(interaction_list)
                        # print(interaction_list)

                        #coding the verbal description of interaction into a 1*2 vector:
                        transformed_description=wholeSequence.label_encoder(description)
                        # print(transformed_description.shape)
                        # print(transformed_description)

                        if colorbased==True:
                          sequence=wholeSequence.interaction(df, participant_id, run, transformed=True)
                          sequences.append(sequence)
                        #   print(sequence)
                        #   print(sequence.shape)
                          
                        
                        else:
                            sequence=np.vstack((x,y))

                            for i in range(len(transformed_description)):
                                sequence=np.vstack((sequence,transformed_description[i]))
                                # print(transformed_description[i])
                            sequence=sequence.T
                            sequences.append(sequence)
                            # print(sequence[0])
        



    # ds = dtw.distance_matrix_fast(sequences)
    # distance_matrix=ds
    # print(distance_matrix[0])


    
    #following fill the distance martix element by element(instead of using dtw.distance_matrix_fast):


    # distance_matrix=np.zeros((len(sequences),len(sequences)))
    # for i in range(len(sequences)):
    #     for j in range(len(sequences)):
    #         if i!=j and i<j:
    #             querry=sequences[i]
    #             reference=sequences[j]
    #             d=dtw_ndim.distance(querry, reference)
    #             print(i,j,d)
    #             distance_matrix[i][j]=d
    #             distance_matrix[j][i]=d
    # # # print(distance_matrix)
    return sequences,ids,interaction_lists

# def label_encoder(description):
#     label_encoder = LabelEncoder()
#     integer_encoded = label_encoder.fit_transform(description)
#     onehot_encoder = OneHotEncoder(sparse_output=False)
#     integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)  
#     onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
#     transformed_description=onehot_encoded
#     transformed_description=np.array(transformed_description,dtype=np.double)
#     transformed_description=transformed_description.T
#     return transformed_description


def stacked_barplot_interaction(interaction_lists, ids, cluster_id):
    unique_labels = ['free', 'obj1', 'obj2','obj3', 'obj4','box1',"Glue"]
    color_map = {label: i for i, label in enumerate(unique_labels)}
    legend_patches = [mpatches.Patch(color=plt.cm.tab10(color_map[label]), label=label) for label in unique_labels]

    plt.figure(figsize=(25, 10))
    for j in range(len(interaction_lists)):
        sequence = interaction_lists[j]
        # Separate the string elements and float elements into separate lists
        labels = [item[0] for item in sequence]
        lengths = [item[1] for item in sequence]

        # Get unique string elements and assign colors based on the unique labels
        colors = [color_map[label] for label in labels]
        #map colors to rgb values
        colors = [plt.cm.tab10(color) for color in colors]

        plt.bar(j, lengths[0], color=colors[0],edgecolor='white')
        for i in range(1, len(lengths)):
            plt.bar(j, lengths[i], bottom=sum(lengths[:i]), color=colors[i],edgecolor='white' )
  
        plt.xticks(range(len(ids)), ids, rotation=90)
        plt.ylabel('Time (s)')
        plt.xlabel('Participant ID')
        plt.title(f'Interaction sequence for cluster {cluster_id}')


    plt.legend(handles=legend_patches)

def Hcluster(numCluster,puzzleNumber, colorbased):
    sequences,ids,interaction_lists = solutions(puzzleNumber, colorbased)

    distance_matrix=dtw.distance_matrix_fast(sequences)

    # distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
    # distance_matrix = distance_matrix / distance_matrix.max()
    # distance_matrix = 1 - distance_matrix
    # distance_matrix = np.triu(distance_matrix)
    # distance_matrix = distance_matrix[~np.all(distance_matrix == 0, axis=1)]

    Z = linkage(distance_matrix, method='ward', metric='euclidean')
    # # print(Z)

   
    clusters = fcluster(Z, numCluster, criterion='maxclust')
    # # print(clusters)

    cluster_ids = {}

    # Iterate over the data points and assign them to their respective clusters
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_ids:
            cluster_ids[cluster_id] = []
        cluster_ids[cluster_id].append(ids[i])

    # # Print the IDs in each cluster
    # for cluster_id, data_ids in cluster_ids.items():
    #     print(f"Cluster {cluster_id}: {data_ids}")


    # make directory for saving plots
    if not os.path.exists(f'./Plots_Text/clustering/puzzle{puzzleNumber}'):
        os.makedirs(f'./Plots_Text/clustering/puzzle{puzzleNumber}')
        plotPath=f'./Plots_Text/clustering/puzzle{puzzleNumber}'
    else:
        plotPath=f'./Plots_Text/clustering/puzzle{puzzleNumber}'
    
    #do the same with relative path


    plt.figure(figsize=(25, 10))
    dendrogram(Z,labels=ids)
    plt.title('Hierarchical Clustering Dendrogram based on path')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.savefig(f'{plotPath}/Dendrogram_puzzle{puzzleNumber}.png', dpi=300)
    # plt.show()

  
    for cluster_id, data_ids in cluster_ids.items():
        # print(f"Cluster {cluster_id}: {data_ids}")
        interaction_lists_cluster = []
        for data_id in data_ids:
            for i in range(len(ids)):
                if ids[i] == data_id:
                    interaction_lists_cluster.append(interaction_lists[i])
                    break
        stacked_barplot_interaction(interaction_lists_cluster, data_ids,cluster_id)
        plt.savefig(f'{plotPath}/Interaction_stackedbar_cluster{cluster_id}_puzzle{puzzleNumber}.png', dpi=300)
        # plt.show()

#implementing the clustering algorithm based on kmedoids
def kmedoids(numCluster,puzzleNumber, colorbased):
    distance_matrix,ids,interaction_lists = disMat(puzzleNumber, colorbased)
    model=clustering.KMedoids().fit(distance_matrix)
    clusters=model.labels_
    cluster_ids = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_ids:
            cluster_ids[cluster_id] = []
        cluster_ids[cluster_id].append(ids[i])
    # # Print the IDs in each cluster
    for cluster_id, data_ids in cluster_ids.items():
        print(f"Cluster {cluster_id}: {data_ids}")


  
puzzleNumber = int(input("Enter the puzzle number: "))
numCluster = int(input("Enter the number of clusters: "))
colorbased = int(input("Enter 1 for color based clustering and 0 for path based clustering: "))
Hcluster(numCluster,puzzleNumber, colorbased)

# kmedoids(numCluster,puzzleNumber, colorbased)





