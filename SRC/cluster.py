import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import movementTracker
import HMPlotter
from gif_generator import gif
from sklearn.preprocessing import OneHotEncoder
from dtaidistance import dtw
from dtaidistance import dtw_ndim
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.patches as mpatches
import time


start_time = time.time()

folder = '/home/erfan/Downloads/pnp'

def possibleInteraction(puzzleNumber):
        #the objective of this function is to find the set of all possible interactions in a puzzle

        # pnp puzzle number: 1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26
        set_of_all_possible_interactions = set()
        for filename in sorted(os.listdir(folder)):
            if filename.endswith('.json'):

                #the participant id, run, puzzle, attempt from the file name:
                participant_id, run, puzzle, attempt = HMPlotter.use_regex(filename)

                if puzzleNumber == puzzle :
                    

                    with open(os.path.join(folder, filename)) as json_file:

                        data = json.load(json_file)
                        df=movementTracker.df_from_json(data)

                        try: 
                            events = df["events"] 
                            
                            df_events = pd.DataFrame(events)
                            df_events["description"] = df['events'].apply(lambda x: x.get('description'))

                            attachIndex = (df_events.index[df_events['description'].str.contains("Attach")]).tolist()
                            releaseIndex = (df_events.index[df_events['description'].str.contains("Release")]).tolist()

                            # assign the name of the object (obj,box) to the events between "attach" and "release" events    
                            for i in range(len(attachIndex)):
                                df_events.loc[attachIndex[i]:releaseIndex[i],'description']= df_events.loc[attachIndex[i],'description'].split(" ")[1]
                            
                            for index, row in df_events.iterrows():
                                if row["description"] == "Moving started ":
                                    df_events.at[index, "description"] = "free"
                                    # assign "free" to all the events that are not  between "attach" or "release" events    
                                elif row["description"] == "Left click":
                                    df_events.at[index, "description"] = "free"
                                    #assign "free" to events that are "left click"
                                elif row["description"].startswith('Glue'):
                                    df_events.at[index, "description"] = "Glue"
                                elif row["description"].startswith('Unglue'):
                                    df_events.at[index, "description"] = "Unglue"
                                    #for now the only row as "Glue" or "Unglue" is marked without details of glueing action


                            possible_interactions = df_events["description"].unique()
                            #add the list of all possible interactions to a set:
                            for i in range(len(possible_interactions)):
                                set_of_all_possible_interactions.add(possible_interactions[i])
                            
                        except:
                            #as for some dataframes, the events column is not available, we pass
                            pass

        set_of_all_possible_interactions=list(set_of_all_possible_interactions)
        #order the list of all possible interactions alphabetically
        set_of_all_possible_interactions.sort(reverse=True)
        return set_of_all_possible_interactions

def df_from_json(file):
    try:
        df = pd.DataFrame(file)
    except :
        df = pd.DataFrame(file, index=[0])
    return df

def label_encoder(labels, set_of_all_possible_interactions):

    #labels are raw labels from the json files
    #set_of_all_possible_interactions is the set of all possible interactions existing in the puzzle

    onehot_encoder = OneHotEncoder(categories=[set_of_all_possible_interactions],sparse_output=False)
    labels = np.array(labels).reshape(-1,1) 
    onehot_encoded=onehot_encoder.fit_transform(labels)
    transformed_description=onehot_encoded
    transformed_description=np.array(transformed_description,dtype=np.double)
    transformed_description=transformed_description.T
    return transformed_description

def getSolutionSequences(df, puzzleNumber, sequence_type):
    """
    returns the solution sequences of given puzzle solution as dataframe depends on the type of sequence requested

    type of sequence points: 

    color: [type of interaction one hot encoded ] , 
    color-time: [type of interaction one hot encoded, duration of interaction], 
    color-trajectory [type of interaction one hot encoded, x, y]
    string-time: [type of interaction as string, duration of interaction]
    
    """
    set_of_all_possible_interactions = possibleInteraction(puzzleNumber)

    # try:
        #relevant columns from the dataframe from events column

    events = df["events"]
    df_events = pd.DataFrame(events)
    df_events["code"] = df['events'].apply(lambda x: x.get('code'))
    df_events["timestamp"] = df['events'].apply(lambda x: x.get('timestamp'))
    #the time stamp (unix time)  
    df_events['timestamp'] = df_events['timestamp'].str.split('-').str[0] 
    df_events['timestamp'] = df_events['timestamp'].astype(int)
    df_events['timestamp'] = pd.to_datetime(df_events['timestamp'], unit='us')
    df_events["x"] = df['events'].apply(lambda x: x.get('x'))
    df_events["y"] = df['events'].apply(lambda x: x.get('y'))
    df_events["description"] = df['events'].apply(lambda x: x.get('description'))

    df_events = df_events.drop('events', axis=1)

    # set and transform the description column to produce the desired labels
    for index, row in df_events.iterrows():
        if row["description"] == "Moving started ":
            df_events.at[index, "description"] = "free"
            # assign "free" to all the events that are not  between "attach" or "release" events    
        elif row["description"] == "Left click":
            df_events.at[index, "description"] = "free"
            #assign "free" to events that are "left click"
        elif row["description"].startswith('Glue'):
            df_events.at[index, "description"] = "Glue"
        elif row["description"].startswith('Unglue'):
            df_events.at[index, "description"] = "Unglue"
            #for now the only row as "Glue" or "Unglue" is marked without details of glueing action
    
    attachIndex = (df_events.index[df_events['description'].str.contains("Attach")]).tolist()
    releaseIndex = (df_events.index[df_events['description'].str.contains("Release")]).tolist()

    # glueIndex = (df_events.index[df_events['description'].str.startswith("Glue")]).tolist()

    
    for i in range(len(attachIndex)):
        df_events.loc[attachIndex[i]:releaseIndex[i],'description']= df_events.loc[attachIndex[i],'description'].split(" ")[1]

        
        

    time_stamp=df_events['timestamp'].values
    x=df_events['x'].values
    y=df_events['y'].values
    description=df_events['description'].values
    onHotDescription=label_encoder(description, set_of_all_possible_interactions)

    if sequence_type == "string-time":
        #in this case we return the sequence is a list of lists, each list contains the type of interaction and its duration
        sequence=[]
        state=description[0]
        start=time_stamp[0]
        for i in range(1,len(description)):
            if description[i]!=state:
                end=time_stamp[i-1]
                duration=end-start
                duration=duration.astype('timedelta64[ns]').astype(int)/1000000000
                if duration !=0: sequence.append([state,duration])
                state=description[i]
                start=time_stamp[i]
        sequence=np.array(sequence)
        return sequence
    
    elif sequence_type == "color-trajectory":
            #in this case we return the sequence as numpy array, each row contains x,y and the type of interaction one hot encoded a
            sequence=np.vstack((x,y))
            for i in range(len(onHotDescription)):
                sequence=np.vstack((sequence,onHotDescription[i]))
                # print(transformed_description[i])
            sequence=sequence.T
            return sequence
    
    elif sequence_type == "color-time":
        #issue: singular tree
        #in this case we return the sequence as numpy array, each row contains the type of interaction one hot encoded and its duration
        sequence=getSolutionSequences(df, puzzleNumber, sequence_type="string-time")
        string_color=sequence[:,0]
        color=label_encoder(string_color, set_of_all_possible_interactions)
        duration=sequence[:,1]
        sequence=np.vstack((color,duration))
        # sequence=sequence.T
        sequence=np.array(sequence,dtype=np.double)
        return sequence
    

    
    elif sequence_type == "color":
        #in this case we return the sequence as numpy array, each row contains the type of interaction one hot encoded
        sequence=getSolutionSequences(df, puzzleNumber, sequence_type="string-time")
        string_color=sequence[:,0]
        color=label_encoder(string_color, set_of_all_possible_interactions)
        # color=color.T
        return color

def getAllSolution(puzzleNumber, sequence_type):
    """
    returns all solutions sequences of requested puzzle for solved solutions and the ids of the participants who solved the puzzle
    """

    sequences=[]
    ids=[]
    
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.json'):

            #the participant id, run, puzzle, attempt from the file name:
            participant_id, run, puzzle, attempt = HMPlotter.use_regex(filename)

            if puzzleNumber == puzzle:
                
                with open(os.path.join(folder, filename)) as json_file:

                    data = json.load(json_file)
                    df=movementTracker.df_from_json(data)
                    
                    solved_stats=movementTracker.interaction(df, participant_id, run,type="total",solved=True)
                    
                    if solved_stats== "True":

                        ids.append(str(participant_id) + "_" + str(run) + "_" +str(puzzle) + "_" +str(attempt))

                        sequence=getSolutionSequences(df, puzzleNumber, sequence_type)
                        sequences.append(sequence)
    return sequences,ids

def stacked_barplot_interaction(interaction_lists, ids, cluster_id, puzzleNumber, bold_label):

    unique_labels = possibleInteraction(puzzleNumber)
    color_map = {label: i for i, label in enumerate(unique_labels)}
    legend_patches = [mpatches.Patch(color=plt.cm.tab10(color_map[label]), label=label) for label in unique_labels]

    plt.figure(figsize=(len(ids)+5, 12))
    #set the legend and title font size to 20 and bold
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.titleweight'] = 'bold'

    for j in range(len(interaction_lists)):
        sequence = interaction_lists[j]
        # Separate the string elements and float elements into separate lists
        labels = [item[0] for item in sequence]
        lengths = [item[1] for item in sequence]
        lengths = [float(l) for l in lengths]

        # Get unique string elements and assign colors based on the unique labels
        colors = [color_map[label] for label in labels]
        #map colors to rgb values
        colors = [plt.cm.tab10(color) for color in colors]

        plt.bar(j, lengths[0], color=colors[0],edgecolor='white', width=0.4)
    
        for i in range(1, len(lengths)):
            plt.bar(j, lengths[i], bottom=sum(lengths[:i]), color=colors[i],edgecolor='white', width=0.4 )
          
        # for index, label in enumerate(plt.gca().get_xticklabels()):
        #     label.set_fontweight('normal')
        #     if ids[index] == bold_label:
        #         label.set_fontweight('bold')
        #         label.set_fontsize(12)
        #         label.set_color('red')
        #         break

        plt.xticks(range(len(ids)), ids, rotation=90)
        plt.ylabel('Time (s)', fontsize=20)
        plt.xlabel('Participant ID', fontsize=20)
        plt.title(f'Interaction sequence for cluster {cluster_id}', fontsize=20)

    plt.legend(handles=legend_patches, loc='upper right')

def hierarchyCluster(numCluster,puzzleNumber, sequence_type):

    sequences,ids = getAllSolution(puzzleNumber, sequence_type)
    # print("\n and it is composed by: \n {}, ... \n, {}".format(sequences[0], sequences[2]))
  
    stringtime_sequences,_=getAllSolution(puzzleNumber, "string-time")

    #dependent DTW distance in upper triangular matrix
    distance_matrix=dtw.distance_matrix_fast(sequences, compact=True)
    # print(distance_matrix)

    Linkage_matrix = linkage(distance_matrix, method='ward', metric='euclidean')

    #convert upper triangular matrix to square matrix
    distance_matrix = squareform(distance_matrix)
    # print(distance_matrix)

    clusters = fcluster(Linkage_matrix, numCluster, criterion='maxclust')

    cluster_ids = {}

    # Iterate over the data points and assign them to their respective clusters
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_ids:
            cluster_ids[cluster_id] = []
        cluster_ids[cluster_id].append(ids[i])

    """
    visualize the clusters
    """

    if not os.path.exists(f'./Plots_Text/clustering/puzzle{puzzleNumber}_{sequence_type}'):
        os.makedirs(f'./Plots_Text/clustering/puzzle{puzzleNumber}_{sequence_type}')
        plotPath=f'./Plots_Text/clustering/puzzle{puzzleNumber}_{sequence_type}'
    else:
        plotPath=f'./Plots_Text/clustering/puzzle{puzzleNumber}_{sequence_type}'
    
    plt.figure(figsize=(20, 10))
    dendrogram(Linkage_matrix,labels=ids)
    plt.title('Hierarchical Clustering Dendrogram of sequences of puzzle '+str(puzzleNumber)+ " with sequence type "+sequence_type, fontweight='bold')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.savefig(f'{plotPath}/Dendrogram_puzzle{puzzleNumber}_{sequence_type}.png', dpi=300)
    plt.close()
    
    for cluster_id, data_ids in cluster_ids.items():
        list_of_index=[]
        # print(f"Cluster {cluster_id}: {data_ids}")
        interaction_lists_cluster = []
        for data_id in data_ids:
            for i in range(len(ids)):
                if ids[i] == data_id:
                    list_of_index.append(i)
                    interaction_lists_cluster.append(stringtime_sequences[i])
                    break
        # print(f"Cluster {cluster_id}: {data_ids}")
        # print(f"Cluster {cluster_id}: {list_of_index}")

        list_of_index=np.array(list_of_index, dtype=int)

        #geometric median of cluster 
        min_distance=np.inf
        for i in list_of_index:
            sumofdistance=0
            for j in list_of_index:
                sumofdistance += distance_matrix[i, j]
            if sumofdistance < min_distance:
                min_distance = sumofdistance
                meanindex = i
        print(f"Cluster {cluster_id} representor is: {ids[meanindex]}")

        stacked_barplot_interaction(interaction_lists_cluster, data_ids,cluster_id, puzzleNumber, ids[meanindex])
        plt.savefig(f'{plotPath}/Interaction_stackedbar_cluster{cluster_id}_puzzle{puzzleNumber}_{sequence_type}.png', dpi=300)
        plt.close()
    

        first_image, frames = gif(desired_puzzle=puzzleNumber,ids=data_ids)
        first_image.save(f'{plotPath}/Cluster{cluster_id}_puzzle{puzzleNumber}_{sequence_type}.gif', save_all=True, append_images=frames, duration=500, loop=0)
                
# puzzleNumber = int(input("Enter the puzzle number: "))
# numCluster = int(input("Enter the number of clusters: "))
# sequence_type = input("Enter the sequence type: ")
numCluster = 3
# for puzzleNumber in [2]:
#     sequence_type = "color-trajectory"
#     hierarchyCluster(numCluster,puzzleNumber, sequence_type)
#     print("--- %s seconds ---" % (time.time() - start_time))

for puzzleNumber in [1]:
    sequence_type = "color"
    hierarchyCluster(numCluster,puzzleNumber, sequence_type)
    print("--- %s seconds ---" % (time.time() - start_time))

