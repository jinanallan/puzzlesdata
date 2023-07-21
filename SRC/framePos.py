import json
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from gif_generator import gif
import time
start_time = time.time()

def positional_vector(data):
    """
    Get the positional vector of the objects from frames json file

    Accepts:
        data: the json file
    Reurns: 
        positional_vector: dataframe with the positional vector
        object_names: dict of object names and their IDs
    """
    data = pd.DataFrame(data)

    last_frame = data.frames[len(data.frames)-1]
    present_objects = {}
    for definition in last_frame:
        present_objects[definition["ID"]] = definition["name"]

    universal_Objects = ["box1","box2", "obj1","obj2", "obj3","obj4","ego"]
    
    for x in list(present_objects):
        if present_objects[x] not in universal_Objects:
            # print(present_objects[x])
            present_objects.pop(x)
    
    positional_vector=pd.DataFrame(columns=present_objects)
    sub_columns = pd.MultiIndex.from_product([positional_vector.columns, ['x', 'y']], names=['ID', 'position'])
    positional_vector = pd.DataFrame(index=range(len(data.frames)), columns=sub_columns)

    row=0
    for frame in data.frames:
        # print(frame)
        for object in frame:
            if object["ID"] in present_objects.keys():
                # print(object["ID"], present_objects[object["ID"]])
                id=object["ID"]
                # id=str(id)
                x=object["X"][0]
                y=object["X"][1]
                positional_vector.at[row,(id,'x')]=x
                positional_vector.at[row,(id,'y')]=y
        row+=1
    return positional_vector, present_objects

def use_regex(input_text):
    pattern = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_frames", re.IGNORECASE)

    match = pattern.match(input_text)
    
    particpants = match.group(3)
    run = match.group(4)
    puzzle_id = match.group(5)
    attempt = match.group(6)
    return int(particpants), int(run), int(puzzle_id), int(attempt)

frame_folder= "./Data/Frames/"
frame_files = os.listdir(frame_folder)

puzzleNumber=5
sequence_type="POSVEC"
numCluster = 5


allSV=[]
ids=[]
for file in frame_files:
    if file.endswith(".json"):
        participant_id, run, puzzle, attempt = use_regex(file)
        if puzzle == puzzleNumber:
            ids.append(str(participant_id) + "_" + str(run) + "_" +str(puzzle) + "_" +str(attempt))
            with open(os.path.join(frame_folder,file)) as json_file:
                data = json.load(json_file)
                vector, object_names = positional_vector(data)
                # print(vector)
                # print(object_names)
                d=len(vector.columns)        
                n=len(vector.index)
                # print(n,d)
                solutionVector = np.empty([n,d])
                for ni in range(n):
                    for di in range(d):
                        solutionVector[ni][di]=vector.iloc[ni,di]
                allSV.append(solutionVector)
            
                # print(solutionVector)
# print(len(allSV))
# print(len(ids))
distanceMatrix = dtw.distance_matrix_fast(allSV, compact=True)
# # print(distanceMatrix)
# plt.figure(figsize=(10, 10))
# plt.imshow(distanceMatrix, cmap='hot', interpolation='nearest')
# plt.show()

if not os.path.exists(f'./Plots_Text/clustering/puzzle{puzzleNumber}_{sequence_type}'):
        os.makedirs(f'./Plots_Text/clustering/puzzle{puzzleNumber}_{sequence_type}')
        plotPath=f'./Plots_Text/clustering/puzzle{puzzleNumber}_{sequence_type}'
else:
    plotPath=f'./Plots_Text/clustering/puzzle{puzzleNumber}_{sequence_type}'

Z = linkage(distanceMatrix, 'ward')

plt.figure(figsize=(20, 10))
dendrogram(Z, labels=ids)
plt.savefig(f'{plotPath}/dendrogram_puzzle{puzzleNumber}_{sequence_type}.png')
# plt.show()
clusters = fcluster(Z, numCluster, criterion='maxclust')

cluster_ids = {}
# Iterate over the data points and assign them to their respective clusters
for i, cluster_id in enumerate(clusters):
    if cluster_id not in cluster_ids:
        cluster_ids[cluster_id] = []
    cluster_ids[cluster_id].append(ids[i])

for cluster_id, data_ids in cluster_ids.items():
    # print('Cluster ID: {}'.format(cluster_id))
    # print('Data IDs: {}\n'.format(data_ids))

    first_image, frames = gif(desired_puzzle=puzzleNumber,ids=data_ids)
    first_image.save(f'{plotPath}/Cluster{cluster_id}_puzzle{puzzleNumber}_{sequence_type}.gif', save_all=True, append_images=frames, duration=500, loop=0)




print("--- %s seconds ---" % (time.time() - start_time))  


# TODO: define objects and their IDs present in the last frame **DONE
# store the X,Y,Z, rotation of each object in a list **Done
#study the velocity profile and match with the intraction 
# concatenate the dataframe to a single vector **done 
# Cluster the vectors using DTW **done 


