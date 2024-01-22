import json
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import matplotlib.cm as cm
from dtaidistance import dtw
from dtaidistance import dtw_barycenter
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, silhouette_samples
from gifGenerator import gif
import time
import subprocess
import json
from clusteringEvaluation import clusteringEvaluation

start_time = time.time()

def coloring(object,dummy = False):
    if dummy:
        if object=='box1':
            return (0,0,1) 
        elif object=='box2':
            return (0,1,0) 
        elif object=='obj1':
            return (1,0,0) 
        elif object=='obj1_a':
            return (1,0.5,0)
        elif object=='obj2':
            return (1,0,1) 
        elif object=='obj3':
            return (1,1,0) 
        elif object=='obj4':
            return (0,1,1) 
        elif object=='ego':
            return (0,0,0) 
    else:
        if object=='box1':
            return [(0,0,1,c) for c in np.linspace(0,1,100)]
        elif object=='box2':
            return [(0,1,0,c) for c in np.linspace(0,1,100)]
        elif object=='obj1':
            return [(1,0,0,c) for c in np.linspace(0,1,100)]
        elif object=='obj1_a':
            return [(1,0.5,0,c) for c in np.linspace(0,1,100)]
        elif object=='obj2':
            return [(1,0,1,c) for c in np.linspace(0,1,100)]
        elif object=='obj3':
            return [(1,1,0,c) for c in np.linspace(0,1,100)]
        elif object=='obj4':
            return [(0,1,1,c) for c in np.linspace(0,1,100)]
        elif object=='ego':
            return [(0,0,0,c) for c in np.linspace(0,1,100)]
    
def positional_vector(data : dict, ignore_Unattached_ego : bool = False) -> pd.DataFrame():
    """
    Get the positional vector of the objects from frames json file

    Accepts:
        data: the json file

    Returns: 
        positional_vector: dataframe with the positional vector
        present_objects: dict of object names and their IDs
    """
    data = pd.DataFrame(data)

    last_frame = data.frames[len(data.frames)-1]
    present_objects = {}
    for definition in last_frame:
        present_objects[definition["ID"]] = definition["name"]

    universal_Objects = ["box1","box2", "obj1","obj2", "obj3","obj4","ego","obj1_a"]
    
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
    
    if ignore_Unattached_ego:

        velocity_vector = positional_vector.diff()
        velocity_vector = velocity_vector.drop(0)
        velocity_vector = velocity_vector.reset_index(drop=True)
        
        v=np.zeros((len(velocity_vector),len(present_objects)))

        for i, object_i in enumerate(present_objects):

            vx_i = velocity_vector[positional_vector.columns[i*2][0],'x']
            vx_i=np.array(vx_i, dtype=np.float64)
            vy_i = velocity_vector[positional_vector.columns[i*2][0],'y']
            vy_i=np.array(vy_i, dtype=np.float64)
            v_temp=np.sqrt(vx_i**2 + vy_i**2)
            v[:,i]=v_temp
        
        for step in range(1,len(v)):
            if v[step,0] == np.sum(v[step,:]) :
                positional_vector.at[step,(6,'x')] = np.nan
                positional_vector.at[step,(6,'y')] = np.nan
                    
        positional_vector[6,'x']=positional_vector[6,'x'].interpolate(method='pad')
        positional_vector[6,'y']=positional_vector[6,'y'].interpolate(method='pad')

    return positional_vector, present_objects

def dtwI(sequences : list) -> np.ndarray:
    """
    Get the distance matrix of the sequences using Independent dtw
    Accepts:
        sequences: list of sequences (each sequence is a positional vector of a puzzle solution)
    Returns:
        distanceMatrix: pairwise distance matrix of the sequences in form of scipy pdist output
    """
    n=len(sequences)
    d=len(sequences[0][0])
    distanceMatrix = np.empty([n,n])
    for i in range(n):
        for j in range(i+1,n):
            dtw_i=0
            for k in range(d):
                print(i,j,k)
                dtw_i+=dtw.distance_fast(sequences[i][:,k], sequences[j][:,k])
                # print(dtw_i)
            distanceMatrix[i][j]=dtw_i
    # distanceMatrix in form of scipy pdist  output
    distanceMatrix = distanceMatrix[np.triu_indices(n, 1)]
    return distanceMatrix

def use_regex(input_text):
    pattern = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_frames", re.IGNORECASE)

    match = pattern.match(input_text)
    
    particpants = match.group(3)
    run = match.group(4)
    puzzle_id = match.group(5)
    attempt = match.group(6)
    return int(particpants), int(run), int(puzzle_id), int(attempt)

def Heatmap(cluster_id, data_ids, puzzleNumber, ignore_ego=False, log_scale=True, dba=True ):
    """
    Output a heatmap of solutions within a cluster 

    Accepts:
        cluster_id: the cluster id
        data_ids: list of ids of solutions within the cluster
    Reurns:
        Plot of the heatmap of solutions within a cluster
    """ 
    cluster_vector = pd.DataFrame()
    n=len(data_ids)
    #solved rate 
    sr=0
    #average time
    at=0

    
    for id in data_ids:
        try:
            filenameTemp = [f for f in os.listdir('./Data/Pilot3/Ego-based/') if f.endswith(f'{id}.json')][0]
            dataTemp = json.load(open(f'./Data/Pilot3/Ego-based/{filenameTemp}'))

            try:
                df = pd.DataFrame(dataTemp)
            except:
                df = pd.DataFrame(dataTemp, index=[0])
        
    
            solved = df['solved'].values[0]
            total_time = df['total-time'].values[0]
            at+=total_time

            if solved:
                sr+=1
    

            filename = [f for f in os.listdir('./Data/Pilot3/Frames/') if f.endswith(f'{id}_frames.json')][0]
            data = json.load(open(f'./Data/Pilot3/Frames/{filename}'))
        except:
            filenameTemp= [f for f in os.listdir('./Data/Pilot4/Ego-based/') if f.endswith(f'{id}.json')][0]
            dataTemp = json.load(open(f'./Data/Pilot4/Ego-based/{filenameTemp}'))


            try:
                df = pd.DataFrame(dataTemp)
            except:
                df = pd.DataFrame(dataTemp, index=[0])

            solved = df['solved'].values[0]
            total_time = df['total-time'].values[0]
            at+=total_time

            if solved:
                sr+=1
             

            filename = [f for f in os.listdir('./Data/Pilot4/Frames/') if f.endswith(f'{id}_frames.json')][0]
            data = json.load(open(f'./Data/Pilot4/Frames/{filename}'))

        vector, present_objects = positional_vector(data)
        
        cluster_vector = pd.concat([cluster_vector, vector], axis=0)

    sr=sr/n
    at=at/n

    cluster_vector = cluster_vector.reset_index(drop=True)
    
    fig, ax = plt.subplots()
    
    imgfolder = './cropped_puzzles_screenshots'
    fname = os.path.join(imgfolder, 'puzzle'+str(puzzleNumber)+'.png')
    img = Image.open(fname).convert('L')
    img = ax.imshow(img, extent=[-2, 2, -2, 2], cmap='gray')
    for i,object in enumerate(present_objects):
        if ignore_ego and present_objects[object]=='ego':
            continue
        else:
            colors = coloring(present_objects[object])
            cmap = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)
            x = cluster_vector[object]['x']
            y = cluster_vector[object]['y']
            if log_scale:
                plt.hist2d(x, y, bins=(45, 45),cmap=cmap, norm=mcolors.LogNorm())
            else:
                plt.hist2d(x, y, bins=(45, 45),cmap=cmap)
            #dummy scatter plot for legend
            sc = plt.scatter([],[], color=coloring(present_objects[object], True), label=present_objects[object])
        
    plt.legend(title=f'Number of solutions: {n}\n Solved rate: {sr:.2f} \n Average time: {at:.2f} seconds',loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(f'cluster {cluster_id}' )
    plt.savefig(f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}_{sequence_type}/Cluster{cluster_id}_puzzle{puzzleNumber}_{sequence_type}_heatmap.png',
                bbox_inches='tight', dpi=720)
    plt.close(fig)

frame_folders = ["./Data/Pilot3/Frames/", "./Data/Pilot4/Frames/"]

sequence_type="POSVEC"
puzzels = [2] #[21,22,23,24,25,26,16,17,18,19,20]

log_scale = True
ignore_Unattached_ego = True
manual_number_of_clusters = True

for puzzleNumber in puzzels:

    if not os.path.exists(f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}_{sequence_type}'):
            os.makedirs(f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}_{sequence_type}')
            plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}_{sequence_type}'
    else:
        plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}_{sequence_type}'

    allSV=[]
    ids=[]

    for frame_folder in frame_folders:
        frame_files = os.listdir(frame_folder)
        for file in frame_files:
            if file.endswith(".json"):
                participant_id, run, puzzle, attempt = use_regex(file)
                if puzzle == puzzleNumber:
                    ids.append(str(participant_id) + "_" + str(run) + "_" +str(puzzle) + "_" +str(attempt))
                    with open(os.path.join(frame_folder,file)) as json_file:
                        data = json.load(json_file)
                        vector, object_names = positional_vector(data, ignore_Unattached_ego)
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

    if os.path.isfile(f'{plotPath}/distanceMatrix_puzzle{puzzleNumber}_{sequence_type}.txt'):
        distanceMatrix = np.loadtxt(f'{plotPath}/distanceMatrix_puzzle{puzzleNumber}_{sequence_type}.txt')
    else:               
        distanceMatrix = dtwI(allSV)
        np.savetxt(f'{plotPath}/distanceMatrix_puzzle{puzzleNumber}_{sequence_type}.txt', distanceMatrix)

    if os.path.isfile(f'{plotPath}/linkage_puzzle{puzzleNumber}_{sequence_type}.txt'):
        Z = np.loadtxt(f'{plotPath}/linkage_puzzle{puzzleNumber}_{sequence_type}.txt')
    else:
        Z = linkage(distanceMatrix, 'ward')
        np.savetxt(f'{plotPath}/linkage_puzzle{puzzleNumber}_{sequence_type}.txt', Z)

    # np.savetxt(f'{plotPath}/ids_puzzle{puzzleNumber}_{sequence_type}.txt', ids, fmt="%s")
    if manual_number_of_clusters:
        numCluster = int(input("Enter the number of clusters: "))
    else:
        distanceMatrixSQ = squareform(distanceMatrix)

        fig = clusteringEvaluation(Z,distanceMatrix,puzzleNumber)

        fig.savefig(f'{plotPath}/evaluation_puzzle{puzzleNumber}_{sequence_type}.png', dpi=300)
        print(f"evaluation_puzzle{puzzleNumber}_{sequence_type}.png saved")
        plt.close(fig)

        # Silhouette analysis plot and deciding the number of clusters based on the max silhouette score
        max_silhouette_avg = 0

        fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=False, sharey=True)

        fig.text(0.5, 0.04, 'Silhouette coefficient values', ha='center', fontsize=14)
        fig.text(0.04, 0.5, 'Cluster label', va='center', rotation='vertical', fontsize=14)
        
        for n_clusters in range(2, 10):

            ax1 = axs[(n_clusters-2)//4][(n_clusters-2)%4]
            
            clusters = fcluster(Z, n_clusters, criterion='maxclust')
            silhouette_avg = silhouette_score(distanceMatrixSQ, clusters, metric='precomputed')
            silhouette_avg = round(silhouette_avg, 3)

            print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

            if silhouette_avg > max_silhouette_avg:
                max_silhouette_avg = silhouette_avg
                numCluster = n_clusters

            sample_silhouette_values = silhouette_samples(distanceMatrixSQ, clusters, metric='precomputed')
            
            y_lower = 10
            for i in range(n_clusters):
                ith_cluster_silhouette_values = sample_silhouette_values[clusters == i+1]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))
                y_lower = y_upper + 10

            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
            ax1.set_yticks([])
            ax1.set_xticks(np.arange(-0.1, 0.8, 0.1))
            #set x tick font size 
            for tick in ax1.xaxis.get_major_ticks():
                tick.label1.set_fontsize(8)
            ax1.set_xlim([-0.1, 0.8])
            ax1.set_ylim([0, len(distanceMatrixSQ) + (n_clusters+1) * 10])
            ax1.set_title(f'Number of clusters: {n_clusters}\nSilhouette score: {silhouette_avg}', fontsize=12)


        plt.suptitle(f"Silhouette analysis for puzzle {puzzleNumber}", fontsize=14, fontweight='bold')
        plt.savefig(f'{plotPath}/silhouette_puzzle{puzzleNumber}_{sequence_type}.png', dpi=300)
        print(f"silhouette_puzzle{puzzleNumber}_{sequence_type}.png saved")
        plt.close(fig)

    clusters = fcluster(Z, numCluster, criterion='maxclust')

    cluster_ids = {}
    # Iterate over the data points and assign them to their respective clusters
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_ids:
            cluster_ids[cluster_id] = []
        cluster_ids[cluster_id].append(ids[i])
    
    #turn dict keys to int
    cluster_ids = {int(k): v for k, v in cluster_ids.items()}

    #save the cluster ids as json file
    with open(f'{plotPath}/cluster_ids_puzzle{puzzleNumber}_{sequence_type}.json', 'w') as fp:
        json.dump(cluster_ids, fp)

    for cluster_id, data_ids in cluster_ids.items():
        first_image, frames = gif(desired_puzzle=puzzleNumber,ids=data_ids, attachment=True)
        first_image.save(f'{plotPath}/Cluster{cluster_id}_puzzle{puzzleNumber}_{sequence_type}.gif', save_all=True, append_images=frames, duration=500, loop=0)
        Heatmap(cluster_id, data_ids, puzzleNumber, ignore_ego=True, log_scale=log_scale)
        
    
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    
    ax1 = plt.subplot2grid((2, numCluster), (0, 0), colspan=numCluster)
    ax1.set_title(f'Dendrogram of puzzle {puzzleNumber} solutions', fontsize=20)
    ax1.set_xlabel('Solution ID')
    # ax1.set_ylabel('Distance')
    dendrogram(Z, labels=ids, ax=ax1, leaf_font_size=10 )
    #horizontal line where we cut the dendrogram
    plt.axhline(y=Z[-numCluster+1,2], color='black', linestyle='--')
    
    #pad between dendrogram and heatmap
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0.4)

    plt.figtext(0.5, 0.45, "Heatmap of solutions within each cluster", ha="center", va="center", fontsize=20)

    for i in np.arange(1,numCluster+1):
        ax2 = plt.subplot2grid((2, numCluster), (1, i-1))
        ax2.imshow(Image.open(f'{plotPath}/Cluster{i}_puzzle{puzzleNumber}_{sequence_type}_heatmap.png')) 
        ax2.set_axis_off()
    plt.savefig(f'{plotPath}/dendrogram_heatmap_puzzle{puzzleNumber}.png', dpi=300)
          
print("--- %s seconds ---" % (time.time() - start_time)) 

# repo_path = './'

# os.chdir(repo_path)

# subprocess.run(['git', 'add', '.'])

# subprocess.run(['git', 'commit', '-m', "Selecting the number of clusters with silhouette analysis"])

# subprocess.run(['git', 'push'])
