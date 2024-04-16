import json
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from PIL import Image
import matplotlib.cm as cm
from dtaidistance import dtw
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import soft_dtw, cdist_soft_dtw_normalized
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score, silhouette_samples
from gifGenerator import gif
import time
import subprocess
import json
from clusteringEvaluation import clusteringEvaluation
import torch
import multiprocessing

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
    
def positional_vector(data : dict, weighted : bool = False, w: int = 50, concat_state : bool = False, ignore_Unattached_ego : bool = False, total_time : bool = False) -> pd.DataFrame: 
    """
    Get the positional vector of the objects from frames json file

    Accepts:
        data: the json file
        weighted: if True, the state of the objects will be weighted
        w: the weight of the state
        concat_state: if True, the state of the objects will be concatenated to the positional vector
        ignore_Unattached_ego: if True, the positional vector of the ego will be ignored if it is not attached to any object
        total_time: if True, the total time of the solution will be returned
        
    Returns: 
        positional_vector: dataframe with the positional vector
        present_objects: dict of object names and their IDs
        total_time: total time of the solution
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

    ego_id= [k for k,v in present_objects.items() if v=='ego'][0]

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

    if ignore_Unattached_ego:
        for step in range(1,len(v)):
            if v[step,0] == np.sum(v[step,:]) :
                positional_vector.at[step,(ego_id,'x')] = np.nan
                positional_vector.at[step,(ego_id,'y')] = np.nan
                    
        positional_vector[ego_id,'x']=positional_vector[ego_id,'x'].interpolate(method='pad')
        positional_vector[ego_id,'y']=positional_vector[ego_id,'y'].interpolate(method='pad')
    
    if concat_state:
            
        states = np.zeros((len(velocity_vector),len(present_objects)-1))
        for i, object_i in enumerate(present_objects):
            object_i_name = present_objects[object_i]
            # print(i, object_i)

            if i != 0:
                same_as_ego = np.where(v[:,i] == v[:,0])
                vline = v[:,i][same_as_ego]
                same_as_ego = np.delete(same_as_ego, np.where(vline == 0))
                states[same_as_ego,i-1] = 1

        if weighted:
            weight = positional_vector.std(ddof=0)
            #exclude the box1 and box2
            # print(weight)
            for i, object_i in enumerate(present_objects):
                object_i_name = present_objects[object_i]
                if object_i_name == 'box1' or object_i_name == 'box2':
                    weight = weight.drop(object_i)
            # print(weight)
            weight = weight.sum()
            # print(weight)
            weight = weight*w
            states = states*weight

        states = pd.DataFrame(states, columns=[present_objects[object] for object in present_objects if present_objects[object] != 'ego'])
        # return states, present_objects
        positional_vector = pd.concat([positional_vector, states], axis=1)
        positional_vector = positional_vector.dropna()
    t = len(positional_vector)*0.01
    t=round(t,2)
    if total_time:
        return positional_vector, present_objects, t
    else:
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

def softdtw_score(puzzle: int, sequences : list, torch_be : bool, gamma: float, device=None ) -> np.ndarray:
    n=len(sequences)
    if torch_be:
        if device is None:
            raise ValueError("Device must be specified when torch_be is True.")
        for i in range(n):
            # print(f"{i}th time series from {n}")
            sequences[i]=torch.from_numpy(sequences[i]).float()
            if torch.cuda.is_available():  # Check if GPU is available
                sequences[i] = sequences[i].to(device)  # Move tensor to GPU

        print(f"computing distance matrix based on normalized softdtw score with pytorch backend for puzzle {puzzle}")
        distanceMatrix = cdist_soft_dtw_normalized(sequences, gamma=gamma, be="pytorch", compute_with_backend=torch_be)
        print("end of distance computation")
    else:
        print(f"computing distance matrix based on normalized softdtw score with numpy backend for puzzle {puzzle}")
        distanceMatrix = cdist_soft_dtw_normalized(sequences, gamma=gamma, be="numpy")
        print("end of distance computation")
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

def Heatmap(cluster_id, data_ids, puzzleNumber, pathplot,ignore_ego=False, log_scale=True ):
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
    plt.savefig(f'{pathplot}/Cluster{cluster_id}_puzzle{puzzleNumber}_heatmap.png',
                bbox_inches='tight', dpi=720)
    plt.close(fig)

def softbarycenter(cluster_id, data_ids, puzzleNumber, pathplot):
    cluster_vector = list() 

    for id in data_ids:
        try:
            
            filename = [f for f in os.listdir('./Data/Pilot3/Frames/') if f.endswith(f'{id}_frames.json')][0]
            data = json.load(open(f'./Data/Pilot3/Frames/{filename}'))

        except:
            
            filename = [f for f in os.listdir('./Data/Pilot4/Frames/') if f.endswith(f'{id}_frames.json')][0]
            data = json.load(open(f'./Data/Pilot4/Frames/{filename}'))

        vector, present_objects = positional_vector(data)
   
        d=len(vector.columns)
        n=len(vector.index)
        solutionVector = np.empty([n,d])
        for ni in range(n):
            for di in range(d):
                solutionVector[ni][di]=vector.iloc[ni,di]
        cluster_vector.append(solutionVector)

    if len(cluster_vector) > 1:
        print(len(cluster_vector))

        #random selection of a solution to initialize the barycenter
        # idx = np.random.choice(len(cluster_vector), 1, replace=False)
        # idx = idx[0]
        idx = np.argmin([len(cluster_vector[i]) for i in range(len(cluster_vector))])

        if os.path.isfile(f'{pathplot}/Cluster{cluster_id}_puzzle{puzzleNumber}_softbarycenter.json'):
            with open(f'{pathplot}/Cluster{cluster_id}_puzzle{puzzleNumber}_softbarycenter.json', 'r') as fp:
                avg = np.array(json.load(fp))
        else:
            avg = softdtw_barycenter(cluster_vector, gamma=1.0, max_iter=50, tol=1e-3, init=cluster_vector[idx])
            #save the barycenter
            with open(f'{pathplot}/Cluster{cluster_id}_puzzle{puzzleNumber}_softbarycenter.json', 'w') as fp:
                json.dump(avg.tolist(), fp)
        
        window_size = 5 # Adjust the window size as needed
        filtered = np.zeros_like(avg)

        for i in range(avg.shape[1]):
            # print(i)
            filtered[:, i] = np.convolve(avg[:, i], np.ones(window_size) / window_size, mode='same')

        for i in range(avg.shape[1]):
            filtered[:, i] = np.convolve(avg[:, i], np.ones(window_size) / window_size, mode='same')

        filtered= filtered[window_size:-window_size]
        avg= filtered

        fig, ax = plt.subplots()
        imgfolder = './cropped_puzzles_screenshots'
        fname = os.path.join(imgfolder, 'puzzle'+str(puzzleNumber)+'.png')
        img = Image.open(fname).convert('L')
        img = ax.imshow(img, extent=[-2, 2, -2, 2], cmap='gray')
        for i,object in enumerate(present_objects):

            x = avg[:,i*2]
            y = avg[:,i*2+1]
            
            plt.scatter(x,y, alpha=0.1, color= coloring(present_objects[object], dummy=True), s=10, edgecolors='face',
                                        marker= ".", label=present_objects[object])
            plt.legend(title=f'Number of solutions: {len(cluster_vector)}, Avg time: {len(avg)*0.01 :.2f} s',loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4)
            
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
        plt.title(f'cluster {cluster_id} barycenter' )
        plt.savefig(f'{pathplot}/Cluster{cluster_id}_puzzle{puzzleNumber}_softbarycenter.png',
                    bbox_inches='tight', dpi=720)
        plt.close(fig)
        
        fig, ax = plt.subplots()
        #open image as float
        img = Image.open(fname).convert('L')
        img = ax.imshow(img, extent=[-2, 2, -2, 2], cmap='gray')

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        num_objects = len(present_objects) -1
         
        num_points = len(avg)
        print(num_points)

        lines =[ax.plot([], [], lw=2)[0] for _ in range(num_objects)]

        # Function to initialize the animation
        def init():
            for line in lines:
                line.set_data([], [])
            return lines
        
        # Function to update the animation
        def update(frame):
            for i,object in enumerate(present_objects):
                if not present_objects[object]=='ego':
                    x = avg[:,i*2]
                    y = avg[:,i*2+1]
                    lines[i-1].set_data(x[:frame], y[:frame])
                    lines[i-1].set_color(coloring(present_objects[object], dummy=True))  # Add color to the trajectory
            # Add time count and label on the frame
            # ax.text.clear()
            # ax.text(1.8, -1.8, f"Time: {frame*0.05:.2f}s", fontsize=10, ha='right', va='bottom')
            return lines
        
        # Create the animation
        print(f"Creating softbarycenter gif for cluster {cluster_id}")
        ani = FuncAnimation(fig, update, frames=num_points, init_func=init, blit=True, interval=0)
        print(f"Saving softbarycenter gif for cluster {cluster_id}")
        ani.save(f'{pathplot}/Cluster{cluster_id}_puzzle{puzzleNumber}_softbarycenter.gif', writer='pillow', fps=100)
        print(f"softbarycenter gif for cluster {cluster_id} saved")
        plt.close(fig)

def silhouette_analysis(Z, distanceMatrixSQ, puzzleNumber,plotPath):
    # Silhouette analysis plot and deciding the number of clusters based on the max silhouette score
    max_silhouette_avg = 0

    fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=False, sharey=True)

    fig.text(0.5, 0.04, 'Silhouette coefficient values', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Cluster label', va='center', rotation='vertical', fontsize=14)
    
    neg_value_fraction= []
    below_avg_fraction= []
    for n_clusters in range(3, 10):
        ax1 = axs[(n_clusters-2)//4][(n_clusters-2)%4]
        
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        silhouette_avg = silhouette_score(distanceMatrixSQ, clusters, metric='precomputed')
        silhouette_avg = round(silhouette_avg, 2)

        # print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

        if silhouette_avg > max_silhouette_avg:
            max_silhouette_avg = silhouette_avg
            numCluster = n_clusters

        sample_silhouette_values = silhouette_samples(distanceMatrixSQ, clusters, metric='precomputed')

        neg_value_fraction.append(sum(sample_silhouette_values < 0) / len(sample_silhouette_values))

        below_avg_fraction.append(sum(sample_silhouette_values < silhouette_avg) / len(sample_silhouette_values))

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[clusters == i+1]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.4)
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
    plt.savefig(f'{plotPath}/silhouette_puzzle{puzzleNumber}.png', dpi=300)
    # print(f"silhouette_puzzle{puzzleNumber}.png saved")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(range(3, 10), neg_value_fraction, label='Negative value fraction')
    ax.plot(range(3, 10), below_avg_fraction, label='Below average fraction')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Fraction')
    ax.legend()
    plt.title(f"Negative and below average fraction for puzzle {puzzleNumber}")
    plt.savefig(f'{plotPath}/silhouette_fraction_puzzle{puzzleNumber}.png', dpi=300)

    return numCluster,neg_value_fraction,below_avg_fraction

def do_cluster(**kwargs):
    """
    Main function to do the clustering of the positional vectors
    """
    start_time = time.time()

    frame_folders = ["./Data/Pilot3/Frames/", "./Data/Pilot4/Frames/"]

    if "torch" in kwargs and kwargs["torch"]: # Check if the user wants to use PyTorch

        # Specify the GPU you want to use
        gpu_id =5  # Change this to the GPU ID you want to use

        # Set the GPU device if CUDA is available, otherwise use CPU
        if torch.cuda.is_available():
            device = torch.cuda.set_device(gpu_id)
            print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        else:
            device = torch.device("cpu")
            print("CUDA is not available. Using CPU.")
    else:
        device = None

    if "puzzles" in kwargs:
        puzzles = kwargs["puzzles"]
    else:
        puzzles = [1,2,3,4,5,6,21,22,23,24,25,26]

    if "preprocessing" in kwargs:
        preprocessing = kwargs["preprocessing"]
    else:
        preprocessing = False
        
    if "softdtwscore" in kwargs:
        softdtwscore = kwargs["softdtwscore"]
    else:
        softdtwscore = True

    if "ignore_Unattached_ego" in kwargs:
        ignore_Unattached_ego = kwargs["ignore_Unattached_ego"]
    else:
        ignore_Unattached_ego = False

    if "torch_be" in kwargs: # Check if the user wants to use PyTorch as backend for softdtw
        torch_be = kwargs["torch_be"]
    else:
        torch_be = False

    if "manual_number_of_clusters" in kwargs:
        manual_number_of_clusters = kwargs["manual_number_of_clusters"]
    else:
        manual_number_of_clusters = False

    if "ignore_ego_visualization" in kwargs:
        ignore_ego_visualization = kwargs["ignore_ego_visualization"]
    else:
        ignore_ego_visualization = False

    if "log_scale" in kwargs:
        log_scale = kwargs["log_scale"]
    else:
        log_scale = True

    if "gamma" in kwargs:
        gamma = kwargs["gamma"]
    else:
        gamma = 1.
    
    if "state" in kwargs:
        state = kwargs["state"]
    else:
        state = False
    
    for puzzleNumber in puzzles:
        if softdtwscore and ignore_Unattached_ego:
            if preprocessing:
                if not os.path.exists(f'./Plots_Text/clustering/Ignore_unattached_ego/softdtwscore/puzzle{puzzleNumber}/preprocessing/'):
                    os.makedirs(f'./Plots_Text/clustering/Ignore_unattached_ego/softdtwscore/puzzle{puzzleNumber}/preprocessing/')
                    plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/softdtwscore/puzzle{puzzleNumber}/preprocessing/'
                else:
                    plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/softdtwscore/puzzle{puzzleNumber}/preprocessing/'
            if state:
                if not os.path.exists(f'./Plots_Text/clustering/Ignore_unattached_ego/softdtwscore/puzzle{puzzleNumber}/state/'):
                    os.makedirs(f'./Plots_Text/clustering/Ignore_unattached_ego/softdtwscore/puzzle{puzzleNumber}/state/')
                    plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/softdtwscore/puzzle{puzzleNumber}/state/'
                else:
                    plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/softdtwscore/puzzle{puzzleNumber}/state/'
            else:
                if not os.path.exists(f'./Plots_Text/clustering/Ignore_unattached_ego/softdtwscore/puzzle{puzzleNumber}'):
                    os.makedirs(f'./Plots_Text/clustering/Ignore_unattached_ego/softdtwscore/puzzle{puzzleNumber}')
                    plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/softdtwscore/puzzle{puzzleNumber}'
                else:
                    plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/softdtwscore/puzzle{puzzleNumber}'
        elif softdtwscore:
            if preprocessing:
                if not os.path.exists(f'./Plots_Text/clustering/softdtwscore/puzzle{puzzleNumber}/preprocessing/'):
                    os.makedirs(f'./Plots_Text/clustering/softdtwscore/puzzle{puzzleNumber}/preprocessing/')
                    plotPath=f'./Plots_Text/clustering/softdtwscore/puzzle{puzzleNumber}/preprocessing/'
                else:
                    plotPath=f'./Plots_Text/clustering/softdtwscore/puzzle{puzzleNumber}/preprocessing/'
            if state:
                if not os.path.exists(f'./Plots_Text/clustering/softdtwscore/puzzle{puzzleNumber}/state/'):
                    os.makedirs(f'./Plots_Text/clustering/softdtwscore/puzzle{puzzleNumber}/state/')
                    plotPath=f'./Plots_Text/clustering/softdtwscore/puzzle{puzzleNumber}/state/'
                else:
                    plotPath=f'./Plots_Text/clustering/softdtwscore/puzzle{puzzleNumber}/state/'
            else:
                if not os.path.exists(f'./Plots_Text/clustering/softdtwscore/puzzle{puzzleNumber}'):
                    os.makedirs(f'./Plots_Text/clustering/softdtwscore/puzzle{puzzleNumber}')
                    plotPath=f'./Plots_Text/clustering/softdtwscore/puzzle{puzzleNumber}'
                else:
                    plotPath=f'./Plots_Text/clustering/softdtwscore/puzzle{puzzleNumber}'
        elif ignore_Unattached_ego:
            if preprocessing:
                if not os.path.exists(f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}/preprocessing/'):
                    os.makedirs(f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}/preprocessing/')
                    plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}/preprocessing/'
                else:
                    plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}/preprocessing/'
            if state:
                if not os.path.exists(f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}/state/'):
                    os.makedirs(f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}/state/')
                    plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}/state/'
                else:
                    plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}/state/'
            else:
                if not os.path.exists(f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}'):
                    os.makedirs(f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}')
                    plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}'
                else:
                    plotPath=f'./Plots_Text/clustering/Ignore_unattached_ego/puzzle{puzzleNumber}'
        else:
            if preprocessing:
                if not os.path.exists(f'./Plots_Text/clustering/puzzle{puzzleNumber}/preprocessing/'):
                    os.makedirs(f'./Plots_Text/clustering/puzzle{puzzleNumber}/preprocessing/')
                    plotPath=f'./Plots_Text/clustering/puzzle{puzzleNumber}/preprocessing/'
                else:
                    plotPath=f'./Plots_Text/clustering/puzzle{puzzleNumber}/preprocessing/'
            if state:
                if not os.path.exists(f'./Plots_Text/clustering/puzzle{puzzleNumber}/state/'):
                    os.makedirs(f'./Plots_Text/clustering/puzzle{puzzleNumber}/state/')
                    plotPath=f'./Plots_Text/clustering/puzzle{puzzleNumber}/state/'
                else:
                    plotPath=f'./Plots_Text/clustering/puzzle{puzzleNumber}/state/'
            else:
                if not os.path.exists(f'./Plots_Text/clustering/puzzle{puzzleNumber}'):
                    os.makedirs(f'./Plots_Text/clustering/puzzle{puzzleNumber}')
                    plotPath=f'./Plots_Text/clustering/puzzle{puzzleNumber}'
                else:
                    plotPath=f'./Plots_Text/clustering/puzzle{puzzleNumber}'
        allSV=[]
        ids=[]
        total_time_list = []
        #set of present objects
        present_objects = {}

        for frame_folder in frame_folders:
            frame_files = os.listdir(frame_folder)
            for file in frame_files:
                if file.endswith(".json"):
                    participant_id, run, puzzle, attempt = use_regex(file)
                    if puzzle == puzzleNumber:
                        ids.append(str(participant_id) + "_" + str(run) + "_" +str(puzzle) + "_" +str(attempt))
                        with open(os.path.join(frame_folder,file)) as json_file:
                            data = json.load(json_file)
                            if preprocessing:
                                vector, object_names, total_time = positional_vector(data, ignore_Unattached_ego, total_time=preprocessing)
                                present_objects.update(object_names)

                                d=len(vector.columns)        
                                n=len(vector.index)

                                solutionVector = np.empty([n,d])
                                for ni in range(n):
                                    for di in range(d):
                                        solutionVector[ni][di]=vector.iloc[ni,di]

                                allSV.append(solutionVector)
                                total_time_list.append(total_time)
                            
                            elif state:
                                vector, object_names = positional_vector(data,weighted=True, concat_state=True)
                                present_objects.update(object_names)
                                
                                d=len(vector.columns)        
                                n=len(vector.index)

                                solutionVector = np.empty([n,d])
                                for ni in range(n):
                                    for di in range(d):
                                        solutionVector[ni][di]=vector.iloc[ni,di]

                                allSV.append(solutionVector)

                            else:
                                vector, object_names = positional_vector(data, ignore_Unattached_ego, total_time=preprocessing)
                                present_objects.update(object_names)
                        
                                d=len(vector.columns)        
                                n=len(vector.index)

                                solutionVector = np.empty([n,d])
                                for ni in range(n):
                                    for di in range(d):
                                        solutionVector[ni][di]=vector.iloc[ni,di]

                                allSV.append(solutionVector)
        #save present objects as json
        with open(f'{plotPath}/present_objects_puzzle{puzzleNumber}.json', 'w') as fp:
            json.dump(present_objects, fp)
            
        if preprocessing:
            # print(total_time_list)
            ouliers=[]
            median_total_time = np.median(total_time_list)
            MAD = np.median([np.abs(x - median_total_time) for x in total_time_list])
            # print(f"Median total time: {median_total_time}")
            # print(f"MAD: {MAD}")

            for i in range(len(total_time_list)):
                if np.abs(total_time_list[i] - median_total_time) > 5*MAD:
                    ouliers.append(i)

            # print([i for j, i in enumerate(ids) if j in ouliers])
            # print([i for j, i in enumerate(total_time_list) if j in ouliers])
            allSV = [i for j, i in enumerate(allSV) if j not in ouliers]
            ids = [i for j, i in enumerate(ids) if j not in ouliers]
            # print(f"Removed {len(ouliers)} outliers")

        if os.path.isfile(f'{plotPath}/distanceMatrix_puzzle{puzzleNumber}.txt'):
            distanceMatrix = np.loadtxt(f'{plotPath}/distanceMatrix_puzzle{puzzleNumber}.txt')
        elif softdtwscore:
            if device is not None:
                distanceMatrix = softdtw_score(puzzleNumber,allSV, torch_be=torch_be,gamma=gamma, device=device)
            else:
                distanceMatrix = softdtw_score(puzzleNumber,allSV, torch_be=torch_be, gamma=gamma)
            np.savetxt(f'{plotPath}/distanceMatrix_puzzle{puzzleNumber}.txt', distanceMatrix)
        else:               
            distanceMatrix = dtwI(allSV)
            np.savetxt(f'{plotPath}/distanceMatrix_puzzle{puzzleNumber}.txt', distanceMatrix)

        if os.path.isfile(f'{plotPath}/linkage_puzzle{puzzleNumber}.txt'):
            Z = np.loadtxt(f'{plotPath}/linkage_puzzle{puzzleNumber}.txt')
        else:
            Z = linkage(distanceMatrix, 'ward')
            np.savetxt(f'{plotPath}/linkage_puzzle{puzzleNumber}.txt', Z)

        # np.savetxt(f'{plotPath}/ids_puzzle{puzzleNumber}.txt', ids, fmt="%s")
        if manual_number_of_clusters:
            numCluster = int(input("Enter the number of clusters: "))

        elif not os.path.isfile(f'{plotPath}/evaluation_puzzle{puzzleNumber}.png'):
            distanceMatrixSQ = squareform(distanceMatrix)

            fig = clusteringEvaluation(Z,distanceMatrix,puzzleNumber)

            fig.savefig(f'{plotPath}/evaluation_puzzle{puzzleNumber}.png', dpi=300)
            print(f"evaluation_puzzle{puzzleNumber}.png saved")
            plt.close(fig)

            numCluster,neg_value_fraction,below_avg_fraction = silhouette_analysis(Z, distanceMatrixSQ, puzzleNumber, plotPath)

        if not os.path.isfile(f'{plotPath}/cluster_ids_puzzle{puzzleNumber}.json'):
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
            with open(f'{plotPath}/cluster_ids_puzzle{puzzleNumber}.json', 'w') as fp:
                json.dump(cluster_ids, fp)
        else:
            with open(f'{plotPath}/cluster_ids_puzzle{puzzleNumber}.json', 'r') as fp:
                cluster_ids = json.load(fp)
            numCluster = len(cluster_ids)

        for cluster_id, data_ids in cluster_ids.items():
            if not os.path.isfile (f'{plotPath}/Cluster{cluster_id}_puzzle{puzzleNumber}.gif'):
                first_image, frames = gif(desired_puzzle=puzzleNumber,ids=data_ids, attachment=True, includeEgo=not ignore_ego_visualization)
                first_image.save(f'{plotPath}/Cluster{cluster_id}_puzzle{puzzleNumber}.gif', save_all=True, append_images=frames, duration=500, loop=0)
            if not os.path.isfile (f'{plotPath}/Cluster{cluster_id}_puzzle{puzzleNumber}_heatmap.png'):
                Heatmap(cluster_id, data_ids, puzzleNumber,plotPath, ignore_ego=ignore_ego_visualization, log_scale=log_scale)
            if not os.path.isfile (f'{plotPath}/Cluster{cluster_id}_puzzle{puzzleNumber}_softbarycenter.json'):
                softbarycenter(cluster_id, data_ids, puzzleNumber,plotPath)
            
        fig = plt.figure()
        fig.set_figheight(15)
        fig.set_figwidth(20)
        
        ax1 = plt.subplot2grid((3, numCluster), (0, 0), colspan=numCluster)
        ax1.set_title(f'Dendrogram of puzzle {puzzleNumber} solutions', fontsize=20)
        ax1.set_xlabel('Solution ID')
        # ax1.set_ylabel('Distance')
        dendrogram(Z, labels=ids, ax=ax1, leaf_font_size=10 )
        #horizontal line where we cut the dendrogram
        plt.axhline(y=Z[-numCluster+1,2], color='black', linestyle='--')
        
        #pad between dendrogram and heatmap
        plt.subplots_adjust(left=0.05, bottom=0.02, right=0.95, top=0.98, hspace=0.1)

        plt.figtext(0.5, 0.60, "Heatmap and Barycenter of solutions within each cluster", ha="center", va="center", fontsize=20)

        for i in np.arange(1,numCluster+1):
            ax2 = plt.subplot2grid((3, numCluster), (1, i-1))
            ax2.imshow(Image.open(f'{plotPath}/Cluster{i}_puzzle{puzzleNumber}_heatmap.png')) 
            ax2.set_axis_off()
        

        for i in np.arange(1,numCluster+1):
            ax3 = plt.subplot2grid((3, numCluster), (2, i-1))
            try :
                ax3.imshow(Image.open(f'{plotPath}/Cluster{i}_puzzle{puzzleNumber}_softbarycenter.png')) 
                ax3.set_axis_off()
            except:
                
                ax3.text(0.5, 0.5, 'None', ha='center', va='center', fontsize=20)

                ax3.set_axis_off()
            
        plt.savefig(f'{plotPath}/dendrogram_heatmap_barycenter_puzzle{puzzleNumber}.png', dpi=300)
            
        plt.close(fig)
        #print for each puzzle how long it took and with which parameters
        print(f"--- Puzzle {puzzleNumber} ---")
        print("--- %s seconds ---" % (time.time() - start_time)) 
        print(f"preprocessing: {preprocessing}")
        if preprocessing:
            print(f"median total time: {median_total_time}")
            print(f"MAD: {MAD}")
            print(f"Removed {len(ouliers)} outliers time and ids")
            print([i for j, i in enumerate(ids) if j in ouliers])
            print([i for j, i in enumerate(total_time_list) if j in ouliers])
        print(f"Number of clusters: {numCluster}")
        print(f"Softdtw score: {softdtwscore}")
        print(f"Ignore unattached ego: {ignore_Unattached_ego}")
        print(f"Log scale: {log_scale}")
        print(f"Preprocessing: {preprocessing}")
        print(f"Manual number of clusters: {manual_number_of_clusters}")
        print(f"Ignore ego visualization: {ignore_ego_visualization}")

        with open(f'{plotPath}/puzzle{puzzleNumber}_info.txt', 'w') as f:
            print(f"--- Puzzle {puzzleNumber} ---", file=f)
            print("--- %s seconds ---" % (time.time() - start_time), file=f)
            print(f"preprocessing: {preprocessing}", file=f)
            if preprocessing:
                print(f"median total time: {median_total_time}", file=f)
                print(f"MAD: {MAD}", file=f)
                print(f"Removed {len(ouliers)} outliers time and ids", file=f)
                print([i for j, i in enumerate(ids) if j in ouliers], file=f)
                print([i for j, i in enumerate(total_time_list) if j in ouliers], file=f)
            print(f"Number of clusters: {numCluster}", file=f)
            print(f"Softdtw score: {softdtwscore}", file=f)
            print(f"Ignore unattached ego: {ignore_Unattached_ego}", file=f)
            print(f"Log scale: {log_scale}", file=f)
            print(f"Preprocessing: {preprocessing}", file=f)
            print(f"Manual number of clusters: {manual_number_of_clusters}", file=f)
            print(f"Ignore ego visualization: {ignore_ego_visualization}", file=f)

            
def process_puzzle(puzzles,preprocessing):
                 do_cluster(puzzles=[puzzles],
                            preprocessing=preprocessing,
                            state=False,
                            softdtwscore=False,
                            ignore_Unattached_ego=False, 
                            log_scale=True, torch=False,
                            torch_be=False, gamma=1,
                            manual_number_of_clusters=False, 
                            ignore_ego_visualization=True)
        # test the positional vector function
        # with open('./Data/Pilot3/Frames/2022-10-27-080305_31_1_1_0_frames.json') as json_file:
        #     data = json.load(json_file)

        # vector, object_names = positional_vector(data, concat_state=True, weighted=True)
        # print(vector)  
    
     
if __name__ == '__main__':
    
    puzzles = [2,3,4,5,6,21,22,23,24,25,26]  # List of puzzles
    preprocessing_options = [ False, False]  # Preprocessing options
    
    # Create a list of arguments for each combination of puzzle and preprocessing option
    arguments = [(puzzle, preprocessing) for puzzle in puzzles for preprocessing in preprocessing_options]
    
    # Create a multiprocessing pool with the number of processes you want to use
    pool = multiprocessing.Pool(processes=10)  # Adjust the number of processes as needed
    
    # Use the pool to map the process_puzzle function to the list of arguments
    pool.starmap(process_puzzle, arguments)
    
    # Close the pool to free up resources
    pool.close()
    pool.join()
     

        
#     plt.figure()
#     plt.plot(range(3, 10), neg_value_fraction, label='Neg' )
#     plt.plot(range(3, 10), below_avg_fraction, label='Below average ')
#     plt.plot(range(3, 10), neg_value_fractionP, label='Neg preprocessed with gamma=0.1')
#     plt.plot(range(3, 10), below_avg_fractionP, label='Below average preprocessed with gamma=0.1')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Fraction')
#     plt.legend()
#     plt.title(f"Negative and below average fraction for puzzle {puzzle}")
#     plt.savefig(f'./Plots_Text/clustering/silhouette_fraction_puzzle{puzzle}.png', dpi=300)

repo_path = './'

os.chdir(repo_path)

subprocess.run(['git', 'add', '.'])

subprocess.run(['git', 'commit', '-m', "add present_objects_puzzle file"])

subprocess.run(['git', 'push'])
