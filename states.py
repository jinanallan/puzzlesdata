import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os 
import json
from tslearn import metrics
from sklearn.metrics import pairwise_distances
from matplotlib.colors import LinearSegmentedColormap

def use_regex(input_text):
    pattern = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_frames", re.IGNORECASE)

    match = pattern.match(input_text)
    
    particpants = match.group(3)
    run = match.group(4)
    puzzle_id = match.group(5)
    attempt = match.group(6)
    return int(particpants), int(run), int(puzzle_id), int(attempt)

def states_vector(data): 
    data = pd.DataFrame(data)

    end_time = data.timestamp[len(data.frames)-2].split("-")[0]
    end_time = int(end_time)
    end_time = pd.to_datetime(end_time, unit='us')
    
    start_time = data.timestamp[0].split("-")[0]
    start_time = int(start_time)
    start_time = pd.to_datetime(start_time, unit='us')
    
    T = end_time - start_time
    T = T.total_seconds()
    # print(T)

    last_frame = data.frames[len(data.frames)-1]
    present_objects = {}
    for definition in last_frame:
        present_objects[definition["ID"]] = definition["name"]

    universal_Objects = ["box1","box2", "obj1","obj2", "obj3","obj4","ego", "obj1_a"]
    
    for x in list(present_objects):
        if present_objects[x] not in universal_Objects:
            # print(present_objects[x])
            present_objects.pop(x)
    
    positional_vector=pd.DataFrame(columns=present_objects)
    sub_columns = pd.MultiIndex.from_product([positional_vector.columns, ['x', 'y']], names=['ID', 'position'])
    positional_vector = pd.DataFrame(index=range(len(data.frames)), columns=sub_columns)
    # print(present_objects)
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
        # print (i, object_i)
        object_i_name = present_objects[object_i]
        vx_i = velocity_vector[positional_vector.columns[i*2][0],'x']
        vx_i=np.array(vx_i, dtype=np.float64)
        vy_i = velocity_vector[positional_vector.columns[i*2][0],'y']
        vy_i=np.array(vy_i, dtype=np.float64)
        v_temp=np.sqrt(vx_i**2 + vy_i**2)
        v[:,i]=v_temp

    states = np.zeros((len(velocity_vector),len(present_objects)-1))
    # print(states.shape) 

    for i, object_i in enumerate(present_objects):
        object_i_name = present_objects[object_i]
        # print(i, object_i)

        if i != 0:
            same_as_ego = np.where(v[:,i] == v[:,0])
            vline = v[:,i][same_as_ego]
            same_as_ego = np.delete(same_as_ego, np.where(vline == 0))
            states[same_as_ego,i-1] = 1
    return states
        
frame_folder= "./Data/Pilot4/Frames/"
frame_files = os.listdir(frame_folder)

for file in frame_files:
    if file.endswith(".json"):
        participant_id, run, puzzle, attempt = use_regex(file)
        if participant_id == 59 and run ==1 and puzzle == 26 and attempt == 0:
            with open(os.path.join(frame_folder,file)) as json_file:
                    data = json.load(json_file)
                    states1=states_vector(data)
        elif participant_id == 59 and run ==2 and puzzle == 26 and attempt == 0:
            with open(os.path.join(frame_folder,file)) as json_file:
                    data = json.load(json_file)
                    states2=states_vector(data)


# distance = metrics.cdist_soft_dtw_normalized(states1, states2, gamma=1., metric="hamming")
# print(distance)
path, sim = metrics.dtw_path_from_metric(states1, states2, metric="hamming")
distance = pairwise_distances(states1, states2, metric="hamming")   
print(sim)


left, bottom = 0.01, 0.1
w_ts = h_ts = 0.2
left_h = left + w_ts + 0.02
width = height = 0.65
bottom_h = bottom + height + 0.02

rect_s_y = [left, bottom, w_ts, height]
rect_dist = [left_h, bottom, width, height]
rect_s_x = [left_h, bottom_h, width, h_ts]

plt.figure(2, figsize=(6, 6))
ax_dist = plt.axes(rect_dist)
ax_s_x = plt.axes(rect_s_x)
ax_s_y = plt.axes(rect_s_y)

ax_dist.imshow(distance, origin='lower')
ax_dist.axis("off")
ax_dist.autoscale(False)
ax_dist.plot(*zip(*path), "w-", linewidth=3.)

colors = [(1, 1, 1), (0, 0, 1)]  # White -> Blue
cmap_name = 'white_blue'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=2)
ax_s_x.imshow(states1.T, aspect="auto", cmap=cm)
ax_s_x.axis("off")

ax_s_y.imshow(np.flip(states2, axis=1), aspect="auto", cmap=cm)
ax_s_y.axis("off")

plt.tight_layout()
plt.show()
