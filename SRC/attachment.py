import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import json

def use_regex(input_text):
    pattern = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_frames", re.IGNORECASE)

    match = pattern.match(input_text)
    
    particpants = match.group(3)
    run = match.group(4)
    puzzle_id = match.group(5)
    attempt = match.group(6)
    return int(particpants), int(run), int(puzzle_id), int(attempt)
def positional_vector(data, ignore_ego=False):
    """
    Get the positional vector of the objects from frames json file

    Accepts:
        data: the json file
    Reurns: 
        positional_vector: dataframe with the positional vector
        present_objects: dict of object names and their IDs
    """
    data = pd.DataFrame(data)

    last_frame = data.frames[len(data.frames)-1]
    present_objects = {}
    for definition in last_frame:
        present_objects[definition["ID"]] = definition["name"]

    universal_Objects = ["box1","box2", "obj1","obj2", "obj3","obj4","ego","obj1_a"]
    if ignore_ego:
        universal_Objects.remove("ego")
    
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
def attachment_plot(positional_vector, present_objects):
    """
    Plot the attachment of objects to the during the solution in form of: whether the object is moving or not 
    and if moving at the same time as another object, then they are attached (valid for pick and place puzzles)

    Accepts:
        positional_vector: dataframe with the positional vector
        present_objects: dict of object names and their IDs
    Reurns:
        Plot of the attachment of objects to the during the solution
    """
    T = len(positional_vector.index)
    fig, ax = plt.subplots(figsize=(10,10))
    
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
        for t in np.arange(0,T-1):
                if v_temp[t]>0:
                    plt.scatter(t, i, color=coloring(present_objects[object_i], True), s=50, marker='s')

    for t in np.arange(0,T-1):
        for i, object_i in enumerate(present_objects):
            for j in range(i+1,len(present_objects)):
                object_j = list(present_objects.keys())[j]
                if v[t,i]>0 and v[t,j]>0:
                    plt.scatter(t, j, color="black", s=20, marker='*')
                    plt.scatter(t, i, color="black", s=20, marker='*')
    plt.xlabel('time steps [s]')
    plt.yticks(np.arange(len(present_objects)), present_objects.values())
    #each time step is 0.01s
    plt.xticks(np.arange(0,T, 1000), np.arange(0,T/100, 10))
    plt.savefig('./Plots_Text/attachment_plot_ex.png', dpi=300)
    plt.show()
                  
                    
    #     for j in range(i+1,len(present_objects)):
    #         object_j = list(present_objects.keys())[j]
    #         print (j, object_j)
    #         object_j_name = present_objects[object_j]
    #         vx_j = velocity_vector[positional_vector.columns[j*2][0],'x']
    #         vx_j=np.array(vx_j, dtype=np.float64)
    #         vy_j = velocity_vector[positional_vector.columns[j*2][0],'y']
    #         vy_j=np.array(vy_j, dtype=np.float64)

    #         for t in np.arange(0,T-1):
    #             norm = np.sqrt((vx_i[t]-vx_j[t])**2 + (vy_i[t]-vy_j[t])**2)
    #             if norm < 0.1:
    #                 # print("Objects {} and {} are attached at time {}".format(object_i_name, object_j_name, t))
    #                 plt.scatter(t, i, color=coloring(present_objects[object_i], True), s=50, marker='s')
    #                 plt.scatter(t, j, color=coloring(present_objects[object_j], True), s=50, marker='s')
    # plt.xlabel('time steps')
    # plt.yticks(np.arange(len(present_objects)), present_objects.values())
    # plt.show()
frame_folders = ["./Data/Pilot3/Frames/", "./Data/Pilot4/Frames/"]

for frame_folder in frame_folders:
            frame_files = os.listdir(frame_folder)
            for file in frame_files:
                if file.endswith(".json"):
                    participant_id, run, puzzle, attempt = use_regex(file)
                    if puzzle == 24 and run == 2 and attempt == 0 and participant_id == 33:
                        # print(file)
                        with open(frame_folder+file) as f:
                            data = json.load(f)
                            positional_vector, present_objects = positional_vector(data, ignore_ego=True)
                            attachment_plot(positional_vector, present_objects)