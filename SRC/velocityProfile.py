import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os 
import json

def use_regex(input_text):
    pattern = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_frames", re.IGNORECASE)

    match = pattern.match(input_text)
    
    particpants = match.group(3)
    run = match.group(4)
    puzzle_id = match.group(5)
    attempt = match.group(6)
    return int(particpants), int(run), int(puzzle_id), int(attempt)

def velocity_profile(data):
    """
    plot the velocity profile of the objects in the puzzle

    Accepts:
        data: the json file
    Reurns: 
        velocity profile of the objects
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

    fig, ax = plt.subplots(len(present_objects),1,figsize=(10,5*len(present_objects)))
    velocity_vector = positional_vector.diff()
    velocity_vector = velocity_vector.drop(0)
    velocity_vector = velocity_vector.reset_index(drop=True)

    vmax=0

    for i in range(len(present_objects)):
        vx = velocity_vector[positional_vector.columns[i*2][0],'x']
        vx=np.array(vx, dtype=np.float64)
        vy = velocity_vector[positional_vector.columns[i*2][0],'y']
        vy=np.array(vy, dtype=np.float64)
        v=np.sqrt(vx**2+vy**2)
        if v.max()>vmax:
            vmax=v.max()
        ax[i].plot(v, label=present_objects[positional_vector.columns[i*2][0]])
        ax[i].set_title(present_objects[positional_vector.columns[i*2][0]]+" velocity profile")
        ax[i].set_xlabel("time step")
        ax[i].set_ylabel("velocity")
        ax[i].set_ylim(0, 1.1*vmax)
        ax[i].legend()
    return fig, ax

frame_folder= "./Data/Pilot3/Frames/"
frame_files = os.listdir(frame_folder)
puzzleNumber = 21


for file in frame_files:
    if file.endswith(".json"):
        participant_id, run, puzzle, attempt = use_regex(file)
        if puzzle == puzzleNumber and attempt == 0 and run == 2 and participant_id == 35:
            with open(os.path.join(frame_folder,file)) as json_file:
                data = json.load(json_file)
                fig, ax = velocity_profile(data)
                #set the title of the plot
                fig.suptitle("Participant: "+str(participant_id)+" Run: "+str(run)+" Puzzle: "+str(puzzle)+" Attempt: "+str(attempt))
                fig.savefig("test.png")
