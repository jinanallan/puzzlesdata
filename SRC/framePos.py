import json
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt

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

for file in frame_files:
    if file.endswith(".json"):
        participant_id, run, puzzle, attempt = use_regex(file)
        if participant_id == 32 and run == 1 and puzzle == 5 and attempt == 0:
            with open(os.path.join(frame_folder,file)) as json_file:
                data = json.load(json_file)
                vector, object_names = positional_vector(data)
                print(vector)
                print(object_names)

# TODO: define objects and their IDs present in the last frame **DONE
# store the X,Y,Z, rotation of each object in a list **Done
#study the velocity profile and match with the intraction


