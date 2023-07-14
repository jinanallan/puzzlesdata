import json
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt


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
        if participant_id == 32 and run == 1 and puzzle == 25 and attempt == 0:
            with open(os.path.join(frame_folder,file)) as json_file:
                data = json.load(json_file)
                data = pd.DataFrame(data)
                # print(file)
                # print(data.head())

            # last_frame = data.frames[len(data.frames)-1]
            # print("these IDs are in the last frame")
            # for definition in last_frame:
            #     print(definition)
            #     print(definition["ID"]," ",definition["name"])

            # frame10 = data.frames[10]
            # date= data.timestamp[10]
            # print(date)
            # # print(type(frame10))
            Xe=[]
            Xb=[]
            for frame in data.frames:
                for definition in frame:
                    if definition["ID"] ==13:
                        Xe.append(definition["X"])
                    if definition["ID"] == 14:
                        Xb.append(definition["X"])



Xe = np.array(Xe)
Xb = np.array(Xb)
#get the first two columns of Xe
Xe = Xe[:,0:2]
Xb = Xb[:,0:2]
Xb=Xb.T
Xbd=np.diff(Xb)
Xe=Xe.T
Xed=np.diff(Xe)
ve = np.linalg.norm(Xed,axis=0)
vb = np.linalg.norm(Xbd,axis=0)
plt.figure()
plt.subplot(211)
plt.plot(vb)
plt.subplot(212)
plt.plot(ve)
#when the velocity is zero, the box is not moving


plt.figure()
plt.gca().set_aspect('equal')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.title("ego and box trajectory")
plt.plot(Xe[0,:],Xe[1,:], label="ego")
plt.plot(Xb[0,:],Xb[1,:], label="box")
plt.legend()
plt.show()

# TODO: define objects and their IDs present in the last frame
# store the X,Y,Z, rotation of each object in a list
#study the velocity profile and match with the intraction


