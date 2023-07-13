import json
import pandas as pd
import os

frame_folder= "./Data/Frames/"
frame_files = os.listdir(frame_folder)

i=0
for file in frame_files:
    if file.endswith(".json"):
        with open(os.path.join(frame_folder,file)) as json_file:
            data = json.load(json_file)
            data = pd.DataFrame(data)
            print(file)
            # print(data.head())

        last_frame = data.frames[len(data.frames)-1]
        for item in last_frame:
            print(item.keys())
        i+=1
        if i==1:
            break


