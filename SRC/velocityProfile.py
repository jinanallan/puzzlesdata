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

def velocity_profile(data, acceleration=False):
    """
    plot the velocity or accelaration profile of the objects in the puzzle

    Accepts:
        data: the json file (frames)
        acceleration: if True, plots the acceleration profile, otherwise plots the velocity profile
    Reurns: 
        velocity profile of the objects
    """
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

    velocity_vector = positional_vector.diff()
    velocity_vector = velocity_vector.drop(0)
    velocity_vector = velocity_vector.reset_index(drop=True)
    if acceleration:
        acceleration_vector = velocity_vector.diff()
        acceleration_vector = acceleration_vector.drop(0)
        acceleration_vector = acceleration_vector.reset_index(drop=True)
        velocity_vector = acceleration_vector
    v=np.zeros((len(velocity_vector),len(present_objects)))

    for i in range(len(present_objects)):
        vx = velocity_vector[positional_vector.columns[i*2][0],'x']
        vx=np.array(vx, dtype=np.float64)
        vy = velocity_vector[positional_vector.columns[i*2][0],'y']
        vy=np.array(vy, dtype=np.float64)
        v_temp=np.sqrt(vx**2+vy**2)
        v[:,i]=v_temp


    fig, ax = plt.subplots(len(present_objects),1,figsize=(10,3*len(present_objects)+4))

    for i in range(len(present_objects)):
        
     
        nl=0
        vmax=v.max()
        ax[i].plot(v[:,i], label=present_objects[positional_vector.columns[i*2][0]], color="C"+str(i))
        #plot vertical lines where v[:,i] is equal to v[:,0]
        if i != 0:
            same_as_ego = np.where(v[:,i] == v[:,0])
            vline = v[:,i][same_as_ego]
            same_as_ego = np.delete(same_as_ego, np.where(vline == 0))


            for index in np.arange(0, len(v)):
                if index in same_as_ego:

                    if nl==0: 
                        ax[0].axvline(x=index, color="C"+str(i), linestyle='--', alpha=0.1, label=present_objects[positional_vector.columns[i*2][0]]+
                                  " match")
                        nl+=1
                    else:
                        ax[0].axvline(x=index, color="C"+str(i), linestyle='--', alpha=0.1)
                        
                    ax[0].legend()
            # for j in range(len(same_as_ego)):
            #     ax[0].axvline(x=same_as_ego[j], color=color, linestyle='--')

        ax[i].set_title(present_objects[positional_vector.columns[i*2][0]]+" velocity profile")
        if acceleration:
            ax[i].set_title(present_objects[positional_vector.columns[i*2][0]]+" acceleration profile")
        ax[i].set_xlabel("time step [s]")
        ax[i].set_ylabel("velocity magnitude [1/s]")

        if acceleration:
            ax[i].set_ylabel("acceleration magnitude [1/s^2]")
        ax[i].set_xlim(0, len(v))  

        if T < 10:
            ax[i].set_xticks(np.arange(0, len(v), step=round(len(v)/T)*1), np.arange(0, T, step=1))
        else:
            ax[i].set_xticks(np.arange(0, len(v), step=round(len(v)/T)*10), np.arange(0, T, step=10))
        ax[i].set_ylim(0, 1.1*vmax)
        ax[i].legend(loc='upper right')

        fig.tight_layout(pad=5.0)

    return fig, ax


frame_folder= "./Data/Pilot4/Frames/"
frame_files = os.listdir(frame_folder)

for file in frame_files:
    if file.endswith(".json"):
        participant_id, run, puzzle, attempt = use_regex(file)
        if participant_id == 59 and run ==1 and puzzle == 26 and attempt == 0:
            with open(os.path.join(frame_folder,file)) as json_file:
                if not os.path.exists("./Plots_Text/Velocity_Profile/"+str(participant_id)+"_"+str(run)+"_"+str(puzzle)+"_"+str(attempt)+".png"):
                    # print("Saved: ", str(participant_id)+"_"+str(run)+"_"+str(puzzle)+"_"+str(attempt)+".png")
                    data = json.load(json_file)
                    fig, ax = velocity_profile(data)
                    figa, axa = velocity_profile(data, acceleration=True)
                #set the title of the plot
                    fig.suptitle("Participant: "+str(participant_id)+" Run: "+str(run)+" Puzzle: "+str(puzzle)+" Attempt: "+str(attempt))
                    figa.suptitle("Participant: "+str(participant_id)+" Run: "+str(run)+" Puzzle: "+str(puzzle)+" Attempt: "+str(attempt))
                    fig.savefig("./Plots_Text/Velocity_Profile/"+str(participant_id)+"_"+str(run)+"_"+str(puzzle)+"_"+str(attempt)+".png", dpi=300)
                    figa.savefig("./Plots_Text/Velocity_Profile/Acceleration/"+str(participant_id)+"_"+str(run)+"_"+str(puzzle)+"_"+str(attempt)+".png", dpi=300)
                    plt.close(fig)
                    plt.close(figa)
