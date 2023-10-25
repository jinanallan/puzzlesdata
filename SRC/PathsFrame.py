import numpy as np
import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import time
import movementTracker
import json
import subprocess
start_time = time.time()
def coloring(object):
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
    # print(present_objects)

    universal_Objects = ["box1","box2", "obj1","obj2", "obj3","obj4","ego","obj1_a"]
    
    for x in list(present_objects):
        if present_objects[x] not in universal_Objects:
            # print(present_objects[x])
            present_objects.pop(x)
    # print(present_objects)
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

def main():
 for pilot in [3,4]:
        folder = './Data/Pilot{}/Frames'.format(pilot)
        ego_folder = './Data/Pilot{}/Ego-based'.format(pilot)

        n = len(os.listdir(folder))
        filecounter = 0
        for filename in os.listdir(folder):
        
            ego_filename = filename[:-12] + '.json'
            # print(filename)
            # print(ego_filename)
            ego_filename=os.path.join(ego_folder, ego_filename)
            # print(ego_filename)
            ego_file = json.load(open(ego_filename))
            try:
                df = pd.DataFrame(ego_file)
            except:
                 df = pd.DataFrame(ego_file, index=[0])
            # print(df)
            print(f'{filecounter}/{n}')
            filecounter += 1
            if filename.endswith('.json'):
                participant_id, run, puzzle, attempt = use_regex(filename)
            #check if the file is already plotted
            
            
            solved , total_time = movementTracker.interaction(df, participant_id, run, "free", solved=True)
            # print(solved, total_time)
            if not os.path.isfile('./Plots_Text/Path_Plots/frameBased/includeEgo'+ str(participant_id)+'_'+ str(run)+'_'+str(puzzle)+'_'+str(attempt)+'.png') and puzzle in [1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26]:

                with open(os.path.join(folder,filename)) as json_file:
                            
                        data = json.load(json_file)
                        vector, objects_names = positional_vector(data)
                        total_objects_len = len(objects_names)

                        # velocity_vector = vector.diff()
                        # velocity_vector = velocity_vector.drop(0)
                        # velocity_vector = velocity_vector.reset_index(drop=True)
                        # #drop ego 
                        

                        # v=np.zeros((len(velocity_vector),(total_objects_len-1)*2))

                        # for i in range((total_objects_len-1)):
                        #     vx = velocity_vector[vector.columns[i*2][0],'x']
                        #     vx=np.array(vx, dtype=np.float64)
                        #     vy = velocity_vector[vector.columns[i*2][0],'y']
                        #     vy=np.array(vy, dtype=np.float64)
                        #     # v_temp=np.sqrt(vx**2+vy**2)
                        #     v[:,i*2]=vx
                        #     v[:,i*2+1]=vy
                        
                        #     # Initialize a dictionary to store which parts of the velocity vectors are equal
                        #     equal_velocity_parts = {}

                        #     for i in range((total_objects_len-1)):
                        #         for j in range(i + 1, (total_objects_len-1)):
                        #             vx1 = v[:, i * 2]  # vx values for object i
                        #             vy1 = v[:, i * 2 + 1]  # vy values for object i
                        #             vx2 = v[:, j * 2]  # vx values for object j
                        #             vy2 = v[:, j * 2 + 1]  # vy values for object j

                        #             # Compare the vx components of object i and object j
                        #             equal_vx = np.where(vx1 == vx2)[0]
                                    
                        #             # Compare the vy components of object i and object j
                        #             equal_vy = np.where(vy1 == vy2)[0]

                        #             # Store the equal components in the dictionary
                        #             if i not in equal_velocity_parts:
                        #                 equal_velocity_parts[i] = {"equal_vx": [], "equal_vy": []}
                        #             if j not in equal_velocity_parts:
                        #                 equal_velocity_parts[j] = {"equal_vx": [], "equal_vy": []}

                        #             equal_velocity_parts[i]["equal_vx"].extend(equal_vx)
                        #             equal_velocity_parts[j]["equal_vx"].extend(equal_vx)

                        #             equal_velocity_parts[i]["equal_vy"].extend(equal_vy)
                        #             equal_velocity_parts[j]["equal_vy"].extend(equal_vy)

                        #     # 'equal_velocity_parts' now contains the parts of the velocity vectors that are equal between objects.
                        #     # The keys are the indices of the objects, and the values are dictionaries containing equal vx and vy components.
                        #     print(equal_velocity_parts.keys())

                                
                        fig, ax = plt.subplots()
                        for i in range(total_objects_len):

                            object = list(objects_names)[i]
                            if True: #objects_names[object] != 'ego' in case ignoring ego
                                
    
                                xi = vector[object]['x']
                                yi = vector[object]['y']
                                xi = np.array(xi)
                                yi = np.array(yi)
                                # print(xi.size, yi.size)
                                # print(xi, yi)


                                imgfolder = './cropped_puzzles_screenshots'
                                fname = os.path.join(imgfolder, 'puzzle'+str(puzzle)+'.png')
                                img = Image.open(fname).convert('L')

                                img = ax.imshow(img, extent=[-2, 2, -2, 2], cmap='gray')

                                if xi.size == 0 or yi.size == 0:
                                    continue
                                else:
                                    t = objects_names[object]
                                    
                                    #diffrent colors for each type
                                    # colors=[coloring(type,i) for i in np.linspace(0.2,1,len(xi))]
                                    # cm=mcolors.LinearSegmentedColormap.from_list('mylist', colors, N=len(xi))
                                    c=np.arange(0,len(xi))
                                    #map alpha from first to last point in xi
                                    #increase alpha from first to last point in xi

                                    ax.scatter(xi, yi,alpha=0.1, color= coloring(objects_names[object]), s=10, edgecolors='face',
                                            marker= ".")
                                    sc = plt.scatter([],[], color=coloring(objects_names[object]), label=objects_names[object])
                                    
                                    plt.xlim(-2, 2)
                                    plt.ylim(-2, 2)
                                    plt.xlabel('x')
                                    plt.ylabel('y')
                                    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
                                    title= f'Participant: {participant_id}, Run: {run}, Puzzle: {puzzle}, Attempt: {attempt}, \n Solved: {solved}, Total time: {total_time} s'.format(participant_id, run, puzzle, attempt, solved, total_time)
                                    plt.title(title, fontsize=14)
                                    # plt.show()
                plt.savefig('./Plots_Text/Path_Plots/frameBased/includeEgo/'+
                                str(participant_id)+'_'+ str(run)+'_'+str(puzzle)+'_'+str(attempt)+'.png', dpi=300)
                plt.close()
                                
if __name__ == "__main__":
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

    repo_path = './'

os.chdir(repo_path)

subprocess.run(['git', 'add', '.'])

subprocess.run(['git', 'commit', '-m', "frame based path plots with ego added "])

subprocess.run(['git', 'push'])