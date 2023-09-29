import numpy as np
import os
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import time

start_time = time.time()

def coloring(type,c):
    if type=='box1':
        return (0,0,1,c)
    elif type=='box2':
        return (0,1,0,c)
    elif type=='obj1':
        return (1,0,0,c)
    elif type=='obj2':
        return (1,0,1,c)
    elif type=='obj3':
        return (1,1,0,c)
    elif type=='obj4':
        return (0,1,1,c)
    elif type=='ego':
        return (0,0,0,c)
    
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

def main():
 for pilot in [3,4]:
        folder = './Data/Pilot{}/Frames'.format(pilot)

        for filename in os.listdir(folder):
            if filename.endswith('.json'):
                participant_id, run, puzzle, attempt = use_regex(filename)
 
            if participant_id==40 :

                with open(os.path.join(folder,filename)) as json_file:
                            
                            data = json.load(json_file)
                            vector, objects_names = positional_vector(data)
                            
                            total_objects = len(objects_names)
                            # print(objects_names)
                            # print(total_objects)
                            # cm_list= ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges','Reds', 'PuRd']
                          
                            fig, ax = plt.subplots()
                            for i in range(total_objects):

                                object = list(objects_names)[i]
                                if True:
                                 
        
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
                                        colors=[coloring(t,j) for j in np.linspace(0.2,1,len(xi))]
                                        #diffrent colors for each type
                                        # colors=[coloring(type,i) for i in np.linspace(0.2,1,len(xi))]
                                        # cm=mcolors.LinearSegmentedColormap.from_list('mylist', colors, N=len(xi))
                                        c=np.arange(0,len(xi))
                                        ax.scatter(xi, yi, alpha=0.1,label=objects_names[object], c = colors, s=15,
                                                marker= ".")
                                        
                                        plt.xlim(-2, 2)
                                        plt.ylim(-2, 2)
                                        plt.xlabel('x')
                                        plt.ylabel('y')
                                        plt.legend()
                                        title= f'Participant: {participant_id}, Run: {run}, Puzzle: {puzzle}, Attempt: {attempt}'.format(participant_id, run, puzzle, attempt)
                                        plt.title(title, fontsize=14)
                                        # plt.show()
                plt.savefig('./Temp_Frame_Path_Plots/'+
                                str(participant_id)+'_'+ str(run)+'_'+str(puzzle)+'_'+str(attempt)+'.png', dpi=300)
                plt.close()
                                
if __name__ == "__main__":
    main()
    print("--- %s seconds ---" % (time.time() - start_time))