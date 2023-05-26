import numpy as np
import os
import json
import movementTracker
import re
import HMPlotter
from dtaidistance import dtw
from dtaidistance import dtw_ndim
from dtaidistance import dtw_ndim_visualisation

    
def main():
     # folder = input("Enter the folder path: ")
    folder = '/home/erfan/Downloads/pnp'
   
    # 1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26
    for desired_puzzle in [2]:

        output_file = os.path.join('/home/erfan/Documents/Puzzle/puzzlesdata/Plots_Text/Direction_Text', 'puzzle'+str(desired_puzzle)+'20_pos.txt')

        with open(output_file, 'w') as f:
            for filename in sorted(os.listdir(folder)):
                if filename.endswith('.json'):
                    participant_id, run, puzzle, attempt = HMPlotter.use_regex(filename)

                    if desired_puzzle == puzzle :

                        f.write(str(participant_id) + "_" + str(run) +
                                "_" + str(puzzle) + "_" +str(attempt)+"\n" )
                        

                        eventlist=np.array([])
                        with open(os.path.join(folder, filename)) as json_file:

                            data = json.load(json_file)
                            df=movementTracker.df_from_json(data) 

                        for tp in ['box1', 'box2', 'obj1', 'obj2', 'obj3', 'obj4','Glue','Unglue','free']:

                            # xi, yi = movementTracker.interaction(df, participant_id, run, tp)
                            # if xi.size == 0 or yi.size == 0:

                            #     pass

                            # else:

                            o=movementTracker.interaction(df, participant_id, run, tp,direction=True, pos=True)
                            eventlist= np.append(eventlist,o)
                    
                        #sortting the list baded on the time of the event
                        eventlist=eventlist[np.argsort(eventlist)]
                        eventlist=[i.split("_")[1] for i in eventlist]

                        if len(eventlist)!=0:
                            f.write(str(eventlist)+"\n")

                        # solved_stats=movementTracker.interaction(df, participant_id, run,type="total",solved=True)
                        # f.write("Solved: " + str(solved_stats)+"\n")  
                        f.write("\n")
   
    with open(output_file, 'r') as f:
        
        text = f.readlines()
        max_length = 0
        ids=[]


        for line in text:
            if line.startswith("["):
                action_list = actions(line)
                naction=len(action_list)
                if max_length < naction:
                    max_length = naction


        dim=0
        for line in text:
            if line.startswith("["):
                dim+=1
        #define a 3D array to store the data with NAn
        p210=np.empty((dim,max_length,4))
        p210[:] = np.nan
        
        j=0
        for line in text:

            if line.startswith("["):
                action_list = actions(line)
                for i in range(len(action_list)):
                    action_list[i] = action(action_list[i])
              
                ac = np.array(action_list)
                p210[j,:ac.shape[0],:ac.shape[1]]=ac
                j+=1
            elif line.startswith("\n"):
                pass
            else:
                line=line.replace("\n","")
                ids.append(line)
        # print(ids)
             

    # print(p210.shape)
    # define a distance matrix
    series = []
    for i in range(p210.shape[0]):
        seri = p210[i,:,:]
        seri = seri[~np.isnan(seri).any(axis=1)]
        series.append(seri)
    # print(len(series))

    ds = dtw.distance_matrix_fast(series,window=3)
    # print(ds)
    return ds,ids


    # dist=np.empty((p210.shape[0],p210.shape[0]))
    # dist[:] = np.nan

    # for i in range(p210.shape[0]):
    #     for j in range(p210.shape[0]):
    #         querry = p210[i,:,:]
    #         reference = p210[j,:,:]
       
    #         querry = querry[~np.isnan(querry).any(axis=1)]
    #         reference = reference[~np.isnan(reference).any(axis=1)]
    #         # print(querry)
    #         # print(reference)
    #         d=dtw_ndim.distance(querry, reference)
    #         # print(d)
    #         dist[i,j]=d
    # print(dist.shape)





   


                

def actions(action_list):
    #convert action_list from string to list
    action_list = action_list.replace("[", "")
    action_list = action_list.replace("]", "")
    action_list = action_list.replace(" '", "")
    action_list = action_list.replace("'", "")
    action_list = action_list.replace("s", "")
    action_list = action_list.replace("\n", "")
    al = action_list.split(",")
    return al

def action(string):
    #convert action from string to list
    s = string.split(" ")
    if s [0] == 'free':
        s [0] = 0
    elif s [0] == 'box1':
        s [0] = 1
    s [1] = float(s[1])
    s [2] = float(s[2])
    
    s [3] = 0
    return s
       


if __name__ == "__main__":
    main()

                        
