import numpy as np
import os
import json
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import movementTracker
import HMPlotter
from PIL import Image
    
def main():
     # folder = input("Enter the folder path: ")
    folder = '/home/erfan/Downloads/pnp'
   
    # 1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26
    for desired_puzzle in [1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26]:

        output_file = os.path.join('/home/erfan/Documents/Puzzle/puzzlesdata/Plots_Text/Direction_Text', 'puzzle'+str(desired_puzzle)+'.txt')

        with open(output_file, 'w') as f:
            for filename in sorted(os.listdir(folder)):
                if filename.endswith('.json'):
                    participant_id, run, puzzle, attempt = HMPlotter.use_regex(filename)

                    if desired_puzzle == puzzle:

                        f.write("Participant:" + str(participant_id) + " for the Puzzle:" + str(puzzle) +
                                " in the Attempt:" + str(attempt) + " and run:" +str(run)+ " took follwing movements:"+"\n" )
                        

                        eventlist=np.array([])
                        with open(os.path.join(folder, filename)) as json_file:

                            data = json.load(json_file)
                            df=movementTracker.df_from_json(data) 

                        for type in ['box1', 'box2', 'obj1', 'obj2', 'obj3', 'obj4','Glue','Unglue']:

                            # xi, yi = movementTracker.interaction(df, participant_id, run, type)
                            # if xi.size == 0 or yi.size == 0:

                            #     pass

                            # else:

                            o=movementTracker.interaction(df, participant_id, run, type,direction=True)
                            eventlist= np.append(eventlist,o)
                    
                        #sortting the list baded on the time of the event
                        eventlist=eventlist[np.argsort(eventlist)]
                        eventlist=[i.split("_")[1] for i in eventlist]

                        if len(eventlist)!=0:
                            f.write(str(eventlist)+"\n")

                        solved_stats=movementTracker.interaction(df, participant_id, run,type="total",solved=True)
                        f.write("Solved: " + str(solved_stats)+"\n")  
                        f.write("\n")

if __name__ == "__main__":
    main()

                        
