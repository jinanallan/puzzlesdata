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
   

    for desired_puzzle in [1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26]:

        output_file = os.path.join('Direction', 'puzzle'+str(desired_puzzle)+'.txt')

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

                        for type in ['box1', 'box2', 'obj1', 'obj2', 'obj3', 'obj4']:

                            xi, yi = movementTracker.interaction(df, participant_id, run, type)
                            if xi.size == 0 or yi.size == 0:

                                pass

                            else:

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




            
               
                
        












            #         if xi.size == 0 or yi.size == 0:
            #                     pass
            #         else:
                        
            #             imgfolder = 'cropped'
            #             fname = os.path.join(imgfolder, 'puzzle'+str(puzzle)+'.png')
            #             img = Image.open(fname).convert('L')

            #             img = ax.imshow(img, extent=[-2, 2, -2, 2], cmap='gray')

            #             #diffrent colors for each type
            #             colors=[coloring(type,i) for i in np.linspace(0.2,1,len(xi))]
            #             # cm=mcolors.LinearSegmentedColormap.from_list('mylist', colors, N=len(xi))

            #             if type=='free':
            #                  s=5
            #             else:
            #                 s=25
            #             ax.scatter(xi, yi,
            #                                    s=s,
            #                                    c=colors,
            #                                    label=type,
            #                        marker= ".")
                        
            #             plt.xlim(-2, 2)
            #             plt.ylim(-2, 2)
            #             plt.xlabel('x')
            #             plt.ylabel('y')
            #             plt.legend()
            # plt.title('Participant:'+str(participant_id)+' Puzzle: '+str(puzzle)+' Attempt:'+str(attempt)+' Run:'+str(run)+'\n'+'sparcity:'+str(sparce))
            # plt.savefig('plots/'+str(puzzle)+'_'+str(participant_id)+'_'+str(attempt)+'_'+str(run)+'.png', dpi=300)
            # plt.close()

if __name__ == "__main__":
    main()

                        
