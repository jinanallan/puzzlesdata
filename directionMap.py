import numpy as np
import os
import json
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import movementTracker
import HMPlotter
from PIL import Image

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
    elif type=='free':
        return (0,0,0,c)
    
def main():
     # folder = input("Enter the folder path: ")
    folder = '/home/erfan/Downloads/pnp'
    # desiredpuzzle = int(input("Enter the puzzle number: "))

    output_file = 'directionmap.txt'

    with open(output_file, 'w') as f:
        for filename in os.listdir(folder):
            if filename.endswith('.json'):
                participant_id, run, puzzle, attempt = HMPlotter.use_regex(filename)
            if puzzle in [1, 2]:

                f.write("Participant:" + str(participant_id) + " for the Puzzle:" + str(puzzle) +
                         " in the Attempt:" + str(attempt) + " and run:" +str(run)+ " took follwing movements:"+"\n" )
                


                with open(os.path.join(folder, filename)) as json_file:

            
                    eventlist=[]

                    data = json.load(json_file)
                    df=movementTracker.df_from_json(data) 

                for j in ['box1', 'box2', 'obj1', 'obj2', 'obj3', 'obj4']:
                    type = j  

                    xi, yi = movementTracker.interaction(df, participant_id, run, type)

                    if xi.size == 0 or yi.size == 0:
                        pass
                    else:
                        o=movementTracker.interaction(df, participant_id, run, type,direction=True)
                        eventlist.append(o)



            
               
                f.write(str(eventlist)+"\n")
                
            f.write("\n")
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

                        
