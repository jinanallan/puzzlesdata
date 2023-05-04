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
    # desiredpuzzle = int(input("Enter the puzzle number: "))
    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            participant_id, run, puzzle, attempt = HMPlotter.use_regex(filename)
        if puzzle in [1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26]:
            
         
            fig, ax = plt.subplots()

            for j in ['box1', 'box2', 'obj1', 'obj2', 'obj3', 'obj4', 'free']:
                #color dict for each type in rgb
                type = j   
                # def color_dict(type):
                #      for 
                     
                # c=0
                # color_dict = {'box1':(0,0,1,c), 'box2':(1,0,1,c), 'obj1':(0,1,1,c), 'obj2':(1,1,1,c), 'obj3':(1,0,0.5,c), 'obj4':(0.5,0,1,c), 'free':(0.5,0.5,1,c)}

                with open(os.path.join(folder, filename)) as json_file:
                    data = json.load(json_file)
                    df=movementTracker.df_from_json(data)

                    xi, yi = movementTracker.interaction(df, participant_id, run, type)

                    if xi.size == 0 or yi.size == 0:
                                pass
                    else:
                        
                        imgfolder = 'cropped'
                        fname = os.path.join(imgfolder, 'puzzle'+str(puzzle)+'.png')
                        img = Image.open(fname).convert('L')

                        img = ax.imshow(img, extent=[-2, 2, -2, 2], cmap='gray')

                        #diffrent colors for each type
                        colors = [(0,0,1,c) for c in np.linspace(0,1,len(xi))]

                        cm = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)

                        if type=='free':
                             s=5
                        else:
                            s=25
                        typescatter=ax.scatter(xi, yi,
                                               s=s,
                                               c=colors,
                                               cmap=cm, 
                                               label=type,
                                   marker= ".")
                        plt.xlim(-2, 2)
                        plt.ylim(-2, 2)
                        plt.xlabel('x')
                        plt.ylabel('y')
            plt.legend()
            plt.title('Participant:'+str(participant_id)+' Puzzle: '+str(puzzle)+' Attempt:'+str(attempt)+' Run:'+str(run))
            plt.savefig('plots/'+str(puzzle)+'_'+str(participant_id)+'_'+str(attempt)+'_'+str(run)+'.png')
            plt.close()

if __name__ == "__main__":
    main()

                        
