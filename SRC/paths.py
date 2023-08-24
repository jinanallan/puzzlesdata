import numpy as np
import os
import json
import matplotlib.pyplot as plt
import movementTracker
import HMPlotter
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
    elif type=='free':
        return (0,0,0,c)
    
def main():
     # folder = input("Enter the folder path: ")
    for pilot in [3,4]:
        folder = './Data/Pilot{}/Ego-based'.format(pilot)
        # desiredpuzzle = int(input("Enter the puzzle number: "))
        for filename in os.listdir(folder):
            if filename.endswith('.json'):
                participant_id, run, puzzle, attempt = HMPlotter.use_regex(filename)
                # print(participant_id, run, puzzle, attempt)
                print(participant_id, run, puzzle, attempt)
            if  os.path.isfile('./Plots_Text/Path_Plots/'+ str(participant_id)+'_'+ str(run)+'_'+str(puzzle)+'_'+str(attempt)+'.png'):

                if puzzle in [1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26]:         
                    fig, ax = plt.subplots()

                    for j in ['box1', 'box2', 'obj1', 'obj2', 'obj3', 'obj4', 'free']:
                        pl=[]
                        #color dict for each type in rgb
                        type = j   
                        # def color_dict(type):
                        #      for 
                            

                        with open(os.path.join(folder, filename)) as json_file:
                            data = json.load(json_file)
                            df=movementTracker.df_from_json(data)
                            sparce=True
                            xi, yi = movementTracker.interaction(df, participant_id, run, type, sparce=sparce)
                            solved , total_time = movementTracker.interaction(df, participant_id, run, type, solved=True)
                            # print(xi.size, yi.size)
                            
                            imgfolder = './cropped_puzzles_screenshots'
                            fname = os.path.join(imgfolder, 'puzzle'+str(puzzle)+'.png')
                            img = Image.open(fname).convert('L')

                            img = ax.imshow(img, extent=[-2, 2, -2, 2], cmap='gray')

                            if xi.size == 0 or yi.size == 0:
                                continue
                            else:

                                #diffrent colors for each type
                                colors=[coloring(type,i) for i in np.linspace(0.2,1,len(xi))]
                                # cm=mcolors.LinearSegmentedColormap.from_list('mylist', colors, N=len(xi))

                                if type=='free':
                                    s=5
                                else:
                                    s=25
                                ax.scatter(xi, yi,
                                                    s=s,
                                                    c=colors,
                                                    label=type,
                                        marker= ".")
                                
                                plt.xlim(-2, 2)
                                plt.ylim(-2, 2)
                                plt.xlabel('x')
                                plt.ylabel('y')
                                plt.legend()
                                title= f'Participant: {participant_id}, Run: {run}, Puzzle: {puzzle}, Attempt: {attempt}, \n Solved: {solved}, Total time: {total_time} s'.format(participant_id, run, puzzle, attempt, solved, total_time)
                                plt.title(title, fontsize=14)
                    plt.savefig('./Plots_Text/Path_Plots/'+
                                str(participant_id)+'_'+ str(run)+'_'+str(puzzle)+'_'+str(attempt)+'.png', dpi=300)
                    plt.close()

if __name__ == "__main__":
    main()
print("--- %s seconds ---" % (time.time() - start_time))
                        
