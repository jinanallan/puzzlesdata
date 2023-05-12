import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import re
import matplotlib.colors as mcolors
import movementTracker


def use_regex(input_text):
    pattern = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)", re.IGNORECASE)

    match = pattern.match(input_text)
    
    particpants = match.group(3)
    run = match.group(4)
    puzzle_id = match.group(5)
    puzzle_id2 = match.group(6)
    return int(particpants), int(run), int(puzzle_id), int(puzzle_id2)

def main():

    # folder = input("Enter the folder path: ")
    folder = '/home/erfan/Downloads/pnp'
    # desiredpuzzle = int(input("Enter the puzzle number: "))
    for i in [1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26]:
        desiredpuzzle = i
        for j in ['box1', 'box2', 'obj1', 'obj2', 'obj3', 'obj4', 'total', 'free']:
            type = j    

            x = np.array([])
            y = np.array([])

            for filename in os.listdir(folder):
                if filename.endswith('.json'):
                    participant_id, run, puzzle, attempt = use_regex(filename)

                    if puzzle == desiredpuzzle:
                        # Load the JSON file
                        with open(os.path.join(folder, filename)) as json_file:
                            data = json.load(json_file)
                            df=movementTracker.df_from_json(data)

                            xi, yi = movementTracker.interaction(df, participant_id, run, type)

                            if xi.size == 0 or yi.size == 0:
                                pass
                            else:
                                if type in ["total", "free"]:
                                    x = np.append(x, xi)
                                    y = np.append(y, yi)
                                elif "Attach "+type or "Release "+ type in movementTracker.get_descriptions(df):
                                    # print("test")
                                    x = np.append(x, xi)
                                    y = np.append(y, yi)
                           
            #check if x and y are not empty
            if len(x) != 0:
                colors = [(1,0,0,c) for c in np.linspace(0,1,100)]
                cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=100)

                fig, ax = plt.subplots()

                #the image of the desired puzzle
                imgfolder = '/home/erfan/Documents/Puzzle/puzzlesdata/cropped_puzzles_screenshots'
                fname = os.path.join(imgfolder, 'puzzle'+str(desiredpuzzle)+'.png')
                img = Image.open(fname).convert('L')
                img = ax.imshow(img, extent=[-2, 2, -2, 2], cmap='gray')

                plt.hist2d(x, y, bins=(45, 45),cmap=cmapred)
                plt.xlim(-2, 2)
                plt.ylim(-2, 2)
                plt.colorbar()
                plt.title(type + ' Interaction Heatmap of the Puzzle ' + str(desiredpuzzle))
                # plt.show()
                #save the plot in the temp folder with same name as the title
                
                fig.savefig(os.path.join('/home/erfan/Documents/Puzzle/puzzlesdata/Plots_Text/HeatMap_Plots', str(desiredpuzzle)+'_'+ type +'.png'), dpi=300)
                plt.close(fig)


if __name__ == '__main__':
    main()
