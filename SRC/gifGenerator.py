from PIL import Image
import os
import re

def gif(desired_puzzle, ids=[],frameBased=True, includeEgo=False, attachment=False):

    pathimage_folder = "./Plots_Text/Path_Plots"
    if frameBased:
        pathimage_folder = "./Plots_Text/Path_Plots/frameBased"
        if includeEgo:
            pathimage_folder = "./Plots_Text/Path_Plots/frameBased/includeEgo"
        if attachment:
            pathimage_folder = "./Plots_Text/Path_Plots/frameBased/pathAttachment"
            if includeEgo:
                pathimage_folder = "./Plots_Text/Path_Plots/frameBased/pathAttachment/includeEgo"

    png_files = [f for f in os.listdir(pathimage_folder) if f.endswith('.png')]

    def use_regex(input_text):
        pattern = re.compile(r"([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)", re.IGNORECASE)

        match = pattern.match(input_text)
        
        particpants = match.group(1)
        run = match.group(2)
        puzzle_id = match.group(3)
        attempt = match.group(4)
        return int(particpants), int(run), int(puzzle_id), int(attempt)
    
    if len(ids) == 0:
        puzzle_images = []
        for filename in png_files:
            participant_id, run, puzzle, attempt = use_regex(filename)
            if puzzle == desired_puzzle:
                puzzle_images.append(filename)

        #add the relative path to the images
        puzzle_images = [os.path.join(pathimage_folder, f) for f in puzzle_images]
    else:
        puzzle_images = [os.path.join(pathimage_folder, f) for f in ids]
        puzzle_images = [f + ".png" for f in puzzle_images]

    # Open the first image

    first_image = Image.open(puzzle_images[0])
    

    # Create a list to store frames
    frames = []

    # Append each subsequent image as a frame
    for file in puzzle_images[1:]:
        frame = Image.open(file)
        frames.append(frame)


    # first_image.save(f'animated{desired_puzzle}.gif', 
    #                 format='GIF', append_images=frames,
    #                 save_all=True, duration=50*len(puzzle_images), loop=0)
    return first_image, frames

        
