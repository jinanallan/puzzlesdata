#get the unique puzzels in the folder pnp
import os
import movementTracker 
import numpy as np
from HMPlotter import use_regex
folder = '/home/erfan/Downloads/pnp'
puzzels = []
for filename in os.listdir(folder):
    if filename.endswith('.json'):
        participant_id, run, puzzel, attempt = use_regex(filename)
        if puzzel not in puzzels:
            puzzels.append(puzzel)
puzzels.sort()
#sort the puzzels
print(puzzels)

#get all the existing type of interactions in each puzzel
def validtype(df,puzzle, )