import re
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def use_regex_frames(input_text):
    pattern = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_frames", re.IGNORECASE)

    match = pattern.match(input_text)
    
    particpants = match.group(3)
    run = match.group(4)
    puzzle_id = match.group(5)
    attempt = match.group(6)
    return int(particpants), int(run), int(puzzle_id), int(attempt)

def use_regex(input_text):
    pattern = re.compile(r"([0-9]{4}-[0-9]{2}-[0-9]{2})-([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)", re.IGNORECASE)

    match = pattern.match(input_text)
    
    particpants = match.group(3)
    run = match.group(4)
    puzzle_id = match.group(5)
    attempt = match.group(6)
    return int(particpants), int(run), int(puzzle_id), int(attempt)

frame_folder= "./Data/Frames/"
pnp_folder= "./Data/pnp/"

# list all files in the folders by using those functin above and visulaize them 
frame_files = os.listdir(frame_folder)
pnp_files = os.listdir(pnp_folder)

df= pd.DataFrame(columns=["participant_id", "run", "puzzle_id", "attempt", "pnp", "frame"])
#create a of all the files in the folder as pandas dataframe
for file in frame_files:
    if file.endswith(".json"):
        # print(file)
        particpants, run, puzzle_id, attempt = use_regex_frames(file)
        #append is not dataframe attribute
        new_row = {"participant_id": particpants, "run": run, "puzzle_id": puzzle_id, "attempt": attempt, "pnp": 0, "frame": file}
        df.loc[len(df)] = new_row
for file in pnp_files:
    if file.endswith(".json"):
        # print(file)
        particpants, run, puzzle_id, attempt = use_regex(file)
        # df = df.append({"participant_id": particpants, "run": run, "puzzle_id": puzzle_id, "attempt": attempt, "pnp": 1, "frame": file}, ignore_index=True)
        new_row = {"participant_id": particpants, "run": run, "puzzle_id": puzzle_id, "attempt": attempt, "pnp": 1, "frame": file}
        df.loc[len(df)] = new_row
#save the dataframe as csv file
df.to_csv("./Data/df.csv", index=False)

