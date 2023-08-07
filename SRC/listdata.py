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

frame_files = os.listdir(frame_folder)
pnp_files = os.listdir(pnp_folder)

df= pd.DataFrame(columns=["participant_id", "run", "puzzle_id", "attempt", "file", "frame file"])

for file in pnp_files:
    if file.endswith(".json"):

        particpants, run, puzzle_id, attempt = use_regex(file)
    
        new_row = {"participant_id": particpants, "run": run, "puzzle_id": puzzle_id, "attempt": attempt, "file": file, "frame file": None}

        df.loc[len(df)] = new_row

for file in frame_files:
    if file.endswith(".json"):

        particpants, run, puzzle_id, attempt = use_regex_frames(file)

        df.loc[(df["participant_id"] == particpants) & (df["run"] == run) &
                (df["puzzle_id"] == puzzle_id) & (df["attempt"] == attempt), "frame file"] =file

df = df.sort_values(by=["participant_id", "run", "puzzle_id", "attempt"])

df.to_csv("./Data/df.csv", index=False)