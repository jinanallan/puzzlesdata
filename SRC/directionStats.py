#the goal is to generate stacked bar plots from the txt files in the Direction folder
# count the number of participants who took box1 NW as their first move


import os
import numpy as np
import re
import matplotlib.pyplot as plt

def info(input_text):
    pattern = re.compile(r"Participant:([0-9]+) for the Puzzle:([0-9])+ in the Attempt:([0-9]+) and run:([0-9]+) took follwing movements:", re.IGNORECASE)
    
    match = pattern.match(input_text)
    
    particpants = match.group(1)
    run = match.group(2)
    # puzzle_id = match.group(2)
    attempt = match.group(4)
    return particpants, run, attempt

def actions(action_list):
    #convert action_list from string to list
    action_list = action_list.replace("[", "")
    action_list = action_list.replace("]", "")
    action_list = action_list.replace("'", "")
    action_list = action_list.replace("(", "[")
    action_list = action_list.replace(")", "]")
    action_list = action_list.replace(",", " ")
    action_list = action_list.split(",")
    return action_list

file = '/home/erfan/Documents/Puzzle/puzzlesdata/Plots_Text/Direction_Text/puzzle2.txt'

with open(file, 'r') as text:
    text = text.readlines()

    max_length = 0
    for line in text:
        if line.startswith("["):
            action_list = actions(line)
            # print(action_list)
            if len(action_list) > max_length:
                max_length = len(action_list)

    i = np.zeros(max_length)
    j = np.zeros(max_length)
    z = np.zeros(max_length)
    k = np.zeros(max_length)
    

    for line in text:
        if line.startswith("Participant"):
            (particpants, run, attempt) = info(line)
        elif line.startswith("["):
            action_list = actions(line)
            
            for t in range(max_length):
                try:
                    action_t = action_list[t]
                    action_t = action_t.strip()
                    action_t = action_t.split(" ")

                    if action_t[0] == "box1" and action_t[1] == "NW":
                        i[t] += 1
                    elif action_t[0] == "box1" and action_t[1] == "W":
                        j[t] += 1
                    else:   
                        z[t] += 1
                except:
                    continue
    
    #get the max length of the lists
    max_length = max(len(i), len(j), len(z))
    order = np.arange(max_length)
    #plot a stacked bar plot with i, j, z for each time step
    plt.bar(order+1, i, label='box 1 NW')
    plt.bar(order+1, j, bottom=i, label='box1 W')
    plt.bar(order+1, z, bottom=i+j, label='box1 *')
    # plt.bar(order+1, k, bottom=i+j+z, label='other')
    plt.legend()
    plt.xlabel('Ordinal number of the interaction')
    plt.ylabel('Number of participants')
    plt.title('Puzzle 2 direction statistics')
    plt.show()










