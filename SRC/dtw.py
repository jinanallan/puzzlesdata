import numpy as np
import os
import json
import movementTracker
import re
import HMPlotter

    
def main():
     # folder = input("Enter the folder path: ")
    folder = '/home/erfan/Downloads/pnp'
   
    # 1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26
    for desired_puzzle in [2]:

        output_file = os.path.join('/home/erfan/Documents/Puzzle/puzzlesdata/Plots_Text/Direction_Text', 'puzzle'+str(desired_puzzle)+'_pos.txt')

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

                        for type in ['box1', 'box2', 'obj1', 'obj2', 'obj3', 'obj4','Glue','Unglue']:

                            # xi, yi = movementTracker.interaction(df, participant_id, run, type)
                            # if xi.size == 0 or yi.size == 0:

                            #     pass

                            # else:

                            o=movementTracker.interaction(df, participant_id, run, type,direction=True, pos=True)
                            eventlist= np.append(eventlist,o)
                    
                        #sortting the list baded on the time of the event
                        eventlist=eventlist[np.argsort(eventlist)]
                        eventlist=[i.split("_")[1] for i in eventlist]

                        if len(eventlist)!=0:
                            f.write(str(eventlist)+"\n")

                        solved_stats=movementTracker.interaction(df, participant_id, run,type="total",solved=True)
                        f.write("Solved: " + str(solved_stats)+"\n")  
                        f.write("\n")
   
    with open(output_file, 'r') as f:
        text = f.readlines()
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
       


if __name__ == "__main__":
    main()

                        
