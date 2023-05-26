import numpy as np
import os
import json
import matplotlib.pyplot as plt
import movementTracker
import HMPlotter
import wholeSequence
from dtaidistance import dtw
from dtaidistance import dtw_ndim
    
def main():
     # folder = input("Enter the folder path: ")
    folder = '/home/erfan/Downloads/pnp'
   
    # 1, 2, 3, 4,5, 6, 21, 22, 23, 24, 25, 26
    sequences=[]
    ids=[]
    for desired_puzzle in [2]:

            for filename in sorted(os.listdir(folder)):
                if filename.endswith('.json'):
                    participant_id, run, puzzle, attempt = HMPlotter.use_regex(filename)

                                        

                    if desired_puzzle == puzzle :
                        
                        ids.append(str(participant_id) + "_" + str(run) + "_" +str(puzzle) + "_" +str(attempt))

                        with open(os.path.join(folder, filename)) as json_file:

                            data = json.load(json_file)
                            df=movementTracker.df_from_json(data)
                            
                            solved_stats=movementTracker.interaction(df, participant_id, run,type="total",solved=True)
                            

                            x,y,description=wholeSequence.interaction(df, participant_id, run)

                            for i in range(len(description)):

                                if description[i]=='box1' :
                                    description[i]=1

                                elif description[i]=='free' :
                                    description[i]=0

                                else:
                                    description[i]=0.5

                        # print(description)
                            sequence=np.array([x,y,description],dtype=np.double)
                            sequence=sequence.T
                            sequences.append(sequence)

            # ds = dtw.distance_matrix_fast(sequences,window=1)
            # print(ds)
           

    distance_matrix=np.zeros((len(sequences),len(sequences)))
    for i in range(len(sequences)):
        for j in range(len(sequences)):
            if i!=j and i<j:
                querry=sequences[i]
                reference=sequences[j]
                d=dtw_ndim.distance_fast(querry, reference, max_step=10)
                # print(i,j,d)
                distance_matrix[i][j]=d
                distance_matrix[j][i]=d
    return distance_matrix,ids
    
    



if __name__ == "__main__":
    main()
