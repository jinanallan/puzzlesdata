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
    for desired_puzzle in [5]:

            for filename in sorted(os.listdir(folder)):
                if filename.endswith('.json'):
                    participant_id, run, puzzle, attempt = HMPlotter.use_regex(filename)

                                        

                    if desired_puzzle == puzzle :
                        

                        with open(os.path.join(folder, filename)) as json_file:

                            data = json.load(json_file)
                            df=movementTracker.df_from_json(data)
                            
                            solved_stats=movementTracker.interaction(df, participant_id, run,type="total",solved=True)
                            # print(solved_stats)
                            # print(type(solved_stats))
                            if solved_stats== "True":
                                ids.append(str(participant_id) + "_" + str(run) + "_" +str(puzzle) + "_" +str(attempt))

                                x,y,description=wholeSequence.interaction(df, participant_id, run)
                                transformed_description=np.zeros((len(description),2))
                                for i in range(len(description)):

                                    if description[i]=='box1' :
                                        transformed_description[i][:]=[0,1]

                                    elif description[i]=='free' :
                                        transformed_description[i][:]=[0,0]

                                    elif description[i]=='obj2' or description[i]=='obj3':
                                        transformed_description[i][:]=[1,0]

                                    elif description[i]=='obj4' or description[i]=='obj1':
                                        transformed_description[i][:]=[-1,0]

                                    else:
                                        transformed_description[i][:]=[0,-1]


                            # print(description)
                                # print(transformed_description.shape)
                                td1=transformed_description[:,0]
                                td2=transformed_description[:,1]
                                # print(x.shape)
                                # print(y.shape)
                                # print(td1.shape)
                                # print(td2.shape)

                                sequence=np.array([x,y,td1,td2],dtype=np.double)
                                sequence=sequence.T
                                sequences.append(sequence)

            ds = dtw.distance_matrix_fast(sequences)
            distance_matrix=ds
            # print(ds)
           

    # distance_matrix=np.zeros((len(sequences),len(sequences)))
    # for i in range(len(sequences)):
    #     for j in range(len(sequences)):
    #         if i!=j and i<j:
    #             querry=sequences[i]
    #             reference=sequences[j]
    #             d=dtw_ndim.distance_fast(querry, reference, max_step=10, compact=True)
    #             # print(i,j,d)
    #             distance_matrix[i][j]=d
    #             distance_matrix[j][i]=d
    return distance_matrix,ids
    
    



if __name__ == "__main__":
    main()
