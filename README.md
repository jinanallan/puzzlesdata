![alt text](https://github.com/jinanallan/puzzlesdata/blob/main/test.png?raw=True)
<<<<<<< HEAD
![alt text](https://github.com/jinanallan/puzzlesdata/blob/main/Data/timeDistribution.png?raw=True)
=======
>>>>>>> 26e08dc8b80974b9fb84cb7b6d217100e2716a3a
![alt text](https://github.com/jinanallan/puzzlesdata/blob/main/Data/Distribution.png?raw=True)

# clustering similar solutions
to run clustering for the desired puzzle:
```
 cd puzzlesdata/SRC
 python3 strategyClastur.py
```
Enter the puzzle number: 

Enter the number of clusters: 

Enter the path of the folder containing pnp the json files: 


The results is saved in ```/puzzlesdata/Plots_Text/clustering/puzzle {puzzleNumber}```

note:
* clustering methode is based on [scipy.cluster.hierarchy.linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) and "ward" methode for cluster distances
* distance between each solution is calculated by [dtw](https://dtaidistance.readthedocs.io/en/latest/usage/dtw.html#dtw-between-multiple-time-series)
* there are parameters in each of the methos as well as number of clusters that need to be determined 
* the labels of interaction types are transformed by [oneHotEncoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) so that each interaction type has same distance to another one

# earlier results
as requsted [here](https://github.com/svetlanalevit/puzzle-scenes/projects/1#card-88830295):  

The number of participants who solved all the puzzles: 12 out of 15  
The participants who eventually solved all the puzzles:  [31 33 34 36 39 41 42 43 45 46 47 48]  
  
The participants who did not solve all the puzzles:  [35 37 44]  
The participant 35 solved 26 puzzles out of 27  
The puzzles that the participant 35 did not solve are: [27]  
The participant 37 solved 25 puzzles out of 27  
The puzzles that the participant 37 did not solve are: [10 27]  
The participant 44 solved 20 puzzles out of 27  
The puzzles that the participant 44 did not solve are: [ 7  8 11 21 24 25 27]  

With [stats](https://github.com/jinanallan/puzzlesdata/blob/main/stats.py) , it is feasible now to request any form of general stats of the data needed
