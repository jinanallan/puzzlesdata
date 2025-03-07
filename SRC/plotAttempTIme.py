import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data_1= pd.read_csv('./Data/participants_avg_time_1.csv', header=None)
data_2= pd.read_csv('./Data/participants_avg_time_2.csv', header=None)
attempts_1 = pd.read_csv('./Data/rawsum1.csv',  header=None)
attempts_2 = pd.read_csv('./Data/rawsum2.csv'   , header=None)

df= pd.read_csv("./Data/df.csv")

unique_participants = df["participant_id"].unique().tolist()
unique_participants= np.array(unique_participants).astype(int)

print("unique_participants", unique_participants)

#turn the data into an array
data_1 = data_1.to_numpy()
data_2 = data_2.to_numpy()
attempts_1 = attempts_1.to_numpy()
attempts_2 = attempts_2.to_numpy()

#remove the 2,8,10 th element from the array
data_1 = np.delete(data_1, [1,7,9])
data_2 = np.delete(data_2, [1,7,9])
attempts_1 = np.delete(attempts_1, [1,7,9])
attempts_2 = np.delete(attempts_2, [1,7,9])
#remove the 2,8,10 th element from unique_participants as well
unique_participants = np.delete(unique_participants, [1,7,9])

#plot the data
plt.figure(figsize=(20,10))
plt.suptitle('Avg Best time solved over puzzles vs Number of attempt (When a puzzle not solved: replace with the max time for that puzzle in that run) ', fontsize=20)

plt.subplot(1, 2, 1)
plt.scatter(-attempts_1, data_1)
plt.xlabel('Avg Number of attempts over puzzles')
plt.ylabel('Avg Best time solved over puzzles[min]')
plt.title('Run 1')
plt.grid()
#make the x and y same scale
plt.axis('equal')

#add a regression line
# Compute and add a regression line
m, b = np.polyfit(-attempts_1.flatten(), data_1.flatten(), 1)
plt.plot(-attempts_1, m * -attempts_1 + b, color='red')

plt.subplot(1, 2, 2)
plt.scatter(-attempts_2, data_2)
plt.xlabel('Avg Number of attempts over puzzles')
plt.ylabel('Avg Best time solved over puzzles [min]')
plt.title('Run 2')
plt.grid()
#make the x and y same scale
plt.axis('equal')
#add a regression line
# Compute and add a regression line
m, b = np.polyfit(-attempts_2.flatten(), data_2.flatten(), 1)
plt.plot(-attempts_2, m * -attempts_2 + b, color='red')

# Compute the Euclidean distance from the origin for each point
distances_1 = np.sqrt(data_1**2 + attempts_1**2)
distances_2 = np.sqrt(data_2**2 + attempts_2**2)

# Save the unique participant and corresponding distances to separate files
np.savetxt('./Data/participants_distances_1.csv', np.column_stack((unique_participants, distances_1)), fmt='%s', delimiter=',')
np.savetxt('./Data/participants_distances_2.csv', np.column_stack((unique_participants, distances_2)), fmt='%s', delimiter=',')

# plt.show()
plt.savefig('./Data/avg_best_time_vs_attempt.png')
