import p2dtw
import p2dtw_path
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# distance_matrix1,ids1 = p2dtw.main()
distance_matrix,ids = p2dtw_path.main()


# distance_matrix = np.maximum(distance_matrix, distance_matrix.T)
# distance_matrix = distance_matrix / distance_matrix.max()
# distance_matrix = 1 - distance_matrix
# distance_matrix = np.triu(distance_matrix)
# distance_matrix = distance_matrix[~np.all(distance_matrix == 0, axis=1)]


# distance_matrix1 = np.maximum(distance_matrix1, distance_matrix1.T)
# distance_matrix1 = distance_matrix1 / distance_matrix1.max()
# distance_matrix1 = 1 - distance_matrix1
# distance_matrix1 = np.triu(distance_matrix1)
# distance_matrix1 = distance_matrix1[~np.all(distance_matrix1 == 0, axis=1)]


# Z1 = linkage(distance_matrix1, method='average')  
Z = linkage(distance_matrix, method='average')  

plt.figure(figsize=(25, 10))
dendrogram(Z,labels=ids)
plt.title('Hierarchical Clustering Dendrogram based on path')
plt.savefig('/home/erfan/Documents/Puzzle/puzzlesdata/Plots_Text/Hierarchical Clustering Dendrogram based on path puzzle 2.png', dpi=300)
plt.show()

# plt.figure(figsize=(25, 10))
# dendrogram(Z1,labels=ids1)
# plt.title('Hierarchical Clustering Dendrogram based on displacement')
# plt.show()







